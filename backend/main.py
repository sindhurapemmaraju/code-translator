"""Code Translation Assistant - FastAPI backend (Gemini or Ollama).

Endpoints:
  POST /api/translate  -> translate code (chunked for large files)
  POST /api/analyze    -> explanation + error detection (chunked, parallel)
"""
import asyncio, json, os, re, shutil, subprocess, sys, tempfile, time
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

load_dotenv()
PROVIDER = os.getenv("PROVIDER", "gemini").lower()  # "gemini" or "ollama"
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
# Comma-separated backups, tried in order when the main model is busy or missing
GEMINI_FALLBACKS = [m.strip() for m in os.getenv("GEMINI_FALLBACKS", "gemini-3.5-flash-lite").split(",") if m.strip()]
RETRY_ROUNDS = int(os.getenv("RETRY_ROUNDS", "4"))
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
CHUNK_CHARS = int(os.getenv("CHUNK_CHARS", "6000"))
CONCURRENCY = int(os.getenv("CONCURRENCY", "2"))
MAX_CHARS = 500_000
app = FastAPI(title="Code Translation Assistant")


class TranslateReq(BaseModel):
    code: str = Field(min_length=1, max_length=MAX_CHARS)
    source: str = "auto-detect"
    target: str


class AnalyzeReq(BaseModel):
    code: str = Field(min_length=1, max_length=MAX_CHARS)
    language: str = "auto-detect"


# ---------- chunking ----------
def chunk_code(code: str, limit: int = CHUNK_CHARS):
    """Split on top-level boundaries (blank line followed by an unindented line)
    so functions/classes are never cut in half. Returns [(start_line, text)]."""
    lines = code.splitlines()
    chunks, cur, cur_len, start = [], [], 0, 1
    for i, line in enumerate(lines):
        boundary = (
            cur and cur_len >= limit * 0.6
            and line.strip() and not line[0].isspace()
            and not line.lstrip().startswith(("}", ")", "]", "else", "elif", "except", "finally", "catch"))
            and i > 0 and not lines[i - 1].strip()
        )
        if boundary or (cur_len + len(line) > limit * 1.5 and cur):  # hard cap
            chunks.append((start, "\n".join(cur)))
            cur, cur_len, start = [], 0, i + 1
        cur.append(line)
        cur_len += len(line) + 1
    if cur:
        chunks.append((start, "\n".join(cur)))
    return chunks


# ---------- LLM helpers ----------
RETRYABLE = (404, 429, 500, 503)  # try the next model / wait and retry


async def _gemini(http, model, system, user, max_tokens):
    r = await http.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        headers={"x-goog-api-key": GEMINI_KEY},
        json={"system_instruction": {"parts": [{"text": system}]},
              "contents": [{"role": "user", "parts": [{"text": user}]}],
              "generationConfig": {"maxOutputTokens": max_tokens}})
    if r.status_code in RETRYABLE:
        return None, r.status_code
    r.raise_for_status()
    parts = r.json()["candidates"][0]["content"]["parts"]
    return "".join(p.get("text", "") for p in parts), 200


async def _ollama(http, system, user):
    r = await http.post(f"{OLLAMA_URL}/api/chat", json={
        "model": OLLAMA_MODEL, "stream": False,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
    if r.status_code in RETRYABLE:
        return None, r.status_code
    r.raise_for_status()
    return r.json()["message"]["content"], 200


async def ask(system: str, user: str, max_tokens: int = 8000) -> str:
    """Try the main model, then each fallback; if all are busy, wait and repeat."""
    if PROVIDER == "gemini" and not GEMINI_KEY:
        raise HTTPException(500, "GEMINI_API_KEY is missing. Add it to backend/.env and restart.")
    models = [GEMINI_MODEL] + [m for m in GEMINI_FALLBACKS if m != GEMINI_MODEL] if PROVIDER == "gemini" else ["ollama"]
    delay, last = 3, "unknown"
    for rnd in range(RETRY_ROUNDS):
        for model in models:
            try:
                async with httpx.AsyncClient(timeout=300) as http:
                    if PROVIDER == "ollama":
                        text, status = await _ollama(http, system, user)
                    else:
                        text, status = await _gemini(http, model, system, user, max_tokens)
            except httpx.HTTPStatusError as e:  # e.g. 400/401/403: retrying will not help
                raise HTTPException(502, f"Model request failed ({model}): {e}")
            except httpx.HTTPError as e:  # network hiccup
                last = f"{model}: {e}"
                continue
            if text is not None:
                return text
            last = f"{model}: HTTP {status}"
        if rnd < RETRY_ROUNDS - 1:
            await asyncio.sleep(delay)
            delay = min(delay * 2, 20)
    raise HTTPException(503, f"All models are busy or unavailable right now (last: {last}). Try again in a minute.")


def strip_fences(text: str) -> str:
    m = re.search(r"```[\w+#-]*\n(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip("\n")


def parse_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        try:
            return json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            return {"explanation": text, "errors": []}  # keep the raw text rather than failing


TRANSLATE_SYS = (
    "You are an expert polyglot engineer translating code between languages. "
    "Produce idiomatic, compilable/runnable code in the target language, preserving behavior. "
    "Map libraries to standard equivalents and add a short comment where semantics differ. "
    "Output ONLY one fenced code block, no prose."
)


async def translate_code(code: str, source: str, target: str):
    chunks = chunk_code(code)
    out, tail = [], ""
    for idx, (_, text) in enumerate(chunks):
        prompt = (
            f"Source language: {source}\nTarget language: {target}\n"
            f"This is part {idx + 1} of {len(chunks)} of a larger file.\n"
        )
        if idx:
            prompt += (
                "Already translated (tail, for naming consistency). Do NOT repeat imports/"
                f"includes or anything already defined:\n```\n{tail}\n```\n"
            )
        else:
            prompt += "Include all needed imports/includes at the top.\n"
        if idx < len(chunks) - 1:
            prompt += "The file continues after this part, so do not add a main/entry point yet unless the source has one here.\n"
        prompt += f"\nTranslate this part:\n```\n{text}\n```"
        piece = strip_fences(await ask(TRANSLATE_SYS, prompt))
        out.append(piece)
        tail = "\n".join(piece.splitlines()[-40:])
    return "\n\n".join(out), len(chunks)


@app.post("/api/translate")
async def translate(req: TranslateReq):
    text, n = await translate_code(req.code, req.source, req.target)
    return {"translated_code": text, "chunks": n}


ANALYZE_SYS = (
    "You are a senior code reviewer. Respond with ONLY valid JSON of this shape:\n"
    '{"language": str, "summary": str, "explanation": str, '
    '"errors": [{"line": int, "severity": "error|warning|info", "message": str, "fix": str}]}\n'
    "Each source line is prefixed with its absolute line number as 'N| '. Report line numbers using those. "
    "Find syntax errors, runtime bugs, logic errors, unsafe patterns, and missing imports. "
    "Do not invent problems; if the code is fine, return an empty errors list. "
    "'explanation' should be a clear step-by-step walkthrough in Markdown for a learner."
)


async def analyze_chunk(sem, start, text, language, idx, total):
    numbered = "\n".join(f"{start + i}| {l}" for i, l in enumerate(text.splitlines()))
    prompt = (
        f"Language: {language}\nThis is part {idx + 1} of {total} of a file; "
        f"code referenced outside this part may be defined elsewhere.\n\n{numbered}"
    )
    async with sem:
        return parse_json(await ask(ANALYZE_SYS, prompt, 6000))


@app.post("/api/analyze")
async def analyze(req: AnalyzeReq):
    chunks = chunk_code(req.code)
    sem = asyncio.Semaphore(CONCURRENCY)
    results = await asyncio.gather(*[
        analyze_chunk(sem, s, t, req.language, i, len(chunks)) for i, (s, t) in enumerate(chunks)
    ])
    errors = sorted(
        (e for r in results for e in r.get("errors", []) if isinstance(e, dict)),
        key=lambda e: e.get("line", 0) if isinstance(e.get("line"), int) else 0,
    )
    return {
        "language": next((r["language"] for r in results if r.get("language")), req.language),
        "summary": " ".join(r.get("summary", "") for r in results).strip(),
        "explanation": "\n\n".join(r.get("explanation", "") for r in results if r.get("explanation")),
        "errors": errors,
        "chunks": len(chunks),
    }



# ---------- running code ----------
# WARNING: this executes code on the machine hosting the server. It is fine for a local
# project you run yourself; do NOT expose it publicly without a real sandbox (e.g. Docker).
ENABLE_RUN = os.getenv("ENABLE_RUN", "true").lower() == "true"
RUN_TIMEOUT = int(os.getenv("RUN_TIMEOUT", "10"))
MAX_OUT = 20_000
EXE = ".exe" if os.name == "nt" else ""


def _have(*cmds):
    return all(shutil.which(c) for c in cmds)


RUNTIMES = {
    "Python": lambda: True,
    "JavaScript": lambda: _have("node"),
    "Java": lambda: _have("javac", "java"),
    "C": lambda: _have("gcc"),
    "C++": lambda: _have("g++"),
    "Go": lambda: _have("go"),
    "PHP": lambda: _have("php"),
    "Ruby": lambda: _have("ruby"),
    "Bash": lambda: _have("bash"),
}
_KEEP = {"PATH", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC", "PATHEXT", "TEMP", "TMP",
         "HOME", "USERPROFILE", "LOCALAPPDATA", "APPDATA", "LANG", "JAVA_HOME", "GOPATH", "GOCACHE", "GOROOT"}


def _safe_env():
    """Pass only what compilers need, so the code cannot read GEMINI_API_KEY."""
    return {k: v for k, v in os.environ.items() if k.upper() in _KEEP}


def _exec(cmd, cwd, stdin_text, timeout):
    t0 = time.time()
    base = {"stdout": "", "stderr": "", "exit_code": None, "timed_out": False}
    try:
        p = subprocess.run(cmd, cwd=cwd, input=stdin_text, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout, env=_safe_env())
        base.update(stdout=p.stdout[:MAX_OUT], stderr=p.stderr[:MAX_OUT], exit_code=p.returncode)
    except subprocess.TimeoutExpired as e:
        dec = lambda x: (x.decode("utf-8", "replace") if isinstance(x, bytes) else (x or ""))[:MAX_OUT]
        base.update(stdout=dec(e.stdout), stderr=dec(e.stderr), timed_out=True)
    except FileNotFoundError:
        base.update(stderr=f"Command not found: {cmd[0]}. Install it and make sure it is on PATH.", exit_code=127)
    base["seconds"] = round(time.time() - t0, 2)
    return base


def run_code(lang: str, code: str, stdin_text: str):
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        def write(name):
            (Path(d) / name).write_text(code, encoding="utf-8")
            return name
        out = "prog" + EXE
        if lang == "Python":
            return {"stage": "run", **_exec([sys.executable, write("main.py")], d, stdin_text, RUN_TIMEOUT)}
        if lang == "JavaScript":
            return {"stage": "run", **_exec(["node", write("main.js")], d, stdin_text, RUN_TIMEOUT)}
        if lang == "PHP":
            return {"stage": "run", **_exec(["php", write("main.php")], d, stdin_text, RUN_TIMEOUT)}
        if lang == "Ruby":
            return {"stage": "run", **_exec(["ruby", write("main.rb")], d, stdin_text, RUN_TIMEOUT)}
        if lang == "Bash":
            return {"stage": "run", **_exec(["bash", write("main.sh")], d, stdin_text, RUN_TIMEOUT)}
        if lang == "Go":
            return {"stage": "run", **_exec(["go", "run", write("main.go")], d, stdin_text, RUN_TIMEOUT + 20)}
        if lang in ("C", "C++"):
            src = write("main.c" if lang == "C" else "main.cpp")
            comp = _exec(["gcc" if lang == "C" else "g++", src, "-o", out] + (["-lm"] if lang == "C" else []), d, "", 60)
            if comp["exit_code"] != 0:
                return {"stage": "compile", **comp}
            return {"stage": "run", **_exec([str(Path(d) / out)], d, stdin_text, RUN_TIMEOUT)}
        if lang == "Java":
            m = re.search(r"public\s+(?:final\s+|abstract\s+)*class\s+(\w+)", code) or re.search(r"class\s+(\w+)", code)
            name = m.group(1) if m else "Main"
            write(name + ".java")
            comp = _exec(["javac", name + ".java"], d, "", 60)
            if comp["exit_code"] != 0:
                return {"stage": "compile", **comp}
            return {"stage": "run", **_exec(["java", "-cp", d, name], d, stdin_text, RUN_TIMEOUT)}
    raise HTTPException(400, f"Running {lang} is not supported.")


class RunReq(BaseModel):
    code: str = Field(min_length=1, max_length=200_000)
    language: str
    stdin: str = ""


@app.get("/api/runtimes")
def runtimes():
    return {"enabled": ENABLE_RUN, "timeout": RUN_TIMEOUT, "languages": {k: bool(v()) for k, v in RUNTIMES.items()}}


@app.post("/api/run")
async def run(req: RunReq):
    if not ENABLE_RUN:
        raise HTTPException(403, "Running code is disabled (ENABLE_RUN=false).")
    if req.language not in RUNTIMES:
        raise HTTPException(400, f"Running {req.language} is not supported. Supported: {', '.join(RUNTIMES)}.")
    if not RUNTIMES[req.language]():
        raise HTTPException(400, f"{req.language} is not installed on the server (compiler/interpreter not found on PATH).")
    return await asyncio.to_thread(run_code, req.language, req.code, req.stdin)



# ---------- auto-fix loop: translate -> run -> compare -> ask model to fix -> repeat ----------
AUTOFIX_MAX_CHARS = int(os.getenv("AUTOFIX_MAX_CHARS", "30000"))

FIX_SYS = (
    "You repair code translations. You get the original program, its translation, and what went wrong "
    "when the translation was compiled or run. Return the COMPLETE corrected translation in the target "
    "language. Preserve the original's exact behavior and printed output format. Change only what is needed. "
    "Output ONLY one fenced code block, no prose."
)


class AutofixReq(BaseModel):
    code: str = Field(min_length=1, max_length=MAX_CHARS)
    source: str = "auto-detect"
    target: str
    stdin: str = ""
    max_attempts: int = Field(3, ge=0, le=5)  # number of fix rounds after the first translation


def _norm(t: str) -> str:
    return "\n".join(l.rstrip() for l in t.replace("\r\n", "\n").splitlines()).strip()


def judge(res: dict, ref):
    """Return (passed, reason) for a run of the translated program."""
    if res["stage"] == "compile":
        return False, "Compile error:\n" + res["stderr"]
    if res["timed_out"]:
        return False, f"The program timed out after {RUN_TIMEOUT}s (infinite loop, or it waits for input that was not provided)."
    if ref is None:
        if res["exit_code"] == 0:
            return True, "Compiled and ran without errors (original could not be run, so output was not compared)."
        return False, f"Runtime error (exit code {res['exit_code']}):\n{res['stderr']}"
    if ref["timed_out"] or ref["exit_code"] != 0:
        return True, "Compiled and ran. The original program itself fails, so output was not compared."
    if res["exit_code"] != 0:
        return False, f"Runtime error (exit code {res['exit_code']}):\n{res['stderr']}"
    if _norm(res["stdout"]) != _norm(ref["stdout"]):
        return False, f"Output differs from the original.\nExpected:\n{ref['stdout'][:1500]}\nGot:\n{res['stdout'][:1500]}"
    return True, "Output matches the original."


async def detect_language(code: str) -> str:
    r = await ask("Identify the programming language. Reply with ONLY one name from this list, or 'Other': "
                  + ", ".join(RUNTIMES), code[:3000], 1000)
    r = r.strip().strip(".`'\"")
    return next((k for k in RUNTIMES if k.lower() == r.lower()), "Other")


async def autofix_events(req: AutofixReq):
    tgt = req.target
    yield {"event": "status", "message": "Translating..."}
    code, chunks = await translate_code(req.code, req.source, tgt)
    yield {"event": "translated", "code": code, "chunks": chunks}

    if not ENABLE_RUN or tgt not in RUNTIMES or not RUNTIMES[tgt]():
        yield {"event": "done", "passed": None, "code": code, "fixes": 0,
               "note": f"Could not verify: {tgt} cannot be run on this computer (not installed or not supported)."}
        return

    src = req.source
    if src == "auto-detect":
        yield {"event": "status", "message": "Detecting the source language..."}
        src = await detect_language(req.code)
    ref = None
    if src in RUNTIMES and RUNTIMES[src]():
        yield {"event": "status", "message": f"Running the original {src} program..."}
        ref = await asyncio.to_thread(run_code, src, req.code, req.stdin)
    yield {"event": "reference", "language": src, "result": ref}

    for attempt in range(req.max_attempts + 1):
        yield {"event": "status", "message": "Running the translation..." if attempt == 0 else f"Running fix {attempt}..."}
        res = await asyncio.to_thread(run_code, tgt, code, req.stdin)
        passed, reason = judge(res, ref)
        yield {"event": "attempt", "n": attempt, "passed": passed, "reason": reason, "result": res}
        if passed:
            yield {"event": "done", "passed": True, "code": code, "fixes": attempt}
            return
        if attempt == req.max_attempts:
            break
        yield {"event": "status", "message": f"Asking the model to fix it (fix {attempt + 1} of {req.max_attempts})..."}
        prompt = (
            f"Original ({src}):\n```\n{req.code}\n```\n\nCurrent translation ({tgt}):\n```\n{code}\n```\n\n"
            f"Problem when running the translation:\n{reason}\n\n"
            + (f"Program input (stdin):\n{req.stdin}\n\n" if req.stdin else "")
            + "Return the complete corrected translation."
        )
        code = strip_fences(await ask(FIX_SYS, prompt))
        yield {"event": "fixed", "n": attempt + 1, "code": code}
    yield {"event": "done", "passed": False, "code": code, "fixes": req.max_attempts}


@app.post("/api/autofix")
async def autofix(req: AutofixReq):
    if len(req.code) > AUTOFIX_MAX_CHARS:
        raise HTTPException(413, f"Code is too large for auto-fix ({AUTOFIX_MAX_CHARS} characters max). Use plain Translate for large files.")

    async def stream():
        try:
            async for ev in autofix_events(req):
                yield json.dumps(ev) + "\n"
        except HTTPException as e:
            yield json.dumps({"event": "error", "message": str(e.detail)}) + "\n"
        except Exception as e:
            yield json.dumps({"event": "error", "message": f"Unexpected error: {e}"}) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


# ---------- frontend ----------
FRONT = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=FRONT), name="static")


@app.get("/")
def index():
    return FileResponse(FRONT / "index.html")
