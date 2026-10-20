# Code Translation Assistant

## Table of Contents

- [Overview](#overview)
- [Supported Languages](#supported-languages)
- [Architecture](#architecture)
- [AI Models Used](#ai-models-used)
- [How Code Translation Works](#how-code-translation-works)
- [How Code Analysis Works](#how-code-analysis-works)
- [How the Auto-Fix Verification Loop Works](#how-the-auto-fix-verification-loop-works)
- [How Code Execution Works](#how-code-execution-works)
- [Backend in Detail](#backend-in-detail)
- [Frontend in Detail](#frontend-in-detail)
- [Frontend to Backend Connection](#frontend-to-backend-connection)
- [Dataset and Training](#dataset-and-training)
- [Test Cases and Verification](#test-cases-and-verification)
- [Configuration Reference](#configuration-reference)
- [Dependencies](#dependencies)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Limitations and Security Notes](#limitations-and-security-notes)


---

## Overview

Code Translation Assistant (branded as CodeShift in the UI) is a full-stack web application that translates source code from one programming language to another using large language models. Beyond translation, it provides step-by-step code explanations, static error detection, live code execution, output comparison between the original and translated programs, and an automated fix loop that detects failures and asks the model to correct them until the translation passes.

The application is designed as a local development tool. The backend is a Python FastAPI server that communicates with either the Google Gemini API or a locally running Ollama instance. The frontend is a single HTML file served directly by the backend, requiring no separate build step or frontend framework.


---

## Supported Languages

The application supports translation between the following 19 programming languages:

| Language   | File Extension | Can Be Executed Locally |
|------------|---------------|------------------------|
| Python     | .py           | Yes                    |
| Java       | .java         | Yes (requires javac and java) |
| C++        | .cpp          | Yes (requires g++)     |
| C          | .c            | Yes (requires gcc)     |
| C#         | .cs           | No                     |
| JavaScript | .js           | Yes (requires node)    |
| TypeScript | .ts           | No                     |
| Go         | .go           | Yes (requires go)      |
| Rust       | .rs           | No                     |
| Kotlin     | .kt           | No                     |
| Swift      | .swift        | No                     |
| PHP        | .php          | Yes (requires php)     |
| Ruby       | .rb           | Yes (requires ruby)    |
| Scala      | .scala        | No                     |
| R          | .r            | No                     |
| Dart       | .dart         | No                     |
| MATLAB     | .m            | No                     |
| SQL        | .sql          | No                     |
| Bash       | .sh           | Yes (requires bash)    |

All 19 languages can be used as source or target for translation. The AI model handles the translation regardless of whether the language can be executed locally. The execution column refers only to whether the server can compile and run code in that language for the verification and auto-fix features.


---

## Architecture

The application follows a two-tier architecture.

### Backend Technology

The backend is built entirely with FastAPI, a Python web framework. It is not built with MySQL, PostgreSQL, MongoDB, or any other database system. There is no database anywhere in this application. No data is stored, no tables are created, no queries are run against any database engine. The backend is purely an API server that receives requests, forwards them to the AI model, and returns responses. FastAPI was chosen because it supports asynchronous request handling natively, which is necessary for making concurrent API calls to the AI model provider and for streaming responses back to the client.

The backend also handles code execution by invoking compilers and interpreters as subprocesses, and it serves the frontend HTML file as a static asset. Everything runs from a single Python file (main.py).

### No Database

To be completely clear: there is no MySQL, no SQLite, no PostgreSQL, no MongoDB, no Redis, and no file-based storage. The application is fully stateless. Nothing is saved between requests. When the user translates code, the result is returned directly to the browser and exists only in the browser memory. If the page is refreshed, everything is gone.

### System Diagram

```
Frontend (Single HTML file)
    |
    | HTTP REST API (JSON)
    | POST /api/translate
    | POST /api/analyze
    | POST /api/run
    | POST /api/autofix (NDJSON streaming)
    | GET  /api/runtimes
    |
Backend (FastAPI / Python, no database)
    |
    | HTTPS (external API) or HTTP (local Ollama)
    |
AI Model Provider
    (Google Gemini API or local Ollama)
```


---

## Deep Learning Model Used

This application does not train any model. It uses pre-trained large language models (LLMs) through API calls. The deep learning architecture behind these models is the Transformer, specifically the decoder-only Transformer variant that powers modern generative AI.

The application supports two model providers, configured through the PROVIDER environment variable.

### Google Gemini (Default Provider)

The primary model is **gemini-3.8-flash**. This is a large language model developed by Google DeepMind, based on the Transformer architecture. It belongs to the Gemini family of multimodal models. The application uses it purely for text-to-text generation (code in, code out).

The model is accessed through the Google Generative Language REST API at the endpoint https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent. An API key is required, which can be obtained for free from aistudio.google.com.

**How much of the model is used per request:**

- For translation: each API call allows up to 8000 output tokens. A single small file uses one API call. A large file that gets split into 5 chunks uses 5 sequential API calls, so up to 40000 output tokens total.
- For code analysis: each API call allows up to 6000 output tokens. Chunks are processed in parallel (default 2 at a time), so a file split into 4 chunks makes 4 API calls totaling up to 24000 output tokens.
- For auto-fix: the initial translation uses the same token budget as above. Each fix attempt adds one more API call (8000 tokens). With 3 fix attempts, that is 3 additional calls.
- Language detection (used in auto-fix when source is set to auto-detect) uses one API call with a 1000 token limit.

The exact number of input tokens depends on the size of the code being translated. The model processes both the system prompt (roughly 50 to 150 words depending on the task) and the user prompt (which contains the actual code) as input.

**Fallback model:** gemini-3.5-flash-lite is configured as the default fallback. It is a smaller, faster variant of Gemini. The fallback is used when the primary model returns HTTP 429 (rate limit), 503 (service unavailable), 500 (server error), or 404 (model not found).

**Retry strategy:**

1. Try the primary model (gemini-3.8-flash).
2. If it returns a retryable error, try each fallback model in order.
3. If all models fail in a round, wait with exponential backoff (starting at 3 seconds, doubling each round, capped at 20 seconds).
4. Repeat for up to 4 rounds (configurable via RETRY_ROUNDS).
5. If all rounds are exhausted, return an HTTP 503 error to the client.

### Ollama with Qwen (Local Alternative)

For fully offline usage without any cloud API, the application supports Ollama, which is a local model runner. The default model is **qwen2.5-coder:7b**, a 7 billion parameter code-specialized language model from the Qwen family developed by Alibaba. This model also uses the Transformer architecture but runs entirely on the user's local machine.

When using Ollama, the application sends requests to the local Ollama server (default URL: http://localhost:11434) using the /api/chat endpoint. No API key is needed. No data leaves the local machine.

No fallback models are configured for Ollama. The same retry logic applies, but there is only one model in the list.

### Why These Models and Not a Custom Trained Model

Code translation is a task that benefits enormously from the broad programming knowledge embedded in large pre-trained models. Training a custom model for code translation would require millions of parallel code examples across all 19 language pairs (which is 342 directional pairs), and even then it would likely underperform a general-purpose LLM that has already seen billions of lines of code during pre-training. The engineering effort in this application is in the prompt design and the verification loop, not in model training.


---

## How Code Translation Works

Translation is the core feature. When a user submits code with a source and target language, the following pipeline executes:

### Step 1: Chunking

Large files are split into smaller pieces before being sent to the model. The chunking algorithm in the chunk_code function works as follows:

- The code is split into lines.
- Lines are accumulated into a chunk until a boundary condition is met.
- A boundary is detected when the accumulated chunk has reached at least 60 percent of the chunk size limit (default 6000 characters), the current line is non-empty and starts at column zero (not indented), and the previous line was blank. This heuristic ensures that splits happen at natural code boundaries like function definitions or class declarations, never in the middle of a function body.
- A hard cap at 150 percent of the chunk size forces a split even without a natural boundary, preventing any single chunk from becoming excessively large.
- Each chunk records its starting line number for reference.

### Step 2: Sequential Translation with Context Passing

Chunks are translated sequentially, not in parallel. This is intentional. Each chunk after the first receives the last 40 lines of the previously translated chunk as context. This context passing serves several purposes:

- It preserves naming consistency across chunks (variable names, function names, class names).
- It prevents duplicate import statements, since the model is instructed not to repeat imports already defined in previous context.
- It maintains structural coherence so the final assembled output reads as a single cohesive file.

### Step 3: Prompt Construction

For each chunk, a prompt is constructed that includes:

- The source and target language names.
- The chunk index and total chunk count (for example, "part 2 of 5").
- For chunks after the first: the tail context from the previous translation, with an explicit instruction not to repeat imports or definitions.
- For the first chunk: an instruction to include all needed imports at the top.
- For non-final chunks: an instruction not to add a main entry point unless the source code has one in this section.
- The actual source code to translate, enclosed in a fenced code block.

### Step 4: System Prompt

The translation system prompt instructs the model to behave as an expert polyglot engineer. The model is told to:

- Produce idiomatic, compilable, runnable code in the target language.
- Preserve the original behavior exactly.
- Map libraries to standard equivalents in the target language.
- Add a short comment where semantics differ between the two languages.
- Output only one fenced code block with no additional prose.

### Step 5: Response Processing

The model response is processed by the strip_fences function, which extracts the content from within the markdown code fence. The extracted code from all chunks is joined with double newlines to produce the final translated file.

### Step 6: Response to Client

The API returns a JSON object containing the translated code and the number of chunks processed.


---

## How Code Analysis Works

The analysis feature provides a step-by-step explanation of the code and detects potential errors. It operates through the /api/analyze endpoint.

### Chunking for Analysis

The same chunk_code function is used to split the code. However, unlike translation, analysis chunks are processed in parallel using Python asyncio with a configurable concurrency semaphore (default: 2 concurrent requests).

### Line Numbering

Before sending each chunk to the model, every line is prefixed with its absolute line number (relative to the full original file, not the chunk). This ensures that when the model reports an error on a specific line, the line number corresponds to the original file regardless of which chunk the line belongs to.

### System Prompt for Analysis

The analysis system prompt instructs the model to act as a senior code reviewer and respond with a specific JSON structure:

- language: the detected programming language
- summary: a brief summary of what the code does
- explanation: a detailed step-by-step walkthrough in Markdown, written for a learner
- errors: an array of objects, each containing line (integer), severity (one of error, warning, or info), message (description of the problem), and fix (suggested correction)

The model is instructed to find syntax errors, runtime bugs, logic errors, unsafe patterns, and missing imports. It is also told not to invent problems: if the code is correct, the errors array should be empty.

### Result Aggregation

After all chunks are analyzed in parallel:

- The language is taken from the first chunk that returned one.
- Summaries from all chunks are concatenated with spaces.
- Explanations from all chunks are concatenated with double newlines.
- All errors from all chunks are merged into a single list and sorted by line number.


---

## How the Auto-Fix Verification Loop Works

The auto-fix feature is the most sophisticated part of the application. It combines translation, execution, output comparison, and iterative model-driven repair into a single automated pipeline. It operates through the /api/autofix endpoint and streams progress to the client using NDJSON (newline-delimited JSON).

### Pipeline Steps

1. Translate the code from source to target language using the standard translation pipeline described above.

2. Detect the source language if it was set to auto-detect. This is done by sending the first 3000 characters of the source code to the model with a system prompt that asks it to identify the language from the list of supported runtime languages.

3. Run the original program if the source language has a runtime available on the server. The output of this run becomes the reference output.

4. Run the translated program using the target language runtime.

5. Judge the result by comparing the translation run against the reference. The judge function applies the following logic:
   - If the translation failed to compile, it fails with the compiler error message.
   - If the translation timed out (default 10 seconds), it fails with a timeout message.
   - If the original program itself fails or times out, the translation is considered passing as long as it runs without errors (since there is no valid reference to compare against).
   - If both programs ran successfully, their stdout outputs are compared after normalizing trailing whitespace and line endings. If outputs match, the translation passes. If they differ, it fails with a diff showing expected versus actual output.

6. If the translation fails, the model is asked to fix it. A repair prompt is constructed containing the original source code, the current (failing) translation, and the specific error or output mismatch. The fix system prompt tells the model to return the complete corrected translation, changing only what is needed to fix the problem.

7. The fixed code replaces the previous translation, and steps 4 through 6 repeat until either the translation passes or the maximum number of fix attempts is reached (configurable, default 3, maximum 5).

### Streaming Events

The auto-fix endpoint streams the following event types as NDJSON:

| Event      | Description |
|------------|-------------|
| status     | A human-readable progress message |
| translated | The initial translation is ready |
| reference  | The result of running the original program |
| attempt    | The result of running a translation attempt, with pass/fail and reason |
| fixed      | The model produced a new fix |
| done       | The loop is complete, with final pass/fail status and fix count |
| error      | An unrecoverable error occurred |

### Size Limit

The auto-fix feature has a separate size limit (default 30000 characters, configurable via AUTOFIX_MAX_CHARS) that is lower than the general translation limit (500000 characters). This is because auto-fix requires running the code and iterating, which is impractical for very large files.


---

## How Code Execution Works

The application can compile and run code in 9 of the 19 supported languages. Code execution is handled entirely on the server machine. There is no sandboxing beyond a clean environment variable filter.

### Execution Process

1. A temporary directory is created for each run.
2. The source code is written to a file with the appropriate name and extension.
3. For interpreted languages (Python, JavaScript, PHP, Ruby, Bash), the interpreter is invoked directly on the file.
4. For compiled languages (C, C++, Java), compilation runs first. If compilation fails, the compiler error is returned immediately. If compilation succeeds, the resulting binary is executed.
5. Go uses "go run" which handles compilation internally but gets an extended timeout (run timeout plus 20 seconds) to account for compilation time.
6. Java detects the public class name from the source code using a regex and uses it as the filename, since Java requires the filename to match the public class name.

### Security Measures

- A safe environment filter strips all environment variables except those needed by compilers and runtimes. This prevents executed code from reading sensitive values like the GEMINI_API_KEY.
- A configurable timeout (default 10 seconds) kills any program that runs too long.
- Output is capped at 20000 characters to prevent memory issues from programs that produce excessive output.
- The entire feature can be disabled by setting ENABLE_RUN to false.

### Runtime Detection

On startup, and when queried via GET /api/runtimes, the server checks which language runtimes are actually installed on the host machine by looking for their executables on PATH (using shutil.which). The frontend uses this information to enable or disable run buttons for specific languages.


---

## Backend in Detail

The backend is a single Python file (backend/main.py, 443 lines) built with FastAPI. It serves as both the API server and the static file server for the frontend.

### What the Backend Actually Does

The backend is the bridge between the frontend and the AI model. Its responsibilities are:

1. Receiving code and language parameters from the frontend via REST API calls.
2. Constructing carefully engineered prompts that guide the AI model to produce correct, idiomatic translations and analyses.
3. Managing the chunking of large files so they fit within model context limits.
4. Handling retry logic with model fallbacks and exponential backoff when the AI provider is rate-limited or unavailable.
5. Parsing model responses, extracting code from markdown fences, and parsing JSON from analysis responses (with fallback parsing for malformed JSON).
6. Orchestrating the auto-fix loop: running code, comparing outputs, feeding errors back to the model, and streaming progress.
7. Executing user code in temporary directories with security-filtered environments.
8. Serving the frontend HTML file and static assets.

### API Endpoints

| Method | Path           | Purpose |
|--------|---------------|---------|
| POST   | /api/translate | Translate code from one language to another |
| POST   | /api/analyze   | Explain code and detect errors |
| POST   | /api/run       | Execute code and return output |
| POST   | /api/autofix   | Translate, run, compare, and iteratively fix (streamed) |
| GET    | /api/runtimes  | List which language runtimes are available |
| GET    | /               | Serve the frontend HTML page |

### Request and Response Models

Translation request accepts code (string, 1 to 500000 characters), source language (string, defaults to auto-detect), and target language (string, required). It returns the translated code and the number of chunks processed.

Analysis request accepts code (string, 1 to 500000 characters) and language (string, defaults to auto-detect). It returns the detected language, summary, explanation in Markdown, a list of errors with line numbers and severities, and the chunk count.

Run request accepts code (string, 1 to 200000 characters), language (string, required), and optional stdin input. It returns the stage (compile or run), stdout, stderr, exit code, whether it timed out, and execution time in seconds.

Auto-fix request accepts code, source, target, optional stdin, and max_attempts (0 to 5, default 3). It returns a stream of NDJSON events.


---

## Frontend in Detail

The frontend is a single self-contained HTML file (frontend/index.html, 569 lines) with all CSS and JavaScript inlined. It uses no frameworks, no build tools, and no external JavaScript libraries. The only external resources are Google Fonts (DM Sans, Inter, and JetBrains Mono).

### UI Layout

The interface is a two-panel layout:

Left panel (Input):
- A language selector dropdown with auto-detect and all 19 languages.
- A code editor textarea with monospaced font, tab key support (inserts 4 spaces), and a character/line counter.
- Buttons for running the input code, loading a sample, opening a file from disk, and clearing the editor.

Right panel (Output):
- A language selector for the target language (defaults to Java).
- Action buttons: Translate, Auto-fix (with a configurable fix attempt dropdown), and Explain and Check.
- Five tabbed views:
  - Translated Code: displays the raw translated code in a pre element.
  - Verify: shows the step-by-step auto-fix log with pass/fail results.
  - Console: provides buttons to run the input code, the translated code, or both side by side with output comparison. Includes a stdin textarea for program input.
  - Explanation: renders the Markdown explanation from the analysis.
  - Errors: lists detected issues with severity badges, line numbers, and suggested fixes.
- Copy and Download buttons for the translated output.

### Theme Support

The UI supports dark and light themes. The theme toggle is a button in the header. Theme preference is saved to localStorage and falls back to the operating system preference via the prefers-color-scheme media query. The theme is applied via a data-theme attribute on the HTML root element, with all colors defined as CSS custom properties.

### Swap Feature

The swap button takes the translated output and moves it into the input editor, switching the source and target languages. This allows chaining translations (for example, translating Python to Java, then swapping to translate that Java to Go).

### File Upload

The Open File button triggers a hidden file input. When a file is selected, its text content is loaded into the editor. The file extension is used to auto-detect and set the source language dropdown.

### Markdown Rendering

The frontend includes a minimal Markdown renderer for the explanation tab. It handles code blocks, inline code, bold text, headings, and bullet lists. It does not use any external Markdown library.


---

## Frontend to Backend Connection

The frontend communicates with the backend entirely through HTTP fetch calls to the API endpoints. Since the frontend HTML is served by the same FastAPI server, all API calls are same-origin requests to relative paths (for example, /api/translate), so there are no CORS issues.

### Standard API Calls

For the /api/translate, /api/analyze, and /api/run endpoints, the frontend sends a POST request with a JSON body and receives a JSON response. The api helper function handles error checking: if the response status is not OK, it attempts to parse the error detail from the JSON body and throws an exception.

### Streaming API Call

The /api/autofix endpoint uses NDJSON streaming. The frontend reads the response body as a stream using the ReadableStream API (response.body.getReader()). It accumulates incoming bytes in a buffer using a TextDecoder, splits on newline characters, and parses each complete line as a JSON event. Each event is processed by the handle function, which updates the UI in real time: showing progress messages, rendering run results, updating the translated code tab, and displaying the final pass/fail banner.

### Runtime Detection

On page load, the frontend sends a GET request to /api/runtimes. The response tells the frontend which languages have runtimes installed, the execution timeout, and whether code execution is enabled at all. This information is used to validate run requests before sending them to the server.

### Static File Serving

The backend mounts the frontend directory as a static file directory at /static and serves index.html at the root path /. This means the entire application runs from a single origin (http://localhost:8000 by default) with no separate frontend server needed.


---

## Dataset and Training

This application does not use any custom dataset. No dataset was collected, curated, or created for this project. No model training, fine-tuning, or transfer learning is performed anywhere in this codebase.

### What Dataset the Underlying Models Were Trained On

The models used by this application (Gemini and Qwen) were pre-trained by their respective organizations before this application was built. This application simply calls them through APIs.

**Google Gemini** was trained by Google DeepMind on a massive multi-modal dataset that includes publicly available web pages, books, code repositories (including open source code from platforms like GitHub), scientific papers, and other text sources. The exact dataset size and composition are not publicly disclosed by Google, but it is estimated to be in the trillions of tokens. The code portion of the training data spans dozens of programming languages, which is why the model can translate between languages it has seen during training.

**Qwen 2.5 Coder (7B)** was trained by Alibaba on a dataset that includes 5.5 trillion tokens of source code collected from public code repositories, code-related web content, synthetic code data, and natural language text. The code portion covers over 90 programming languages.

Neither of these datasets was created by or for this application. This application is a consumer of these pre-trained models, not a trainer.

### How This Application Uses the Models Without a Dataset

Instead of training on a dataset, this application uses prompt engineering. The system prompts written in the backend code are the mechanism that directs the model to perform code translation, analysis, and repair. These prompts are carefully engineered to:

- Constrain the output format (code only, no prose for translation; strict JSON for analysis).
- Specify the expected behavior (idiomatic, compilable, behavior-preserving).
- Handle edge cases (multi-part files, context passing, absolute line numbers).
- Guide the model to avoid common failure modes (inventing errors, repeating imports, adding unnecessary entry points).

There is no retrieval-augmented generation (RAG), no vector database, no fine-tuning, no few-shot examples, and no training loop anywhere in this application. The model receives a prompt, generates a response, and the application parses and returns that response.


---

## Test Cases and Verification

The application does not include a traditional test suite with unit tests or integration tests. Instead, it uses a runtime verification approach built directly into the product.

### How Verification Works

The auto-fix feature acts as an automated test framework:

1. The original program is run with optional stdin input to produce a reference output.
2. The translated program is run with the same stdin input.
3. The outputs are compared after normalizing whitespace and line endings.
4. If the outputs match, the translation is considered correct (the test passes).
5. If the outputs differ or the translation crashes, the test fails and the model is given the failure details to produce a fix.
6. This cycle repeats up to 5 times.

### What Counts as Passing

A translation passes verification when:

- It compiles without errors (for compiled languages).
- It runs without crashing (exit code 0).
- It completes within the timeout (default 10 seconds).
- Its stdout output matches the original program's stdout output exactly (after whitespace normalization).

### What Counts as Failing

A translation fails verification when:

- Compilation fails (compiler errors are reported).
- Runtime error occurs (non-zero exit code with stderr).
- The program times out (infinite loop or waiting for input not provided).
- The output differs from the original (exact text shown for both expected and actual).

### Sample Code

The frontend includes a built-in sample: a Python function that calculates the average of a list and calls it with an empty list (which produces a ZeroDivisionError). This sample is useful for testing the error detection feature and demonstrating how the auto-fix loop handles edge cases.


---

## Configuration Reference

All configuration is done through environment variables in the backend/.env file.

| Variable           | Default              | Description |
|--------------------|---------------------|-------------|
| PROVIDER           | gemini              | Model provider. Set to "gemini" for Google Gemini API or "ollama" for local Ollama. |
| GEMINI_API_KEY     | (empty)             | API key for Google Gemini. Required when PROVIDER is gemini. Obtain from aistudio.google.com. |
| GEMINI_MODEL       | gemini-3.8-flash    | Primary Gemini model to use. |
| GEMINI_FALLBACKS   | gemini-3.5-flash-lite | Comma-separated list of fallback models tried in order when the primary is unavailable. |
| RETRY_ROUNDS       | 4                   | Number of complete retry rounds (trying all models) before giving up. |
| OLLAMA_MODEL       | qwen2.5-coder:7b    | Model name for Ollama. |
| OLLAMA_URL         | http://localhost:11434 | URL of the local Ollama server. |
| CHUNK_CHARS        | 6000                | Target chunk size in characters for splitting large files. |
| CONCURRENCY        | 2                   | Maximum number of parallel analysis requests sent to the model. |
| ENABLE_RUN         | true                | Set to false to disable code execution entirely. |
| RUN_TIMEOUT        | 10                  | Maximum seconds a program is allowed to run before being killed. |
| AUTOFIX_MAX_CHARS  | 30000               | Maximum code size (characters) accepted by the auto-fix endpoint. |


---

## Dependencies

### Python Dependencies (backend/requirements.txt)

| Package        | Version  | Purpose |
|----------------|---------|---------|
| fastapi        | 0.115.0 | Web framework for the API server |
| uvicorn        | 0.30.6  | ASGI server to run FastAPI |
| httpx          | 0.27.0+ | Async HTTP client for calling Gemini and Ollama APIs |
| python-dotenv  | 1.0.1   | Load environment variables from the .env file |

### Frontend Dependencies

None. The frontend is a single HTML file with no npm packages, no build step, and no JavaScript dependencies. External resources are limited to Google Fonts loaded via CDN.

### System Dependencies for Code Execution

These are optional. Code execution works for whichever runtimes are installed:

| Language   | Required on PATH |
|------------|-----------------|
| Python     | python (uses the same Python running the server) |
| JavaScript | node |
| Java       | javac, java |
| C          | gcc |
| C++        | g++ |
| Go         | go |
| PHP        | php |
| Ruby       | ruby |
| Bash       | bash |


---

## Project Structure

```
code-translator/
    README.md
    backend/
        main.py              The entire backend: API, LLM integration, chunking,
                             code execution, auto-fix loop, and static file serving
        requirements.txt     Python package dependencies
        .env.example         Template for environment variables
        .env                 Actual environment variables (not committed to version control)
    frontend/
        index.html           The entire frontend: HTML structure, CSS styling,
                             and JavaScript application logic in one file
```


---

## Setup and Installation

### Prerequisites

- Python 3.9 or later
- A Google Gemini API key (free from aistudio.google.com), or Ollama installed locally

### Steps

Clone or download the project.

Navigate to the backend directory and create a virtual environment:

```
cd backend
python -m venv venv
```

Activate the virtual environment:

On macOS and Linux:
```
source venv/bin/activate
```

On Windows PowerShell:
```
venv\Scripts\Activate.ps1
```

Install Python dependencies:
```
pip install -r requirements.txt
```

Create the environment file from the template:
```
cp .env.example .env
```

Open .env in a text editor and add your Gemini API key, or set PROVIDER to ollama if using a local model.

Start the server:
```
uvicorn main:app --reload
```

Open http://localhost:8000 in a browser.


---

## Limitations and Security Notes

Code execution runs directly on the host machine with no container isolation. The environment variable filter prevents executed code from reading the API key, but executed code has full access to the filesystem and network of the host. Do not expose this application to the public internet without adding proper sandboxing (such as Docker containers or a serverless execution environment).

The maximum input size is 500000 characters for translation and analysis, and 30000 characters for auto-fix. These limits are configurable but are in place to avoid excessive API costs and long processing times.

Translation quality depends entirely on the capabilities of the underlying AI model. The application does not guarantee that translated code is correct, performant, or idiomatic. The auto-fix loop improves reliability but cannot fix all possible translation errors, especially for complex programs with intricate runtime behavior, concurrency, or platform-specific dependencies.

The application is stateless. There is no history, no user accounts, and no way to save or retrieve past translations. Each page refresh starts fresh.

Languages that cannot be executed locally (C#, TypeScript, Rust, Kotlin, Swift, Scala, R, Dart, MATLAB, SQL) can still be translated, but the auto-fix verification loop cannot run or compare their output. The translation is returned as-is without runtime validation.
