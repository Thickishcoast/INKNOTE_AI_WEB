# InkNote AI — Unified Agent Notebook

InkNote provides a shared user canvas and AI canvas. A user can write, sketch,
label a diagram, or combine all three, then submit the page with one **Click**
button.

Qwen3.5 routes each canvas submission to one of four internal actions:

- `chat` — answer or analyze the canvas in text
- `image` — send a refined, text-free visual prompt to FLUX
- `both` — return text and a generated image
- `clarify` — ask a question when the intent is uncertain

## Requirements

- Windows
- Python 3.14
- A CUDA-capable GPU for the default local FLUX configuration
- [Ollama](https://ollama.com/) installed and running
- Enough disk space for the Python environment and all three models

InkNote does **not** download models automatically. Install the exact models
below before starting the application.

| Provider | Model | Purpose |
| --- | --- | --- |
| Ollama | `qwen3.5:4b` | Thinking-enabled canvas understanding, routing, and chat |
| Ollama | `qwen3-embedding:0.6b` | Semantic-memory embeddings |
| Hugging Face | `black-forest-labs/FLUX.2-klein-4B` | Local image generation |

## Installation

### 1. Create the Python environment

Run these commands from the project root:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

If PowerShell blocks activation, run the executables through their full
`.venv\Scripts\` paths instead.

### 2. Install the Ollama models manually

Start the Ollama application, then run:

```powershell
ollama pull qwen3.5:4b
ollama pull qwen3-embedding:0.6b
ollama list
```

Confirm that both exact model names appear in `ollama list`.

If you change `INKNOTE_QWEN_MODEL` or `INKNOTE_EMBEDDING_MODEL` in `.env`,
manually pull the replacement model names as well.

### 3. Download the FLUX model manually

The application loads FLUX from the local Hugging Face cache and uses
`local_files_only=True`; it will not download missing files during startup or
image generation.

First, visit the model repository, accept any license or access terms, and
create a Hugging Face user access token if the repository requires one:

<https://huggingface.co/black-forest-labs/FLUX.2-klein-4B>

Authenticate from the project environment:

```powershell
.\.venv\Scripts\hf.exe auth login
```

Then download the complete repository into the standard Hugging Face cache:

```powershell
.\.venv\Scripts\hf.exe download black-forest-labs/FLUX.2-klein-4B
```

As an alternative to the CLI:

```powershell
.\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('black-forest-labs/FLUX.2-klein-4B')"
```

To verify the cached model without downloading anything:

```powershell
.\.venv\Scripts\hf.exe cache ls
```

The listing should include `black-forest-labs/FLUX.2-klein-4B`. If you configure
a different `INKNOTE_IMAGE_MODEL` in `.env`, download that repository instead.

`HUGGINGFACE_HUB_TOKEN` may be placed in the local `.env` file when needed.
Never commit a real token or add it to `.env.example`.

### 4. Start InkNote

```powershell
.\run_windows.bat
```

Open <http://127.0.0.1:8000>.

The launcher creates `.env` from `.env.example` when needed and starts Uvicorn.
It does not install, check, or download models.

## Troubleshooting model errors

- **Ollama connection error:** open the Ollama application and confirm
  `ollama list` works.
- **Ollama model not found:** pull the exact model name configured in `.env`.
- **FLUX files not found:** rerun `hf download` in the same Windows account that
  runs InkNote.
- **Hugging Face access denied:** accept the repository terms and run
  `hf auth login` again with a token that has read access.
- **CUDA memory error:** keep
  `INKNOTE_IMAGE_RELEASE_AFTER_GENERATION=true` and
  `INKNOTE_IMAGE_CPU_OFFLOAD=true`.
- **Dependency problem:** run
  `.\.venv\Scripts\python.exe -m pip check`.


## Canvas tools

- Pen and eraser
- Stroke-level undo and redo
- Preset and custom colors
- Optional notebook-line layout
- Low, medium, and high image-quality choices
- One automatic Send action

## Project structure

```text
backend/
  agents/       Request routing and chat orchestration
  services/     Ollama, FLUX, and semantic-memory services
  storage/      SQLite notes and JSON conversation history
  app.py        FastAPI application and routes
  config.py     Environment and project paths
frontend/
  css/          Stylesheets
  js/           Browser application
  icons/        PWA assets
  index.html
run_windows.bat Windows development launcher
tests/          Backend smoke tests
docs/           Architecture, learning, and API guides
data/           Runtime database, histories, memory, and images
```

## Environment

Configuration defaults are documented in `.env.example`. Keep
`INKNOTE_IMAGE_RELEASE_AFTER_GENERATION=true` when Qwen and FLUX share a GPU.
The model names in `.env` must exactly match the Ollama tags and Hugging Face
repository that you installed manually. `INKNOTE_MODEL_CONTEXT_SIZE=32768`
configures a 32K-token context for deep chat.
`INKNOTE_ROUTER_CONTEXT_SIZE=8192` keeps the unified router and memory extraction
bounded. `INKNOTE_QWEN_KEEP_ALIVE=5m` avoids
repeated Qwen reloads between chat requests.
