# 🕺 RAG Russian Breakdance Trainer

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-0.11+-purple.svg)](https://www.llamaindex.ai/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991.svg)](https://openai.com/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-FRED--T5--1.7B-yellow.svg)](https://huggingface.co/ai-forever/FRED-T5-1.7B)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An interactive Russian-speaking chatbot that answers breakdance questions
using Retrieval-Augmented Generation (RAG) over a local knowledge base,
with two interchangeable LLM backends (cloud OpenAI + local FRED-T5),
a safety filter, and a Streamlit web UI.

---

## Note on origin

The project was **originally written ~6 months ago in Google Colab** as a
single notebook-style script (with `!pip install` magics, `getpass` for the
API key, and everything at module top level). It has since been **refactored
into a plain Python 3.12 script** that runs locally in VS Code or any
terminal, with proper `.env` handling, `if __name__ == "__main__"` guard,
CUDA/CPU device detection, and a full Streamlit wrapper.

---

## Example Result

Screenshots and a demo video of the running application:

- 📸 Photos: `doc/1.jpg`, `doc/2.jpg`, `doc/3.jpg`, `doc/4.jpg`, `doc/5.jpg`
- 🎥 Demo video: `doc/work.mp4`

| Chat dialog | Mode switching | Loading screen |
|---|---|---|
| ![1](doc/1.jpg) | ![2](doc/2.jpg) | ![3](doc/3.jpg) |

| Answer with logs | Blocked / easter egg |
|---|---|
| ![4](doc/4.jpg) | ![5](doc/5.jpg) |

**To view the video**, open `doc/work.mp4` in any player (VLC, MPV, browser).

### Compress screenshots (optional)

Screenshots are stored at full resolution. To shrink them ~8× smaller
(useful to keep the repo lightweight), run:

```powershell
pip install Pillow
python - <<'PY'
from pathlib import Path
from PIL import Image

DOC = Path("doc")
for p in sorted(DOC.glob("*.jpg")):
    img = Image.open(p)
    w, h = img.size
    img.resize((w // 8, h // 8), Image.LANCZOS).save(p.with_suffix(".small.jpg"), quality=82)
    print(f"{p.name}: {w}x{h} -> {w//8}x{h//8}")
PY
```

Or with ImageMagick in one line:

```bash
magick mogrify -resize 12.5% -path doc doc/*.jpg
```

---

## About the Project

| Feature | Where in code | Manual / Library |
|---|---|---|
| RAG pipeline assembly (index → rerank → reorder → HyDE) | `main.py` → `build_query_engine()` | Library (LlamaIndex) |
| Logging LLM reranker (`LoggingLLMRerank`) | `main.py` | **Manual** subclass |
| Logging reorder (`LoggingLongContextReorder`) | `main.py` | **Manual** subclass |
| Knowledge loading (local `doc/` → Google Docs fallback) | `main.py` → `prepare_knowledge()` | **Manual** |
| Dual backend routing (`fun_mode`) | `main.py` → `answer()` | **Manual** |
| FRED-T5 loader with CUDA/CPU auto-detect | `main.py` → `load_fred()` | Library + **manual** device logic |
| Safety filter (blocklist) | `main.py` → `is_dangerous()` / `safe_query()` | **Manual** |
| Hard-coded reactions (`hard_motivation`, Dima easter egg) | `main.py` + `app.py` | **Manual** |
| Streamlit state machine (`age → preparing → chat → blocked`) | `app.py` | **Manual** |
| Logging buffer (`LOG`, `log()`, `run_with_logs()`) | `main.py` + `app.py` | **Manual** |
| Environment handling (`.env`, `python-dotenv`) | `main.py` | Library |

---

## Dataset / Data

| Property | Value |
|---|---|
| Source (preferred) | Local `doc/` folder |
| Source (fallback) | [Google Docs export](https://docs.google.com/document/d/1KuDv9PwgTVfybpNIKL33_Jr4OTF4gdgmVAkapc0cvRk/export?format=txt) |
| Format | `.txt` (also `.md`, `.docx`, `.pdf`) |
| Size | ~14 KB plain text, ~150 paragraphs |
| Topics | History, philosophy (18–30), warm-up, culture, 4 elements (toprock/downrock/freezes/powermoves), Air Twist, Flare, Swipe, nutrition, sleep, coaching |
| Chunking | 512 tokens per node, 50-token overlap |

`SimpleDirectoryReader` picks up **all** files in `doc/`, so you can drop more
texts in and they'll be indexed together on the next run.

---

## Installation and Run

### 1. Clone the repo

```bash
git clone https://github.com/artrokza-lang/rag_russian_breakdance_trainer.git
cd rag_russian_breakdance_trainer
```

### 2. Create a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Or on Linux / macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If you plan to load `.docx` files, also install:

```bash
pip install docx2txt
```

For `.pdf`:

```bash
pip install pypdf
```

### 4. Configure the API key

```powershell
copy .env.example .env
notepad .env
```

Fill in your real key:

```env
OPENAI_API_KEY=sk-proj-...
```

**Never commit `.env`** — it's already in `.gitignore`.

### 5. Add your knowledge base

```powershell
mkdir doc
notepad doc/breakdance_knowledge.txt
```

Paste the breakdance text, save as UTF-8.

### 6. Run

**Web UI (recommended):**

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

**CLI:**

```bash
python main.py
```

### 7. GPU acceleration (optional)

Default `torch` is CPU-only. For a fast FRED-T5 on NVIDIA:

```powershell
pip uninstall torch -y
pip install torch --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Hyperparameters

### LLM & embeddings

| Parameter | Value | Where |
|---|---|---|
| LLM model | `gpt-4o-mini` | `main.py` → `build_query_engine()` |
| Embedding model | `text-embedding-3-small` | `main.py` → `build_query_engine()` |
| LLM temperature | `0.1` | `Settings.llm` |
| Chunk size | `512` tokens | `Settings.chunk_size` |
| Chunk overlap | `50` tokens | `Settings.chunk_overlap` |
| Retrieval `top_k` | `10` | `similarity_top_k` |
| Rerank `top_n` | `3` | `LLMRerank(top_n=3)` |
| Rerank `choice_batch_size` | `5` | `LLMRerank(choice_batch_size=5)` |
| HyDE `include_original` | `True` | `HyDEQueryTransform` |

### FRED-T5 generation

| Parameter | Value | CPU / GPU |
|---|---|---|
| `max_new_tokens` | `256` | both |
| `temperature` | `0.2` | both |
| `top_k` | `50` | both |
| `top_p` | `0.95` | both |
| `repetition_penalty` | `1.03` | both |
| `dtype` | `float32` / `float16` | CPU / CUDA |

---

## Results Analysis

| Aspect | Status | Notes |
|---|---|---|
| RAG retrieval quality | ✅ | `LLMRerank` + `LongContextReorder` + HyDE work — answers cite relevant sections |
| Grounding on local base | ✅ | Prompt-injected chunks from `doc/`, not model memory |
| Dual backend switching | ✅ | `/fun` (FRED) and `/normal` (OpenAI) route correctly via `fun_mode` |
| Safety filter | ⚠️ | Keyword-based blocklist catches obvious cases, misses paraphrases |
| Fun mode quality | ⚠️ | FRED-T5 answers are hit-or-miss — small 1.7B model, no RAG context |
| Response time `/normal` | ✅ | 2–10 s per question (4–5 OpenAI calls: HyDE + embeddings + rerank + generate) |
| Response time `/fun` (CPU) | ❌ | **30–90 s** per question — too slow for real-time |
| Response time `/fun` (GPU) | ✅ | 2–5 s per question |
| Streaming | ❌ | Answers appear only after full generation |
| Multi-file `doc/` support | ✅ | `SimpleDirectoryReader(input_dir=...)` picks up everything |
| Error surfacing | ✅ | Live logs, per-request log, download log, full traceback |
| Region handling | ❌ | OpenAI returns `403 unsupported_country_region_territory` from Russia — requires VPN/proxy or local models |

### Known issues

- `torch_dtype=` is deprecated in `transformers ≥ 4.48` — will produce a `FutureWarning` (harmless, but should be switched to `dtype=`).
- If a knowledge download is interrupted, `breakdance_knowledge.txt` may be empty and silently produce an empty index.
- `HyDE + LLMRerank` = **4–5 OpenAI requests per question**, so cost per answer is higher than a plain RAG.

---

## Troubleshooting / Known Issues

### `403 unsupported_country_region_territory` (from Russia)

**Cause:** OpenAI blocks requests from Russian IPs before validating the key.

**Fix (pick one):**

Option A — VPN on the whole PC (US/EU exit node) *before* starting Streamlit.

Option B — Proxy via environment variables:

```powershell
$env:HTTPS_PROXY="http://user:pass@host:port"
$env:HTTP_PROXY="http://user:pass@host:port"
streamlit run app.py
```

Option C — Fully local models (no OpenAI at all). `pip install llama-index-embeddings-huggingface llama-index-llms-ollama` and patch `build_query_engine()`:

```python
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

Settings.llm = Ollama(model="llama3.2", request_timeout=180.0)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

Install [Ollama](https://ollama.com/) and run `ollama pull llama3.2` first.

### `401 Incorrect API key`

Verify the key:

```powershell
python -c "from openai import OpenAI; print(len(OpenAI().models.list().data))"
```

- Number → key is valid.
- `AuthenticationError` → wrong key, generate a new one at https://platform.openai.com/api-keys.

### `429 insufficient_quota`

Key is valid, but account has $0 balance. Top up at https://platform.openai.com/account/billing.

### FRED-T5 returns empty / NaN answers

Cause: `float16` on CPU is unstable.

Fix: ensure device detection is correct:

```python
use_cuda = torch.cuda.is_available()
dtype = torch.float16 if use_cuda else torch.float32
```

Already implemented in `load_fred()`.

### RAG index rebuilds on every Streamlit click

Cause: `@st.cache_resource` not applied.

Fix: `_get_query_engine()` and `_get_fred_generator()` are already wrapped — do not remove the decorator.

### FRED-T5 downloads ~7 GB every run

Cause: corrupted or missing Hugging Face cache.

Fix: delete and re-download:

```powershell
Remove-Item -Recurse C:\Users\User\.cache\huggingface\hub\models--ai-forever--FRED-T5-1.7B
```

### `Port 8501 is already in use`

```powershell
streamlit run app.py --server.port 8502
```

Or kill the old process:

```powershell
Get-Process python | Stop-Process -Force
```

### Sidebar with mode toggle is not visible

Streamlit hides the sidebar on narrow windows — click the `>` chevron in the top-left corner.

---

## Roadmap

### Phase 1 — Stability & Cost
- Add **response streaming** (`response_gen` from LlamaIndex) so answers appear word-by-word.
- Cache RAG responses in SQLite keyed by `(question, model)` to avoid re-paying for repeated questions.
- Add `--no-hyde` flag to skip HyDE for cheaper queries.
- Migrate `except Exception` to specific `requests.RequestException` / `OSError` (fix `BLE001`).

### Phase 2 — Local-only mode
- Ship a `--local` profile: `HuggingFaceEmbedding` + `Ollama` backend, no OpenAI.
- Replace FRED-T5 with a stronger Russian LLM (Saiga, T-lite, or Qwen2.5-7B-Instruct) as an option.
- Add `bitsandbytes` 4-bit quantization for smaller GPUs.

### Phase 3 — Better UX
- Persist chat history in Streamlit via `st.session_state` + `localStorage`.
- Add file uploader in the sidebar to drop new `.txt`/`.docx` into `doc/` without leaving the browser.
- Show per-chunk source attribution next to every answer.

### Phase 4 — Evaluation & Safety
- Add an eval script: 20 golden Q/A pairs, score by LLM-as-judge, publish metrics in README.
- Replace keyword blocklist with **LlamaGuard** or OpenAI moderation endpoint.
- Add unit tests with `FakeEmbedding` / `FakeLLM` mocks — no API cost during CI.

---

## Project Structure

```
rag_russian_breakdance_trainer/
├── .env.example                # placeholder, safe to commit
├── .env                        # real key — NEVER commit
├── .gitignore
├── README.md
├── requirements.txt
├── main.py                     # RAG + FRED-T5 core, CLI entry point
├── app.py                      # Streamlit web UI
└── doc/
    ├── breakdance_knowledge.txt   # knowledge base (indexed by SimpleDirectoryReader)
    ├── work.mp4                   # demo video
    ├── 1.jpg                      # screenshot
    ├── 2.jpg
    ├── 3.jpg
    ├── 4.jpg
    └── 5.jpg
```

---

## Limitations

- **FRED-T5 is not RAG-aware.** In `fun` mode, the model receives only the raw
  question and a "be funny" instruction — knowledge chunks from `doc/` are **not**
  passed to it. Fun-mode answers are therefore not grounded.
- **Cost per question** in `normal` mode is 4–5 OpenAI calls
  (HyDE generation, two embeddings, rerank, final generation).
- **Region-locked.** OpenAI is unavailable from Russian IPs — see Troubleshooting.
- **CPU inference of FRED-T5** is very slow (30–90 s/answer) — GPU is strongly recommended.
- **Easter eggs** (`Dima`, `hard_motivation`, `Будет вкусно`) are hard-coded strings and will not translate well to other languages.
- **No streaming** — every answer is returned in one chunk after full generation.
- **No persistence** — chat history is lost when the browser tab is closed.
- **No test suite yet** — logic is validated manually.

---

## License

MIT — free to use, modify and distribute.
