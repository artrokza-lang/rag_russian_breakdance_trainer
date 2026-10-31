# Neuro-trainer for breakdance (RAG)

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-0.11+-purple.svg)](https://www.llamaindex.ai/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--3.5--turbo-412991.svg)](https://openai.com/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-FRED--T5--1.7B-yellow.svg)](https://huggingface.co/ai-forever/FRED-T5-1.7B)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-ee4c2c.svg)](https://pytorch.org/)

Interactive chatbot that answers breakdance questions using
Retrieval-Augmented Generation (RAG) over a local knowledge base.

The bot speaks Russian, has two generation modes (cloud + local),
a basic safety filter, and a couple of scripted easter eggs.

---

## History

- **Originally written ~6 months ago in Google Colab** as a single
  notebook-style script with `!pip install` magic commands, `getpass`
  for the API key, and the whole flow at module top level.
- **Now ported to a plain Python 3.12 script** that runs locally
  in VS Code (or any terminal) — no Colab magic, no `exit()`,
  proper `.env` handling, `if __name__ == "__main__"` guard,
  and CUDA/CPU device detection for the local model.

---

## Stack

- **LlamaIndex** — RAG pipeline (vector index, reranking, reordering, HyDE)
- **OpenAI GPT-3.5-turbo** — "normal" mode (cloud, requires API key)
- **FRED-T5-1.7B** (`ai-forever/FRED-T5-1.7B`) — "fun" mode (local, Russian)
- **Google Docs / local `doc/` folder** — knowledge base source

### RAG enhancements

1. **VectorStoreIndex** with `chunk_size=512`, `chunk_overlap=50`
2. **LLMRerank** — LLM-based reranking of retrieved chunks
3. **LongContextReorder** — mitigates the "lost in the middle" effect
4. **HyDE** (Hypothetical Document Embeddings) — search with a pseudo-answer
5. **Custom logging wrappers** around post-processors for full trace

---

## Requirements

- **Python 3.12** (developed and tested on this version)
- Windows, Linux or macOS
- ~8–16 GB RAM if you plan to run FRED-T5 locally
- (optional) CUDA GPU with ≥6 GB VRAM for fast FRED-T5 inference
- An OpenAI API key with access to `gpt-3.5-turbo` and `text-embedding-*`

---

## Install

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
```

Activate it on Windows (cmd):

```bat
.venv\Scripts\activate.bat
```

Activate it on Linux / macOS / Git Bash:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install the `.docx` reader (if you plan to load Word files from `doc/`):

```bash
pip install docx2txt
```

Install the `.pdf` reader (if you plan to load PDFs from `doc/`):

```bash
pip install pypdf
```

---

## Configure

Create your local `.env` from the template on Windows:

```powershell
copy .env.example .env
```

Or on Linux / macOS:

```bash
cp .env.example .env
```

Open `.env` and set your key:

```env
OPENAI_API_KEY=sk-...
```

**Never commit `.env`** — it is already listed in `.gitignore`.

---

## Knowledge base

Put your breakdance texts into the `doc/` folder next to `main.py`:

```
NLP/
├── main.py
├── .env
├── requirements.txt
└── doc/
    └── breakdance_knowledge.txt
```

All `.txt` / `.md` / `.docx` / `.pdf` files inside `doc/` are loaded
and indexed automatically. If `doc/` is missing or empty, the script
falls back to downloading the knowledge base from Google Docs and
caching it as `breakdance_knowledge.txt` in the project root.

> ⚠️ Legacy `.doc` files are **not** supported — resave them as `.docx`.

---

## Run

Start the bot:

```bash
python main.py
```

You will be asked for your age, then you can ask questions.

### Commands

- `/fun` — switch to fun mode (local FRED-T5)
- `/normal` — switch to normal mode (OpenAI RAG)
- `/exit` — restart the session (ask age again)

---

## Free / offline testing

Avoid any OpenAI calls while testing local logic:

```bash
python main.py --fun-only
```

Uses a stub query engine instead of the real RAG pipeline.

Skip loading FRED-T5 too (no ~7 GB download, fully offline):

```bash
python main.py --fun-only --no-fred
```

Everything runs with stub answers — useful for smoke tests, CI,
or testing on machines without a GPU.

---

## Dev tools (optional)

Static checks:

```bash
ruff check main.py
```

Syntax check:

```bash
python -m py_compile main.py
```

Run tests (if you add `test_logic.py` / `test_rag_offline.py`):

```bash
pytest -v
```

---

## Project structure

```
rag_russian_breakdance_trainer/
├── .env.example          # placeholder, safe to commit
├── .gitignore            # .env, .venv, __pycache__, ...
├── README.md
├── requirements.txt
├── main.py               # single-file application
└── doc/                  # your knowledge base (.txt / .docx / .pdf)
```

---

## Notes

- FRED-T5 is downloaded from Hugging Face on first run
  (cached in `~/.cache/huggingface`). On CPU with `float32`,
  a single answer may take 30–90 seconds. On GPU with `float16`,
  it runs in a few seconds.
- The script automatically picks `float16` on CUDA and `float32`
  on CPU to avoid NaN outputs on CPU.
- HyDE adds one extra LLM call per query. Together with `LLMRerank`,
  a single question may cost 3 OpenAI requests. If cost matters,
  you can disable HyDE by returning `base_engine` directly from
  `build_query_engine()`.

## Example

```
Привет! Меня зовут Антон, я твой нейро-тренер по брейк-дансу.
Сколько тебе лет? 25
Отлично! Ты в нужной возрастной группе. Начнём тренировки!

--- Вопрос: Что такое freeze? ---
[OpenAI RAG] Query: Что такое freeze?
=== Nodes AFTER reranking (top 3) ===
1. Freeze — это заморозка в брейк-дансе, статичная поза...
2. ...
Ответ: Freeze — это статичная позиция...
💡 Подсказка: можешь спросить меня о правильном питании...
```

## What it can answer

- Базовые элементы: top rock, down rock, freeze, power move
- Питание, сон, режим тренировок
- Возрастные особенности 18–30 лет
- Отказывает на небезопасные запросы (блок-лист)
---

## License

MIT — free to use, modify and distribute.
