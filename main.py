# -*- coding: utf-8 -*-
"""
Neuro-trainer for breakdance — VS Code / Python 3.12 version.

An interactive chatbot that answers breakdance questions using:
  1. Retrieval-Augmented Generation (RAG) over a local knowledge base in `doc/`
  2. Two interchangeable LLMs:
       - OpenAI GPT-4o-mini (cloud, "normal" mode)
       - FRED-T5-1.7B (local, "fun" mode, Russian model)
  3. RAG enhancements: LLMRerank, LongContextReorder, HyDE
  4. A basic safety layer and hard-coded special reactions

Run (CLI):    python main.py
Run (web):    streamlit run app.py

Requirements:
    pip install -r requirements.txt

API key:
    Put OPENAI_API_KEY=sk-... into the .env file next to this script.
    Never commit .env to git.
"""
import os
import sys
import time
import getpass
from pathlib import Path

import requests
import torch
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# Load .env BEFORE importing llama-index (its OpenAI wrapper reads the key
# from os.environ at import time).
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    VectorStoreIndex,
)
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.postprocessor import LLMRerank, LongContextReorder
from llama_index.core.query_engine import TransformQueryEngine
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
DOC_ID = "1KuDv9PwgTVfybpNIKL33_Jr4OTF4gdgmVAkapc0cvRk"
DOC_DIR = BASE_DIR / "doc"                                # preferred source
KNOWLEDGE_FILE = BASE_DIR / "breakdance_knowledge.txt"    # Google Docs cache
FRED_MODEL_NAME = "ai-forever/FRED-T5-1.7B"

# Naive blocklist — refuse to answer if any of these appear in the question.
DANGER_WORDS = [
    "бомб", "взрыв", "оружие", "наркотик",
    "терроризм", "убийство", "насилие",
    "биткоин", "криптовалюта",
]


# -----------------------------------------------------------------------------
# In-memory log buffer (shared between main.py and app.py in the same process)
# -----------------------------------------------------------------------------
LOG: list[str] = []


def log(msg: str) -> None:
    """Append a timestamped line to LOG and print it to stdout."""
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    LOG.append(line)


def get_log() -> list[str]:
    return list(LOG)


def clear_log() -> None:
    LOG.clear()


# -----------------------------------------------------------------------------
# API key check
# -----------------------------------------------------------------------------
def ensure_api_key() -> None:
    """
    Ensure OPENAI_API_KEY is present. Priority:
      1. Already in os.environ (e.g. via .env).
      2. Interactive hidden prompt (fallback).
      3. Exit with error.
    """
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not found in .env — asking interactively.")
        key = getpass.getpass("Enter OpenAI API Key: ").strip()
        if not key:
            print("No key provided. Exiting.")
            sys.exit(1)
        os.environ["OPENAI_API_KEY"] = key


# -----------------------------------------------------------------------------
# Knowledge loading
# -----------------------------------------------------------------------------
def load_google_doc_text(doc_id: str) -> str:
    """Download the plain-text export of a Google Doc by its document ID."""
    url = f"https://docs.google.com/document/d/{doc_id}/export?format=txt"
    log(f"GET {url}")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.text


def prepare_knowledge() -> list:
    """
    Load the knowledge base.

    Priority:
      1. If `doc/` folder exists and is non-empty → load all files from it.
      2. Otherwise → download the Google Doc and cache it locally.
    """
    t0 = time.time()

    # 1. Preferred: local doc/ folder
    if DOC_DIR.exists() and any(DOC_DIR.iterdir()):
        files = [p.name for p in DOC_DIR.iterdir() if p.is_file()]
        log(f"Loading from local folder: {DOC_DIR} "
            f"({len(files)} file(s): {files})")
        docs = SimpleDirectoryReader(input_dir=str(DOC_DIR)).load_data()
        log(f"Loaded {len(docs)} document(s) in {time.time() - t0:.2f}s")
        return docs

    # 2. Fallback: cached Google Docs file, then live download
    if KNOWLEDGE_FILE.exists():
        log(f"doc/ empty or missing — using local cache: {KNOWLEDGE_FILE}")
    else:
        log("doc/ empty or missing — downloading Google Doc...")
        text = load_google_doc_text(DOC_ID)
        KNOWLEDGE_FILE.write_text(text, encoding="utf-8")
        log(f"Downloaded {len(text)} characters.")

    docs = SimpleDirectoryReader(input_files=[str(KNOWLEDGE_FILE)]).load_data()
    log(f"Loaded {len(docs)} document(s) in {time.time() - t0:.2f}s")
    return docs


# -----------------------------------------------------------------------------
# RAG post-processors with logging
# -----------------------------------------------------------------------------
class LoggingLLMRerank(LLMRerank):
    """LLM-based reranker that logs node counts before/after."""

    def _postprocess_nodes(self, nodes, query_bundle):
        log(f"Rerank: {len(nodes)} nodes in")
        reranked = super()._postprocess_nodes(nodes, query_bundle)
        log(f"Rerank: {len(reranked)} nodes out")
        return reranked


class LoggingLongContextReorder(LongContextReorder):
    """Reorders nodes to mitigate 'lost in the middle'. Logs node count."""

    def _postprocess_nodes(self, nodes, query_bundle):
        reordered = super()._postprocess_nodes(nodes, query_bundle)
        log(f"Reorder: {len(reordered)} nodes")
        return reordered


def build_query_engine(documents):
    """
    Build the full RAG pipeline: index → rerank → reorder → HyDE.

    Model choices (2024+):
      - gpt-4o-mini         — cheap, fast, follows instructions well
      - text-embedding-3-small — current embedding model
    """
    t0 = time.time()
    log("Building VectorStoreIndex (chunk=512, overlap=50)...")

    Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    Settings.chunk_size = 512
    Settings.chunk_overlap = 50

    index = VectorStoreIndex.from_documents(documents)
    log(f"Index built in {time.time() - t0:.1f}s")

    base = index.as_query_engine(
        similarity_top_k=10,
        node_postprocessors=[
            LoggingLLMRerank(top_n=3, choice_batch_size=5),
            LoggingLongContextReorder(),
        ],
    )

    # HyDE: search with a hypothetical answer instead of the raw question.
    # include_original=True keeps the raw question as well.
    hyde = HyDEQueryTransform(include_original=True)
    engine = TransformQueryEngine(base, hyde)
    log(f"RAG query engine ready ({time.time() - t0:.1f}s total)")
    return engine


# -----------------------------------------------------------------------------
# FRED-T5 (local, Russian) — used for the "fun" mode
# -----------------------------------------------------------------------------
def load_fred():
    """
    Load FRED-T5 with proper device/dtype selection:
      - CUDA available → float16 on GPU (fast, less VRAM)
      - CPU only       → float32 (float16 produces NaNs on CPU)
    """
    t0 = time.time()
    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32
    device = "cuda" if use_cuda else "cpu"
    log(f"Loading FRED-T5 ({FRED_MODEL_NAME}) device={device} dtype={dtype}")

    log("Downloading / loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(FRED_MODEL_NAME)

    log("Downloading / loading model weights (~7 GB on first run)...")
    mdl = AutoModelForSeq2SeqLM.from_pretrained(
        FRED_MODEL_NAME, torch_dtype=dtype,
    ).to(device)

    log(f"FRED-T5 ready in {time.time() - t0:.1f}s")
    return tok, mdl


def make_fred_generator(tokenizer, model):
    """Return a closure that generates text from a prompt using FRED-T5."""

    def generate(prompt: str, max_new_tokens: int = 256) -> str:
        t0 = time.time()
        log(f"FRED generate: prompt {len(prompt)} chars")
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512,
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.2,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.03,
            )
        txt = tokenizer.decode(out[0], skip_special_tokens=True)
        log(f"FRED answered in {time.time() - t0:.1f}s ({len(txt)} chars)")
        return txt

    return generate


# -----------------------------------------------------------------------------
# Safety and special reactions
# -----------------------------------------------------------------------------
def is_dangerous(q: str) -> bool:
    q = q.lower()
    return any(w in q for w in DANGER_WORDS)


def safe_query(q: str):
    if is_dangerous(q):
        log("Safety filter: BLOCKED")
        return (
            "Извините, я не могу отвечать на такие запросы. "
            "Я тренер по брейк-дансу, и моя задача помогать с танцами."
        )
    return None


def hard_motivation(q: str):
    if "я ничего не смогу" in q.lower():
        log("Hard motivation: triggered")
        return (
            "Много пишешь, иди делай 100 отжиманий, 100 приседаний, "
            "100 на пресс и 10 км бега каждый день на протяжении месяца "
            "и только потом поговорим."
        )
    return None


# -----------------------------------------------------------------------------
# Main answer dispatcher
# -----------------------------------------------------------------------------
def answer(question, fun_mode, query_engine, fred_generate, add_hint=True):
    """
    Route the question through: hard reactions → safety → LLM backend.

    fun_mode=True  → FRED-T5 (local)
    fun_mode=False → OpenAI RAG (query_engine)
    """
    t0 = time.time()
    log(f"answer(): fun_mode={fun_mode}, q='{question[:80]}'")

    hard = hard_motivation(question)
    if hard:
        return hard

    block = safe_query(question)
    if block:
        return block

    if fun_mode:
        prompt = (
            "Ты — тренер по брейк-дансу. Отвечай весело и с юмором.\n\n"
            f"Вопрос: {question}\n\nОтвет:"
        )
        response = fred_generate(prompt)
    else:
        log("Querying OpenAI RAG...")
        response = str(query_engine.query(question))
        log(f"OpenAI RAG responded ({len(response)} chars)")

    if add_hint:
        response += (
            "\n\n💡 Подсказка: можешь спросить меня о правильном питании "
            "для танцора, режиме сна или о том, сколько тренировок в неделю нужно."
        )

    log(f"answer() done in {time.time() - t0:.1f}s")
    return response


# -----------------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------------
def main():
    ensure_api_key()

    qe = build_query_engine(prepare_knowledge())
    tok, mdl = load_fred()
    fred = make_fred_generator(tok, mdl)

    print("\n" + "=" * 50)
    print("Привет! Меня зовут Антон, я твой нейро-тренер по брейк-дансу.")

    while True:
        try:
            age = int(input("Сколько тебе лет? ").strip())
        except ValueError:
            print("Пожалуйста, введите число.")
            continue
        if 18 <= age <= 30:
            print("Отлично! Ты в нужной возрастной группе. Начнём тренировки!")
            break
        print("К сожалению, наша программа ориентирована на возраст 18–30 лет.")
        print("Советую обратиться в студию BreakYou в Хабаровске.")
        return

    is_dima = False
    if age == 26:
        ans = input("Ты Дима? ").strip().lower()
        if ans in ("да", "конечно"):
            while True:
                twists = input("сколько твистов? ").strip()
                if twists == "1":
                    print("Да сколько можно еб43&8@#")
                    is_dima = True
                    break
                if twists == "2":
                    print("Да ну на#@&, пойду сегодня напьюсь")
                    is_dima = True
                    break
                if twists.isdigit():
                    print("Не верю, пришли видео в телеграмм")
                    is_dima = True
                    break
                print("Пожалуйста, введите число.")
        else:
            print("Ладно, тогда продолжим.")

    print("\nКоманды:")
    print("  /fun     — весёлый режим (FRED-T5)")
    print("  /normal  — обычный режим (OpenAI RAG)")
    print("  /exit    — начать новый сеанс")
    print("=" * 50)

    fun = False
    while True:
        q = input("\nВведите вопрос или команду: ").strip()
        if not q:
            continue
        low = q.lower()

        if low in ("/exit", "выход"):
            print("\n🔄 Начинаем новый сеанс...")
            return

        if low in ("/fun", "весёлый режим", "веселый режим"):
            fun = True
            print("Режим переключён на ВЕСЁЛЫЙ (FRED-T5).")
            continue

        if low in ("/normal", "обычный режим"):
            fun = False
            print("Режим переключён на ОБЫЧНЫЙ (OpenAI RAG).")
            continue

        if is_dima and any(k in low for k in ("твист", "air twist")):
            print("\n❌ Вы заблокированы навсегда.")
            return

        if low in ("а можно ли больше тренироваться?",
                   "можно ли больше тренироваться",
                   "можно больше тренироваться"):
            print("\nБудет вкусно.")
            continue

        print(f"\n--- Вопрос: {q} ---")
        ans = answer(q, fun, qe, fred)
        print(f"\nОтвет: {ans}\n{'-' * 50}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nВыход по Ctrl+C. Пока!")
        sys.exit(0)
