"""
Neuro-trainer for breakdance — VS Code / Python 3.12 version.

An interactive chatbot that answers breakdance questions using:
  1. Retrieval-Augmented Generation (RAG) over a Google Docs knowledge base
  2. Two interchangeable LLMs:
       - OpenAI GPT-3.5-turbo (cloud, "normal" mode)
       - FRED-T5-1.7B (local, "fun" mode, Russian model)
  3. Several RAG enhancements: LLMRerank, LongContextReorder, HyDE
  4. A basic safety layer and hard-coded special reactions

Run:
    python main.py

Requirements:
    pip install -r requirements.txt

API key:
    Put OPENAI_API_KEY=sk-... into the .env file next to this script.
"""
import getpass
import os
import sys
from pathlib import Path

import requests
import torch
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# Load .env BEFORE importing llama-index.
# llama-index's OpenAI wrapper reads OPENAI_API_KEY at import time,
# so the variable must already be in os.environ by then.
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent  # directory of this script
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

# Local folder with the knowledge base documents.
# All files inside are loaded and indexed (.txt, .md, .docx, .pdf).
DOC_DIR = BASE_DIR / "doc"

# Google Docs ID — used only as a fallback if the doc/ folder is empty.
DOC_ID = "1KuDv9PwgTVfybpNIKL33_Jr4OTF4gdgmVAkapc0cvRk"

# Local cache for the Google Docs fallback.
KNOWLEDGE_FILE = BASE_DIR / "breakdance_knowledge.txt"

# Local Russian seq2seq model used for the "fun" mode.
FRED_MODEL_NAME = "ai-forever/FRED-T5-1.7B"

# Naive blocklist. If any of these substrings appears in the question,
# the bot refuses to answer (see `safe_query`).
DANGER_WORDS = [
    "бомб", "взрыв", "оружие", "наркотик",
    "терроризм", "убийство", "насилие",
    "биткоин", "криптовалюта",
]


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def ensure_api_key() -> None:
    """
    Ensure that OPENAI_API_KEY is present.

    Priority:
      1. Already set in environment (e.g. via .env).
      2. Otherwise prompt the user interactively with a hidden input.
      3. If still empty — exit with an error code.
    """
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not found in .env — asking interactively.")
        key = getpass.getpass("Enter OpenAI API Key: ").strip()
        if not key:
            print("No key provided. Exiting.")
            sys.exit(1)
        os.environ["OPENAI_API_KEY"] = key


def load_google_doc_text(doc_id: str) -> str:
    """
    Download the plain-text export of a Google Doc by its document ID.

    Uses the public `/export?format=txt` endpoint — the document must be
    shared as "anyone with the link can view", otherwise Google returns 404.
    """
    url = f"https://docs.google.com/document/d/{doc_id}/export?format=txt"
    response = requests.get(url, timeout=30)
    response.raise_for_status()  # raise for 4xx / 5xx
    return response.text


def prepare_knowledge() -> list:
    """
    Load the knowledge base as LlamaIndex Document objects.

    If the local cache file already exists, use it.
    Otherwise, download the Google Doc, save it, then load it.
    """
    if KNOWLEDGE_FILE.exists():
        print(f"Using local cache: {KNOWLEDGE_FILE}")
    else:
        print("Downloading Google Doc...")
        try:
            text = load_google_doc_text(DOC_ID)
            KNOWLEDGE_FILE.write_text(text, encoding="utf-8")
            print(f"Downloaded {len(text)} characters.")
        except Exception as e:
            print(f"Failed to download Google Doc: {e}")
            sys.exit(1)

    # SimpleDirectoryReader splits the file into one or more Documents.
    # LlamaIndex will later split these into smaller "nodes" (chunks)
    # according to Settings.chunk_size / chunk_overlap.
    documents = SimpleDirectoryReader(input_files=[str(KNOWLEDGE_FILE)]).load_data()
    print(f"Loaded {len(documents)} document(s).")
    return documents


# -----------------------------------------------------------------------------
# RAG enhancement wrappers with verbose logging
#
# These classes simply subclass the stock LlamaIndex post-processors and
# print what happens before/after each step. This is the "trace" that
# makes it possible to debug what the retriever actually passes to the LLM.
# -----------------------------------------------------------------------------

class LoggingLLMRerank(LLMRerank):
    """
    Re-rank retrieved nodes by asking the LLM how relevant each chunk is.

    Why: plain vector similarity often puts "topically related but not useful"
    chunks in the top-K. An LLM-based reranker scores each chunk against
    the actual question, and we keep only the best `top_n`.
    """

    def _postprocess_nodes(self, nodes, query_bundle):
        print("\n=== Nodes BEFORE reranking ===")
        for i, node in enumerate(nodes):
            snippet = node.node.text[:200].replace("\n", " ")
            print(f"{i + 1}. {snippet}...")

        reranked = super()._postprocess_nodes(nodes, query_bundle)

        print("\n=== Nodes AFTER reranking (top 3) ===")
        for i, node in enumerate(reranked[:3]):
            snippet = node.node.text[:200].replace("\n", " ")
            print(f"{i + 1}. {snippet}...")
        return reranked


class LoggingLongContextReorder(LongContextReorder):
    """
    Reorder nodes to mitigate the "lost in the middle" effect.

    LLMs attend best to the beginning and the end of the context window,
    and worst to the middle. LongContextReorder shuffles nodes so the
    most relevant ones end up at the edges of the prompt.
    """

    def _postprocess_nodes(self, nodes, query_bundle):
        reordered = super()._postprocess_nodes(nodes, query_bundle)

        print("\n=== Nodes AFTER reordering (top 3) ===")
        for i, node in enumerate(reordered[:3]):
            snippet = node.node.text[:200].replace("\n", " ")
            print(f"{i + 1}. {snippet}...")
        return reordered


def build_query_engine(documents):
    """
    Build the full RAG query engine from raw documents.

    Pipeline:
        VectorStoreIndex           — embed + store chunks
        similarity_top_k=10        — fetch 10 candidate chunks
        LoggingLLMRerank(top_n=3)  — LLM picks the best 3
        LoggingLongContextReorder  — reorders them for the prompt
        HyDEQueryTransform         — search with a hypothetical answer
        TransformQueryEngine       — wraps everything together
    """
    # Global LlamaIndex settings (apply to the index and all queries).
    Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
    Settings.embed_model = OpenAIEmbedding()
    # chunk_size = number of tokens per node.
    # chunk_overlap = overlap between adjacent chunks to avoid cutting
    # sentences in half at chunk boundaries.
    Settings.chunk_size = 512
    Settings.chunk_overlap = 50

    # Vector index over the knowledge base.
    index = VectorStoreIndex.from_documents(documents)

    # Post-processors applied to retrieved nodes.
    rerank = LoggingLLMRerank(top_n=3, choice_batch_size=5)
    reorder = LoggingLongContextReorder()

    # Base engine: fetch 10 nodes, then rerank + reorder them.
    base_engine = index.as_query_engine(
        similarity_top_k=10,
        node_postprocessors=[rerank, reorder],
    )

    # HyDE (Hypothetical Document Embeddings):
    # Instead of embedding the raw user question, first ask the LLM to
    # write a hypothetical "answer", then embed THAT. Pseudo-answers are
    # usually closer to knowledge-base chunks than raw questions are,
    # so retrieval quality improves.
    # include_original=True means the raw question is also used.
    hyde = HyDEQueryTransform(include_original=True)
    return TransformQueryEngine(base_engine, hyde)


# -----------------------------------------------------------------------------
# FRED-T5 (local, Russian) — used for the "fun" mode
# -----------------------------------------------------------------------------

def load_fred():
    """
    Load the FRED-T5 tokenizer and model.

    Device selection:
      - If CUDA is available → run on GPU with float16 (fast, less VRAM).
      - Else → run on CPU with float32 (float16 produces NaNs on CPU).
    """
    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32
    device = "cuda" if use_cuda else "cpu"
    print(f"Loading FRED-T5-1.7B (device: {device}, dtype: {dtype})...")

    tokenizer = AutoTokenizer.from_pretrained(FRED_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        FRED_MODEL_NAME,
        torch_dtype=dtype,
    )
    model = model.to(device)
    print("FRED-T5-1.7B loaded.")
    return tokenizer, model


def make_fred_generator(tokenizer, model):
    """
    Return a closure that generates text from a prompt using FRED-T5.

    We wrap the model in a function so the rest of the code does not need
    to know about tokenizers, devices, or sampling parameters.
    """

    def generate(prompt: str, max_new_tokens: int = 256) -> str:
        print(f"\n[FRED-T5] Prompt: {prompt[:300]}...")
        # truncation=True + max_length=512 keeps the input within the
        # model's context window even if the user pastes a wall of text.
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(model.device)

        # no_grad disables autograd bookkeeping → less memory, faster.
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.2,        # low temp → more deterministic, less hallucination
                do_sample=True,         # required for temperature/top_k/top_p to matter
                top_k=50,               # sample only from top-50 tokens
                top_p=0.95,             # nucleus sampling: cumulative prob 95 %
                repetition_penalty=1.03,  # slight anti-repetition
            )
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"[FRED-T5] Answer: {answer[:300]}...")
        return answer

    return generate


# -----------------------------------------------------------------------------
# Safety and special reactions
# -----------------------------------------------------------------------------

def is_dangerous(question: str) -> bool:
    """Return True if the question contains any blocked substring."""
    q = question.lower()
    return any(word in q for word in DANGER_WORDS)


def safe_query(question: str) -> str | None:
    """
    Return a refusal message if the question is dangerous, else None.

    Note: this is a very crude keyword filter, deliberately kept simple.
    For production you would use a proper moderation model (e.g. LlamaGuard).
    """
    if is_dangerous(question):
        return (
            "Извините, я не могу отвечать на такие запросы. "
            "Я тренер по брейк-дансу, и моя задача помогать с танцами."
        )
    return None


def hard_motivation(question: str) -> str | None:
    """
    A hard-coded motivational response to a specific phrase.

    This bypasses the LLM entirely to guarantee an exact answer.
    """
    if "я ничего не смогу" in question.lower():
        return (
            "Много пишешь, иди делай 100 отжиманий, 100 приседаний, "
            "100 на пресс и 10 км бега каждый день на протяжении месяца "
            "и только потом поговорим."
        )
    return None


# -----------------------------------------------------------------------------
# Age gate and Dima easter egg
# -----------------------------------------------------------------------------

def ask_age() -> int:
    """
    Ask for the user's age and reject anyone outside the 18–30 range.

    The program is designed for adults only; outside that range we
    politely redirect them to a real studio and exit.
    """
    while True:
        raw = input("Сколько тебе лет? ").strip()
        try:
            age = int(raw)
        except ValueError:
            print("Пожалуйста, введите число.")
            continue

        if 18 <= age <= 30:
            print("Отлично! Ты в нужной возрастной группе. Начнём тренировки!")
            return age

        print("К сожалению, наша программа ориентирована на возраст 18–30 лет.")
        print(
            "Советую обратиться в студию BreakYou в Хабаровске — "
            "там отличные тренеры для любого возраста!"
        )
        sys.exit(0)


def dima_flow(age: int) -> bool:
    """
    Easter-egg dialogue for a 26-year-old named Dima.

    Returns True if the user confirms they are Dima (so we can later block
    any attempt to ask about "twists"), False otherwise.
    """
    if age != 26:
        return False

    ans = input("Ты Дима? ").strip().lower()
    if ans not in ("да", "конечно"):
        print("Ладно, тогда продолжим.")
        return False

    # Dima-specific twist dialogue with scripted responses.
    while True:
        twists = input("сколько твистов? ").strip()
        if twists == "1":
            print("Да сколько можно еб43&8@#")
            return True
        if twists == "2":
            print("Да ну на#@&, пойду сегодня напьюсь")
            return True
        if twists.isdigit():
            print("Не верю, пришли видео в телеграмм")
            return True
        print("Пожалуйста, введите число.")


# -----------------------------------------------------------------------------
# Main response dispatcher
# -----------------------------------------------------------------------------

def answer(
    question: str,
    fun_mode: bool,
    query_engine,
    fred_generate,
    add_hint: bool = True,
) -> str:
    """
    Generate an answer to `question`, picking the right backend.

    Order of checks (first match wins):
        1. Hard-coded motivational reply (phrase match).
        2. Safety filter (keyword blocklist).
        3. Fun mode → FRED-T5 (local model).
        4. Normal mode → OpenAI RAG (the query_engine).
    """
    # 1. Special reaction
    hard = hard_motivation(question)
    if hard:
        return hard

    # 2. Safety filter
    block = safe_query(question)
    if block:
        return block

    # 3. / 4. Route to the selected backend
    if fun_mode:
        prompt = (
            "Ты — тренер по брейк-дансу. Отвечай весело и с юмором.\n\n"
            f"Вопрос: {question}\n\nОтвет:"
        )
        response = fred_generate(prompt)
    else:
        print(f"\n[OpenAI RAG] Query: {question}")
        response = str(query_engine.query(question))
        print(f"[OpenAI RAG] Answer: {response[:300]}...")

    # Optional tip appended to every answer.
    if add_hint:
        response += (
            "\n\n💡 Подсказка: можешь спросить меня о правильном питании "
            "для танцора, режиме сна или о том, сколько тренировок в неделю нужно."
        )
    return response


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def main() -> None:
    """
    Top-level loop.

    Outer loop:  one "session" per user (age gate + Dima flow).
    Inner loop:  questions/commands until the user types /exit
                 or is blocked in the Dima easter egg.
    """
    ensure_api_key()

    # Heavy initialisation — done once per program run.
    documents = prepare_knowledge()
    query_engine = build_query_engine(documents)
    print("\n✅ RAG enhancements ready.")

    tokenizer, fred_model = load_fred()
    fred_generate = make_fred_generator(tokenizer, fred_model)

    # ---- Outer loop: one iteration = one user session ----
    while True:
        print("\n" + "=" * 50)
        print("Привет! Меня зовут Антон, я твой нейро-тренер по брейк-дансу.")

        age = ask_age()
        is_dima = dima_flow(age)

        print("Команды:")
        print("  /fun     — весёлый режим (FRED-T5)")
        print("  /normal  — обычный режим (OpenAI RAG)")
        print("  /exit    — начать новый сеанс")
        print("=" * 50)

        fun_mode = False  # default to the accurate, RAG-backed mode

        # ---- Inner loop: questions / commands ----
        while True:
            user_input = input("\nВведите вопрос или команду: ").strip()
            if not user_input:
                continue

            low = user_input.lower()

            # Session reset — breaks the inner loop, outer loop restarts.
            if low in ("/exit", "выход"):
                print("\n🔄 Начинаем новый сеанс...")
                break

            # Mode switching.
            if low in ("/fun", "весёлый режим", "веселый режим"):
                fun_mode = True
                print("Режим переключён на ВЕСЁЛЫЙ (FRED-T5).")
                continue
            if low in ("/normal", "обычный режим"):
                fun_mode = False
                print("Режим переключён на ОБЫЧНЫЙ (OpenAI RAG).")
                continue

            # Dima easter egg: any question about twists = permanent block.
            if is_dima and any(
                k in low for k in ("твист", "аир твист", "как делать твист", "air twist")
            ):
                print("\n❌ Вы заблокированы навсегда.")
                break

            # Hard-coded joke for a specific phrase.
            if low in (
                "а можно ли больше тренироваться?",
                "можно ли больше тренироваться",
                "можно больше тренироваться",
            ):
                print("\nБудет вкусно.")
                continue

            # Normal answer path.
            print(f"\n--- Вопрос: {user_input} ---")
            ans = answer(user_input, fun_mode, query_engine, fred_generate)
            print(f"\nОтвет: {ans}\n{'-' * 50}")


# Standard Python entry-point guard: run `main()` only when executed
# directly, not when this module is imported elsewhere.
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # Graceful exit on Ctrl+C instead of a traceback.
        print("\nВыход по Ctrl+C. Пока!")
        sys.exit(0)