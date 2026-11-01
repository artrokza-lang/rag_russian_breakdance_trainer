# -*- coding: utf-8 -*-
"""
Streamlit web UI for the breakdance neuro-trainer.

Reuses the same RAG pipeline and LLM helpers as main.py,
but renders everything as a chat in the browser, with a
preparation screen (load models BEFORE the dialog starts)
and live logs.

Run:
    streamlit run app.py
"""
import io
import time
import contextlib
import traceback

import streamlit as st

from main import (
    answer,
    build_query_engine,
    clear_log,
    get_log,
    load_fred,
    make_fred_generator,
    prepare_knowledge,
)


# -----------------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Neuro-Trainer по брейк-дансу",
    page_icon="🕺",
    layout="wide",
)


# -----------------------------------------------------------------------------
# Session defaults
# -----------------------------------------------------------------------------
DEFAULTS = {
    "stage": "age",        # age | dima_name | dima_twists | preparing | chat | blocked
    "age": None,
    "is_dima": False,
    "fun_mode": False,
    "messages": [],
    "rag_ready": False,
    "fred_ready": False,
    "rag_log": "",
    "fred_log": "",
    "last_q_log": "",
}

for _k, _v in DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = list(_v) if isinstance(_v, list) else _v


def reset_session() -> None:
    for _k, _v in DEFAULTS.items():
        st.session_state[_k] = list(_v) if isinstance(_v, list) else _v
    clear_log()


# -----------------------------------------------------------------------------
# Cached heavy inits — load once per Streamlit process
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _get_query_engine():
    return build_query_engine(prepare_knowledge())


@st.cache_resource(show_spinner=False)
def _get_fred_generator():
    tok, mdl = load_fred()
    return make_fred_generator(tok, mdl)


def run_with_logs(fn):
    """Run fn(), capture stdout/stderr into a string. Returns (result, log, err)."""
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            result = fn()
        return result, buf.getvalue(), None
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        return None, buf.getvalue() + "\n" + tb, e


# -----------------------------------------------------------------------------
# Sidebar — mode + status + logs
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Настройки")

    st.session_state.fun_mode = st.toggle(
        "🎉 Весёлый режим (FRED-T5)",
        value=st.session_state.fun_mode,
        help="Выкл: OpenAI RAG. Вкл: локальная FRED-T5.",
    )

    st.divider()
    st.subheader("📊 Статус моделей")
    st.markdown(f"🎯 **RAG (OpenAI):** {'✅' if st.session_state.rag_ready else '❌'}")
    if st.session_state.fun_mode:
        st.markdown(f"🎉 **FRED-T5:** {'✅' if st.session_state.fred_ready else '❌'}")
    else:
        st.markdown("🎉 **FRED-T5:** ⚪ выключен")

    st.divider()
    if st.button("🔄 Новый сеанс", use_container_width=True):
        reset_session()
        st.rerun()

    full_log = "\n".join(get_log()) or "(пусто)"
    st.download_button(
        "💾 Скачать лог",
        data=full_log,
        file_name=f"neuro_trainer_log_{time.strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
        use_container_width=True,
    )

    with st.expander("📜 Логи (live)", expanded=False):
        st.code(full_log[-4000:] or "(пока ничего)", language="text")

    if st.session_state.age is not None:
        st.caption(f"Возраст: {st.session_state.age}")
    if st.session_state.is_dima:
        st.caption("👤 Режим Димы")


# -----------------------------------------------------------------------------
# Header + chat history
# -----------------------------------------------------------------------------
st.title("🕺 Neuro-Trainer по брейк-дансу")
st.caption("Привет! Меня зовут Антон, я твой нейро-тренер по брейк-дансу.")

for _msg in st.session_state.messages:
    with st.chat_message(_msg["role"]):
        st.markdown(_msg["content"])


# -----------------------------------------------------------------------------
# Stage routing
# -----------------------------------------------------------------------------
stage = st.session_state.stage

# ====== BLOCKED ======
if stage == "blocked":
    st.error("❌ Вы заблокированы навсегда.")
    st.info("Нажмите «Новый сеанс», чтобы начать заново.")
    st.stop()

# ====== AGE GATE ======
if stage == "age":
    with st.form("age_form"):
        age = st.number_input("Сколько тебе лет?", 1, 120, 25)
        ok = st.form_submit_button("Продолжить", use_container_width=True)
    if ok:
        if 18 <= age <= 30:
            st.session_state.age = int(age)
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Отлично! Ты в нужной возрастной группе. Начнём тренировки!",
            })
            if age == 26:
                st.session_state.messages.append({
                    "role": "assistant", "content": "Ты Дима?",
                })
                st.session_state.stage = "dima_name"
            else:
                st.session_state.stage = "preparing"
            st.rerun()
        else:
            st.error("К сожалению, программа для 18–30 лет.")
            st.info("Советую студию BreakYou в Хабаровске.")
    st.stop()

# ====== DIMA: NAME ======
if stage == "dima_name":
    a = st.chat_input("Да / Нет")
    if a:
        st.session_state.messages.append({"role": "user", "content": a})
        if a.strip().lower() in ("да", "конечно"):
            st.session_state.messages.append({
                "role": "assistant", "content": "Сколько твистов?",
            })
            st.session_state.stage = "dima_twists"
        else:
            st.session_state.messages.append({
                "role": "assistant", "content": "Ладно, продолжим.",
            })
            st.session_state.stage = "preparing"
        st.rerun()
    st.stop()

# ====== DIMA: TWISTS ======
if stage == "dima_twists":
    a = st.chat_input("Сколько твистов?")
    if a:
        st.session_state.messages.append({"role": "user", "content": a})
        raw = a.strip()
        if raw == "1":
            reply = "Да сколько можно еб43&8@#"
        elif raw == "2":
            reply = "Да ну на#@&, пойду сегодня напьюсь"
        elif raw.isdigit():
            reply = "Не верю, пришли видео в телеграмм"
        else:
            reply = "Пожалуйста, введите число."
        st.session_state.messages.append({"role": "assistant", "content": reply})
        if raw.isdigit():
            st.session_state.is_dima = True
            st.session_state.stage = "preparing"
        st.rerun()
    st.stop()

# ====== PREPARING — load models BEFORE the dialog ======
if stage == "preparing":
    st.subheader("⏳ Подготовка тренера")
    st.caption(
        "Загружаю RAG-индекс и (если включён весёлый режим) FRED-T5. "
        "Нажмите «Обновить», если колонки пустые."
    )

    col1, col2 = st.columns(2)

    # --- RAG (OpenAI) ---
    with col1:
        st.markdown("### 🎯 OpenAI RAG")
        if st.session_state.rag_ready:
            st.success("✅ готов")
        else:
            if st.button("▶️ Загрузить RAG", key="btn_rag", use_container_width=True):
                with st.spinner("Строю индекс (OpenAI embeddings)..."):
                    _, log_text, err = run_with_logs(_get_query_engine)
                    st.session_state.rag_log = log_text
                    if err is None:
                        st.session_state.rag_ready = True
                        st.success("✅ готов")
                    else:
                        st.error(f"❌ {type(err).__name__}: {err}")
                st.rerun()

        if st.session_state.rag_log:
            with st.expander("📜 Лог RAG", expanded=not st.session_state.rag_ready):
                st.code(st.session_state.rag_log[-6000:], language="text")

    # --- FRED ---
    with col2:
        st.markdown("### 🎉 FRED-T5 (локальная)")
        if not st.session_state.fun_mode:
            st.info("⚪ Выключен — не загружаю")
            st.session_state.fred_ready = False
        elif st.session_state.fred_ready:
            st.success("✅ готов")
        else:
            if st.button("▶️ Загрузить FRED (~7 ГБ)", key="btn_fred",
                         use_container_width=True):
                with st.spinner("Загружаю FRED-T5 (первый раз 5–15 минут)..."):
                    _, log_text, err = run_with_logs(_get_fred_generator)
                    st.session_state.fred_log = log_text
                    if err is None:
                        st.session_state.fred_ready = True
                        st.success("✅ готов")
                    else:
                        st.error(f"❌ {type(err).__name__}: {err}")
                st.rerun()

        if st.session_state.fred_log:
            with st.expander("📜 Лог FRED",
                             expanded=not st.session_state.fred_ready):
                st.code(st.session_state.fred_log[-6000:], language="text")

    # --- Actions ---
    st.divider()
    any_ready = st.session_state.rag_ready or (
        st.session_state.fun_mode and st.session_state.fred_ready
    )
    c1, c2, c3 = st.columns(3)
    if c1.button("▶️ Начать диалог", disabled=not any_ready,
                 use_container_width=True):
        st.session_state.stage = "chat"
        st.rerun()
    if c2.button("🔄 Обновить", use_container_width=True):
        st.rerun()
    if c3.button("⏭️ Пропустить", use_container_width=True):
        st.session_state.stage = "chat"
        st.rerun()

    st.caption(
        f"RAG: {'✅' if st.session_state.rag_ready else '❌'}  •  "
        f"FRED: {'✅' if st.session_state.fred_ready else ('⚪' if not st.session_state.fun_mode else '❌')}"
    )
    st.stop()

# ====== CHAT ======
if stage == "chat":
    c1, c2 = st.columns(2)
    c1.caption(f"🎯 RAG: {'✅' if st.session_state.rag_ready else '❌'}")
    c2.caption(
        f"🎉 FRED: {'✅' if st.session_state.fred_ready else ('⚪ выкл' if not st.session_state.fun_mode else '❌')}"
    )

    prompt = st.chat_input("Задай вопрос по брейк-дансу...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Dima easter egg
        if st.session_state.is_dima and any(
            k in prompt.lower() for k in ("твист", "twist")
        ):
            msg = "❌ Вы заблокированы навсегда."
            st.session_state.messages.append({"role": "assistant", "content": msg})
            st.session_state.stage = "blocked"
            with st.chat_message("assistant"):
                st.markdown(msg)
            st.rerun()

        with st.chat_message("assistant"):
            with st.spinner("Думаю..."):
                need_fred = st.session_state.fun_mode
                ready = (
                    st.session_state.fred_ready
                    if need_fred
                    else st.session_state.rag_ready
                )
                if not ready:
                    st.warning(
                        "Модель для этого режима ещё не загружена. "
                        "Вернитесь на экран «Подготовка»."
                    )
                    st.stop()

                clear_log()
                try:
                    qe = _get_query_engine() if not need_fred else None
                    fred = _get_fred_generator() if need_fred else None
                    ans, _, err = run_with_logs(
                        lambda: answer(
                            prompt,
                            fun_mode=need_fred,
                            query_engine=qe,
                            fred_generate=fred,
                        )
                    )
                    if err is not None:
                        st.error(f"❌ {type(err).__name__}: {err}")
                        with st.expander("📜 Traceback", expanded=True):
                            st.code("".join(get_log()), language="text")
                    else:
                        st.markdown(ans)
                        st.session_state.messages.append({
                            "role": "assistant", "content": ans,
                        })
                finally:
                    st.session_state.last_q_log = "\n".join(get_log())

            if st.session_state.last_q_log:
                with st.expander("📜 Логи этого запроса", expanded=False):
                    st.code(st.session_state.last_q_log[-6000:], language="text")
        st.rerun()
