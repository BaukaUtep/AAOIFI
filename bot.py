import os
import time
import requests
import json
import re

from openai import OpenAI
from langdetect import detect
from pinecone import Pinecone       # ← вместо pinecone.init()

# ─── 1. Конфиг ────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN     = os.getenv("TELEGRAM_TOKEN")        # ваш Telegram token
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY")        # ваш OpenAI key
PINECONE_API_KEY   = os.getenv("PINECONE_API_KEY")      # ваш Pinecone key
PINECONE_ENV       = os.getenv("PINECONE_ENVIRONMENT")  # e.g. "aped-4627-b74a"
PINECONE_INDEX     = "aaofi-standards"

EMBED_MODEL        = "text-embedding-ada-002"
CHAT_MODEL         = "gpt-3.5-turbo"
TOP_K              = 5
TELEGRAM_API       = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

# ─── 2. Инициализация клиентов ─────────────────────────────────────────────────
openai = OpenAI(api_key=OPENAI_API_KEY)

# ← Здесь мы больше НЕ используем pinecone.init()
pc = Pinecone(
    api_key=PINECONE_API_KEY,
     environment=PINECONE_ENV
)
index = pc.Index(PINECONE_INDEX)

# ─── 3. Функции для работы с Telegram ─────────────────────────────────────────
def get_updates(offset=None, timeout=30):
    r = requests.get(f"{TELEGRAM_API}/getUpdates", params={"offset": offset, "timeout": timeout})
    return r.json().get("result", [])

def send_message(chat_id, text):
    requests.get(
        f"{TELEGRAM_API}/sendMessage",
        params={"chat_id": chat_id, "text": text}
    )

# ─── 4. Логика ответа на вопрос ────────────────────────────────────────────────
def answer_question(question: str) -> str:
    # ── 0) Проверяем, есть ли в вопросе кириллица ──────────────────────
    is_cyrillic = bool(re.search(r"[\u0400-\u04FF]", question))

    # ── 1) Если вопрос на кириллице, переводим его на английский ───────
    if is_cyrillic:
        tran = openai.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role":"system", "content":"Translate the following into English, preserving meaning and style:"},
                {"role":"user",   "content": question}
            ]
        )
        eng_question = tran.choices[0].message.content.strip()
    else:
        eng_question = question

    # ── 2) Embedding + поиск в Pinecone ───────────────────────────────
    resp = openai.embeddings.create(model=EMBED_MODEL, input=eng_question)
    q_emb = resp.data[0].embedding

    qr = index.query(vector=q_emb, top_k=TOP_K, include_metadata=True)
    contexts = []
    for match in qr.matches:
        md    = match.metadata
        txt   = md.get("chunk_text","")
        title = md.get("section_title","")
        num   = md.get("standard_number","")
        contexts.append(f"{title} (Std {num}):\n{txt}")

    # ── 3) Генерация ответа на английском ───────────────────────────────
    system = {
        "role":"system",
        "content":(
            "You are a knowledgeable AAOIFI standards expert. "
            "Using only the provided excerpts, compose a coherent and detailed answer "
            "that explains and synthesizes the relevant sections. "
            "If the information is incomplete, clearly state what is missing."
        )
    }
    user = {
        "role":"user",
        "content":(
            "Here are the relevant AAOIFI excerpts:\n\n"
            + "\n---\n".join(contexts)
            + f"\n\nQuestion: {eng_question}\nAnswer:"
        )
    }
    chat = openai.chat.completions.create(
        model=CHAT_MODEL,
        messages=[system, user],
        temperature=0.3,
        max_tokens=512
    )
    eng_answer = chat.choices[0].message.content.strip()

    # ── 4) Если изначальный вопрос был на кириллице, переводим ответ обратно ─
    if is_cyrillic:
        tran_back = openai.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role":"system","content":"Translate the following into Russian, preserving meaning and style:"},
                {"role":"user",  "content": eng_answer}
            ]
        )
        return tran_back.choices[0].message.content.strip()

    # ── 5) Иначе — возвращаем ответ на английском ─────────────────────────
    return eng_answer

# ─── 5. Основной polling-цикл ─────────────────────────────────────────────────
def main():
    offset = None
    while True:
        for u in get_updates(offset):
            offset = u["update_id"] + 1
            msg    = u.get("message", {})
            text   = msg.get("text")
            chat_id= msg.get("chat", {}).get("id")
            if not text or not chat_id:
                continue

            if text.startswith("/start"):
                send_message(chat_id, "👋 Salam! Ask me anything about AAOIFI standards.")
            else:
                send_message(chat_id, "⌛ Thinking…")
                try:
                    ans = answer_question(text)
                except Exception as e:
                    ans = "❌ Error:\n" + str(e)
                send_message(chat_id, ans)

        time.sleep(1)

if __name__ == "__main__":
    main()
