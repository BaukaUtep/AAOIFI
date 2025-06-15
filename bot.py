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
# ── Language detection based on specific Unicode patterns ──
def detect_language(text: str) -> str:
    if re.search(r"[ңғүұқәі]", text.lower()):
        return 'kk'  # Kazakh
    elif re.search(r"[\u0500-\u052F]", text):  # Cyrillic Supplement
        return 'kk'
    elif re.search(r"[\u0600-\u06FF]", text):  # Arabic
        return 'ar'
    elif re.search(r"[\u0750-\u077F\uFB50-\uFDFF]", text):  # Urdu/Arabic extended
        return 'ur'
    elif re.search(r"[\u0400-\u04FF]", text):  # General Cyrillic (likely Russian)
        return 'ru'
    else:
        return 'en'

# ── Translation using GPT ──
def translate_text(text: str, target_lang: str) -> str:
    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": f"Translate the following text into {target_lang}. Preserve the meaning, style, and any Islamic finance technical terms (like Zakah, Mudarabah, etc.) without changing them."},
                {"role": "user", "content": text}
            ],
            temperature=0,
            max_tokens=512
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Translation Error: {str(e)}]"

# ── Main QA Function ──
def answer_question(question: str) -> str:
    # 1. Detect original language
    lang = detect_language(question)

    # 2. Translate question to English if needed
    if lang != 'en':
        eng_question = translate_text(question, target_lang='en')
    else:
        eng_question = question

    # 3. Get embedding and search in Pinecone
    resp = client.embeddings.create(model=EMBED_MODEL, input=eng_question)
    q_emb = resp.data[0].embedding

    qr = index.query(vector=q_emb, top_k=TOP_K, include_metadata=True)
    contexts = []
    for match in qr.matches:
        md = match.metadata
        txt = md.get("chunk_text", "")
        title = md.get("section_title", "")
        num = md.get("standard_number", "")
        contexts.append(f"{title} (Std {num}):\n{txt}")

    # 4. Generate English answer using only relevant excerpts
    system = {
        "role": "system",
        "content": (
            "You are a knowledgeable AAOIFI standards expert. "
            "Using only the provided excerpts, compose a coherent and detailed answer "
            "that explains and synthesizes the relevant sections. "
            "If the information is incomplete, clearly state what is missing. "
            "Maintain all technical terms in their original form."
        )
    }
    user = {
        "role": "user",
        "content": (
            "Here are the relevant AAOIFI excerpts:\n\n"
            + "\n---\n".join(contexts)
            + f"\n\nQuestion: {eng_question}\nAnswer:"
        )
    }
    chat = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[system, user],
        temperature=0.3,
        max_tokens=512
    )
    eng_answer = chat.choices[0].message.content.strip()

    # 5. Translate back into original language if needed
    if lang != 'en':
        final_answer = translate_text(eng_answer, target_lang=lang)
    else:
        final_answer = eng_answer

    return final_answer

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
