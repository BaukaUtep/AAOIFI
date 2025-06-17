import json, os, math, time
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

# ─── API Ключи ─────────────────────────────────────────────────────
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc     = Pinecone(api_key=os.getenv("PINECONE_API_KEY"),
                  environment="aped-4627-b74a")
index  = pc.Index("aaofi-standards")

# ─── Загрузка JSON ─────────────────────────────────────────────────
with open("standards.json", encoding="utf-8") as f:
    chunks = json.load(f)

# ─── Генерация embedding ───────────────────────────────────────────
vectors = []
for i, chunk in enumerate(chunks, start=1):
    print(f"⏳ Embedding chunk {i}/{len(chunks)} – {chunk['_id']}", flush=True)
    try:
        resp = openai.embeddings.create(
            model="text-embedding-ada-002",
            input=chunk["chunk_text"]
        )
        emb = resp.data[0].embedding
    except Exception as e:
        print(f"❌ Error embedding chunk {i} ({chunk['_id']}): {e}")
        continue

    # Очистка и нормализация метаданных
    vectors.append({
        "id":     chunk["_id"],
        "values": emb,
        "metadata": {
            "standard_number": str(chunk.get("standard_number") or ""),
            "standard_name":   str(chunk.get("standard_name") or ""),
            "section_number":  str(chunk.get("section_number") or ""),
            "section_title":   str(chunk.get("section_title") or ""),
            "paragraph_id":    str(chunk.get("paragraph_id") or ""),
            "chunk_text":      str(chunk.get("chunk_text") or ""),
            "keywords":        ", ".join(chunk.get("keywords", []))  # список в строку
        }
    })

    time.sleep(0.5)  # Пауза для API стабильности

# ─── Загрузка в Pinecone батчами ───────────────────────────────────
batch_size = 50
total_batches = math.ceil(len(vectors) / batch_size)
for batch_num in range(total_batches):
    start = batch_num * batch_size
    end   = start + batch_size
    batch = vectors[start:end]
    print(f"➡️ Upserting batch {batch_num+1}/{total_batches} ({len(batch)} vectors)", flush=True)
    try:
        index.upsert(vectors=batch)
    except Exception as e:
        print(f"❌ Error during upsert batch {batch_num+1}: {e}")
        continue

print("✅ All chunks ingested!")
