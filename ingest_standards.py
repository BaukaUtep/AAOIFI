import json, os, math
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc     = Pinecone(api_key=os.getenv("PINECONE_API_KEY"),
                  environment="aped-4627-b74a")
index  = pc.Index("aaofi-standards")

with open("standards.json", encoding="utf-8") as f:
    chunks = json.load(f)

# 1) Build all your vectors
vectors = []
for i, chunk in enumerate(chunks, start=1):
    print(f"⏳ Embedding chunk {i}/{len(chunks)} – {chunk['_id']}", flush=True)
    resp = openai.embeddings.create(
        model="text-embedding-ada-002",
        input=chunk["chunk_text"]
    )
    emb = resp.data[0].embedding
    vectors.append({
        "id":       chunk["_id"],
        "values":   emb,
        "metadata": {
            "standard_number": chunk["standard_number"],
            "section_title":   chunk["section_title"],
            "chunk_text":      chunk["chunk_text"],
            "keywords":        chunk.get("keywords", [])
        }
    })

# 2) Upsert in batches of, say, 50 vectors at a time
batch_size = 50
total_batches = math.ceil(len(vectors) / batch_size)
for batch_num in range(total_batches):
    start = batch_num * batch_size
    end   = start + batch_size
    batch = vectors[start:end]
    print(f"➡️ Upserting batch {batch_num+1}/{total_batches} ({len(batch)} vectors)", flush=True)
    index.upsert(vectors=batch)

print("✅ All chunks ingested!")
