import os
from pinecone import Pinecone, ServerlessSpec

# 1) Read your API key
api_key     = os.getenv("PINECONE_API_KEY")

# 2) Set the Pinecone environment (from your host URL)
environment = "aped-4627-b74a"

# 3) Instantiate the client
pc = Pinecone(api_key=api_key, environment=environment)

# 4) Create or reuse your index
index_name = "aaofi-standards"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,               # for OpenAI embeddings
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

print(f"✅ Index '{index_name}' is ready!")
