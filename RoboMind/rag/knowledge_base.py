import chromadb

DB_PATH = "rag/chroma_db"

client = chromadb.PersistentClient(path=DB_PATH)

collection = client.get_or_create_collection(
    name="robomind_knowledge"
)


def add_document(text, source):
    collection.add(
        documents=[text],
        metadatas=[{"source": source}],
        ids=[source]
    )


def search_knowledge(query, n_results=3):
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )

    return results["documents"][0]