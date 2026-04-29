import os
import chromadb
from chromadb.utils import embedding_functions

# Import the universal parser you just perfected!
from chunking import clean_universal_sec_html, chunk_clean_text


def build_vector_database():
    print("Initializing local ChromaDB...")
    # This creates a persistent local folder './chroma_db' to save your vectors permanently
    chroma_client = chromadb.PersistentClient(path="./chroma_db")

    # We use BAAI's bge-small model.
    # It is incredibly fast, runs completely offline, and is highly ranked on the Massive Text Embedding Benchmark (MTEB).
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-small-en-v1.5")

    # Create or load the collection (Think of a collection like a SQL table)
    collection = chroma_client.get_or_create_collection(
        name="sec_filings",
        embedding_function=emb_fn
    )

    # Dynamically find our Apple 10-K document
    aapl_dir = os.path.join("sec_data", "sec-edgar-filings", "AAPL", "10-K")
    target_file = None
    for root, dirs, files in os.walk(aapl_dir):
        for f in files:
            if f.endswith('.txt') or f.endswith('.html'):
                target_file = os.path.join(root, f)
                break
        if target_file:
            break

    if not target_file:
        print("Could not find AAPL filing to vectorize.")
        return

    print(f"Processing document for vectorization: {target_file}")

    # 1. Clean the text using your universal parser
    clean_text = clean_universal_sec_html(target_file)

    # 2. Chunk it and attach metadata (CRUCIAL for Phase 2 filtering)
    chunks = chunk_clean_text(clean_text, source_meta="AAPL_2024_10K")

    print(f"Embedding {len(chunks)} chunks and inserting into ChromaDB...")
    print("Note: The first time you run this, it will download the embedding model (approx 130MB).")

    # We must extract the text strings and metadata dictionaries from the LangChain Document objects
    documents_to_insert = [chunk.page_content for chunk in chunks]
    metadatas_to_insert = [chunk.metadata for chunk in chunks]

    # Create unique IDs for every single chunk
    ids_to_insert = [f"AAPL_2024_chunk_{i}" for i in range(len(chunks))]

    # Execute the database insertion
    collection.add(
        documents=documents_to_insert,
        metadatas=metadatas_to_insert,
        ids=ids_to_insert
    )

    print("\n[SUCCESS] Vector database built and populated!")
    print(
        f"Total documents currently in the 'sec_filings' collection: {collection.count()}")


if __name__ == "__main__":
    build_vector_database()
