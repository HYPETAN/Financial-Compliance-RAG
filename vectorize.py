import os
import chromadb
from chromadb.utils import embedding_functions

# Import your masterpiece parser
from chunking import clean_universal_sec_html, chunk_clean_text


def batch_build_vector_database():
    print("Initializing Enterprise Local ChromaDB for Batch Processing...")
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-small-en-v1.5")

    collection = chroma_client.get_or_create_collection(
        name="sec_filings",
        embedding_function=emb_fn
    )

    base_dir = os.path.join("sec_data", "sec-edgar-filings")

    if not os.path.exists(base_dir):
        print("Data directory not found. Did ingest.py finish?")
        return

    # Iterate through all ticker folders (AAPL, MSFT, JPM, etc.)
    for ticker in os.listdir(base_dir):
        ticker_dir = os.path.join(base_dir, ticker, "10-K")
        if not os.path.exists(ticker_dir):
            continue

        # Iterate through all filing years for this ticker
        for accession_num in os.listdir(ticker_dir):
            accession_dir = os.path.join(ticker_dir, accession_num)

            # Find the actual text/html file
            target_file = None
            for f in os.listdir(accession_dir):
                if f.endswith('.txt') or f.endswith('.html'):
                    target_file = os.path.join(accession_dir, f)
                    break

            if target_file:
                print(f"\n{'-'*40}")
                print(f"Processing: {ticker} | Filing: {accession_num}")
                try:
                    # 1. Clean the massive HTML bloat
                    clean_text = clean_universal_sec_html(target_file)

                    # 2. Chunk it with highly specific metadata
                    # E.g., MSFT_0000062818-23-000034
                    source_tag = f"{ticker}_{accession_num}"
                    chunks = chunk_clean_text(
                        clean_text, source_meta=source_tag)

                    if not chunks:
                        continue

                    # 3. Extract data for ChromaDB
                    documents_to_insert = [
                        chunk.page_content for chunk in chunks]
                    metadatas_to_insert = [chunk.metadata for chunk in chunks]

                    # Unique IDs are strictly required so we don't overwrite previous runs
                    ids_to_insert = [
                        f"{source_tag}_chunk_{i}" for i in range(len(chunks))]

                    print(
                        f"Embedding {len(chunks)} chunks into the database...")
                    collection.add(
                        documents=documents_to_insert,
                        metadatas=metadatas_to_insert,
                        ids=ids_to_insert
                    )
                except Exception as e:
                    print(
                        f"[ERROR] Failed processing {ticker} {accession_num}: {e}")

    print("\n" + "="*50)
    print("[SUCCESS] Massive Vector Database Built!")
    print(
        f"Total documents currently in the 'sec_filings' collection: {collection.count()}")
    print("="*50)


if __name__ == "__main__":
    batch_build_vector_database()
