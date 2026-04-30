import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
import numpy as np


class HybridRetriever:
    def __init__(self):
        print("Booting up Hybrid Retrieval Engine...")

        # 1. Connect to our existing local ChromaDB
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="BAAI/bge-small-en-v1.5")
        self.collection = self.chroma_client.get_collection(
            name="sec_filings", embedding_function=self.emb_fn)

        # 2. Load all documents into memory to build the BM25 Keyword Index
        # (Note: For 239 chunks, RAM is fine. For 2 million, we would use Elasticsearch/Weaviate)
        db_data = self.collection.get(include=["documents", "metadatas"])
        self.documents = db_data['documents']
        self.ids = db_data['ids']

        # Tokenize documents for BM25 (split by spaces and lowercase)
        tokenized_docs = [doc.lower().split(" ") for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        print(
            f"Engine Ready. Indexed {len(self.documents)} documents for Hybrid Search.\n")

    def vector_search(self, query, top_k=5):
        """Standard semantic search using ChromaDB."""
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        # Returns a list of document IDs
        return results['ids'][0]

    def keyword_search(self, query, top_k=5):
        """Exact keyword matching using BM25."""
        tokenized_query = query.lower().split(" ")
        # Get BM25 scores for all documents
        doc_scores = self.bm25.get_scores(tokenized_query)

        # Get the indices of the highest scoring documents
        top_indices = np.argsort(doc_scores)[::-1][:top_k]

        # Return the corresponding IDs
        return [self.ids[i] for i in top_indices]

    def reciprocal_rank_fusion(self, vector_results, keyword_results, k=60):
        """
        Merges the two ranked lists using the RRF algorithm.
        Formula: score = 1 / (k + rank)
        """
        fused_scores = {}

        # Score Vector Results
        for rank, doc_id in enumerate(vector_results):
            fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1 / (k + rank)

        # Score Keyword Results
        for rank, doc_id in enumerate(keyword_results):
            fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1 / (k + rank)

        # Sort documents by their new fused score
        sorted_fused = sorted(fused_scores.items(),
                              key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, score in sorted_fused]

    def search(self, query, top_k=5):
        """The Master Function: Runs both searches and fuses them."""
        print(f"QUERY: '{query}'")

        vec_ids = self.vector_search(query, top_k=10)
        kw_ids = self.keyword_search(query, top_k=10)

        fused_ids = self.reciprocal_rank_fusion(vec_ids, kw_ids)[:top_k]

        # Fetch the actual text for the winning IDs
        final_docs = self.collection.get(ids=fused_ids)['documents']

        return final_docs


if __name__ == "__main__":
    retriever = HybridRetriever()

    # A highly specific query that tests both semantic meaning and exact keyword tracking
    test_query = "What are the risks associated with the App Store and third-party developers?"

    results = retriever.search(test_query, top_k=3)

    print("\n--- TOP 3 HYBRID SEARCH RESULTS ---")
    for i, doc in enumerate(results, 1):
        print(f"\n[RESULT {i}]")
        # Print the first 300 characters to keep the terminal clean
        print(doc[:300] + "...\n" + "-"*50)
