import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
import numpy as np
from sentence_transformers import CrossEncoder


class HybridRetriever:
    def __init__(self):
        print("Booting up Hybrid Retrieval & Re-ranking Engine...")

        # 1. Connect to our existing local ChromaDB (Bi-Encoder)
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="BAAI/bge-small-en-v1.5")
        self.collection = self.chroma_client.get_collection(
            name="sec_filings", embedding_function=self.emb_fn)

        # 2. Build Keyword DB (Sparse Encoder)
        db_data = self.collection.get(include=["documents", "metadatas"])
        self.documents = db_data['documents']
        self.ids = db_data['ids']
        self.metadatas = db_data['metadatas']  # Added to fetch citations later

        tokenized_docs = [doc.lower().split(" ") for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_docs)

        # 3. Load the Cross-Encoder for Re-ranking
        # This model is specifically trained by Microsoft (MS MARCO) to score search relevance
        self.cross_encoder = CrossEncoder(
            'cross-encoder/ms-marco-MiniLM-L-6-v2')

        print(f"Engine Ready. Indexed {len(self.documents)} documents.\n")

    def vector_search(self, query, top_k=15):
        results = self.collection.query(query_texts=[query], n_results=top_k)
        return results['ids'][0]

    def keyword_search(self, query, top_k=15):
        tokenized_query = query.lower().split(" ")
        doc_scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(doc_scores)[::-1][:top_k]
        return [self.ids[i] for i in top_indices]

    def reciprocal_rank_fusion(self, vector_results, keyword_results, k=60):
        fused_scores = {}
        for rank, doc_id in enumerate(vector_results):
            fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1 / (k + rank)
        for rank, doc_id in enumerate(keyword_results):
            fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1 / (k + rank)

        sorted_fused = sorted(fused_scores.items(),
                              key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, score in sorted_fused]

    def search_and_rerank(self, query, top_k=3):
        """The Master Function: Vector + BM25 -> RRF -> Cross-Encoder Rerank"""
        print(f"Executing deep search for: '{query}'...")

        # Phase A: Fast Retrieval (Get top 15 candidates)
        vec_ids = self.vector_search(query, top_k=15)
        kw_ids = self.keyword_search(query, top_k=15)
        fused_ids = self.reciprocal_rank_fusion(vec_ids, kw_ids)[:15]

        # Fetch the actual text for the candidates
        candidate_docs = self.collection.get(ids=fused_ids)['documents']
        candidate_metadatas = self.collection.get(ids=fused_ids)['metadatas']

        # Phase B: Precision Re-ranking (Cross-Encoder)
        # Create (query, document) pairs to feed the neural network
        cross_inp = [[query, doc] for doc in candidate_docs]
        cross_scores = self.cross_encoder.predict(cross_inp)

        # Sort the documents based on the Cross-Encoder's highly accurate scores
        scored_docs = zip(cross_scores, fused_ids,
                          candidate_docs, candidate_metadatas)
        sorted_docs = sorted(scored_docs, key=lambda x: x[0], reverse=True)

        # Return the ultimate top_k winners, including their metadata for citations
        return sorted_docs[:top_k]


if __name__ == "__main__":
    retriever = HybridRetriever()
    test_query = "What are the regulatory risks associated with the App Store?"

    results = retriever.search_and_rerank(test_query, top_k=3)

    print("\n--- FINAL RE-RANKED RESULTS ---")
    for score, doc_id, text, meta in results:
        print(
            f"\n[Score: {score:.2f} | Source: {meta['source']} | ID: {doc_id}]")
        print(text[:250] + "...\n" + "-"*50)
