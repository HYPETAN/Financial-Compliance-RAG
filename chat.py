import os
from openai import OpenAI
from search import HybridRetriever


class FinancialAssistant:
    def __init__(self):
        self.retriever = HybridRetriever()

        # --- THE AIR-GAP TRICK ---
        # We use the OpenAI library, but route it to your local Ollama server.
        # This means zero data ever leaves your laptop.
        self.client = OpenAI(
            base_url='http://localhost:11434/v1',
            api_key='ollama',  # Required by the library, but not actually checked
        )

    def generate_answer(self, query):
        # 1. Get the highly curated context from your Engine
        top_chunks = self.retriever.search_and_rerank(query, top_k=3)

        # 2. Format the context for the LLM
        context_string = ""
        for score, doc_id, text, meta in top_chunks:
            context_string += f"\n[CITATION_ID: {doc_id} | SOURCE: {meta['source']}]\nCONTENT: {text}\n"

        # 3. The Strict Enterprise Prompt
        system_prompt = """
        You are a strict SEC Compliance Assistant. 
        You must answer the user's query using ONLY the context provided below.
        
        RULES:
        1. If the answer is not contained in the context, you must reply: "I cannot answer this based on the provided SEC filings." Do not guess.
        2. You MUST cite your sources at the end of relevant sentences using the [CITATION_ID].
        3. Be professional, concise, and analytical.
        
        CONTEXT:
        {context}
        """

        print("\nSynthesizing answer locally (Zero Data Leakage)...\n")

        # 4. Call the Local LLM
        response = self.client.chat.completions.create(
            model="llama3",  # We call the local model you downloaded
            messages=[
                {"role": "system", "content": system_prompt.format(
                    context=context_string)},
                {"role": "user", "content": query}
            ],
            temperature=0.1  # Low temperature prevents hallucination
        )

        return response.choices[0].message.content


if __name__ == "__main__":
    assistant = FinancialAssistant()

    print("\n" + "="*60)
    print("Welcome to the Air-Gapped SEC RAG Assistant")
    print("="*60)

    while True:
        user_query = input("\nAsk the SEC Assistant (or type 'quit'): ")
        if user_query.lower() in ['quit', 'exit']:
            print("Shutting down...")
            break

        answer = assistant.generate_answer(user_query)

        print("\n" + "="*60)
        print("AI RESPONSE:")
        print("="*60)
        print(answer)
        print("="*60 + "\n")
