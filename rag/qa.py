from rag.config import RAG_INDEX_PATH
from rag.llm import LLMClient
from rag.store import TfidfRAGStore


class RAGAssistant:
    def __init__(
        self,
        store: TfidfRAGStore | None = None,
        llm_client: LLMClient | None = None,
    ):
        self.store = store or TfidfRAGStore.load(RAG_INDEX_PATH)
        self.llm_client = llm_client or LLMClient()

    def _build_prompt(self, question: str, results: list[dict]) -> str:
        context_blocks = []
        for result in results:
            citation = result["chunk"].as_citation(result["rank"], result["score"])
            context_blocks.append(
                f"{citation['citation_id']} {citation['label']}\n{result['text']}"
            )
        context = "\n\n".join(context_blocks)
        return f"""Question: {question}

Context:
{context}

Instructions:
- Answer in 3 to 6 concise sentences.
- Use citations like [1] when making factual claims.
- If the context is insufficient, say what is missing.
"""

    def _fallback_answer(self, question: str, results: list[dict]) -> str:
        if not results:
            return (
                "No relevant document chunks were found. Upload technical PDF documents, "
                "then ask a question grounded in those sources."
            )

        snippets = []
        for result in results[:3]:
            citation = result["chunk"].as_citation(result["rank"], result["score"])
            text = result["text"]
            if len(text) > 360:
                text = text[:357].rstrip() + "..."
            snippets.append(f"{citation['citation_id']} {text}")
        return (
            "Based on the retrieved technical context, the most relevant evidence is: "
            + " ".join(snippets)
            + " Configure OPENAI_API_KEY to enable full LLM synthesis."
        )

    def answer(self, question: str, top_k: int = 4) -> dict:
        results = self.store.search(question, top_k=top_k)
        prompt = self._build_prompt(question, results)
        generated = self.llm_client.generate(prompt)
        if not generated:
            generated = self._fallback_answer(question, results)

        citations = [
            result["chunk"].as_citation(rank=result["rank"], score=result["score"])
            for result in results
        ]
        retrieved_context = [
            {
                "citation_id": citation["citation_id"],
                "source": citation["source"],
                "page": citation["page"],
                "score": citation["score"],
                "text": result["text"],
            }
            for result, citation in zip(results, citations, strict=False)
        ]
        return {
            "answer": generated,
            "citations": citations,
            "retrieved_context": retrieved_context,
        }
