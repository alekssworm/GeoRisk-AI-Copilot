import os

from rag.config import LLM_MAX_PROMPT_CHARS


class LLMClient:
    def __init__(self, model: str | None = None):
        self.model = model or os.getenv("GEORISK_LLM_MODEL", "gpt-4o-mini")
        self.api_key = os.getenv("OPENAI_API_KEY")

    def generate(self, prompt: str) -> str | None:
        if not self.api_key:
            return None

        if len(prompt) > LLM_MAX_PROMPT_CHARS:
            prompt = prompt[:LLM_MAX_PROMPT_CHARS]

        try:
            from openai import OpenAI

            client = OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model,
                temperature=0.1,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a radiation risk analysis assistant. "
                            "Answer only from the supplied context and cite sources."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            return response.choices[0].message.content
        except Exception:
            return "LLM generation failed, using retrieved evidence only."
