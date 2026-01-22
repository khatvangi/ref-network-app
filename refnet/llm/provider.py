"""
llm provider - abstraction for LLM backends.

supports:
- ollama (local, default)
- can be extended for openai, anthropic, etc.

usage:
    provider = OllamaProvider(model="qwen3:32b")
    response = provider.generate("What is the key finding?")
"""

import logging
import httpx
import json
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

logger = logging.getLogger("refnet.llm.provider")


@dataclass
class LLMResponse:
    """response from LLM."""
    text: str
    model: str
    tokens_used: int = 0
    duration_ms: float = 0.0
    raw: Optional[Dict[str, Any]] = None


class LLMProvider(ABC):
    """abstract base for LLM providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """provider name."""
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2000
    ) -> LLMResponse:
        """generate text from prompt."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """check if provider is available."""
        pass


class OllamaProvider(LLMProvider):
    """
    ollama local LLM provider.

    usage:
        provider = OllamaProvider(model="qwen3:32b")
        response = provider.generate("Summarize this paper...")
    """

    def __init__(
        self,
        model: str = "qwen3:32b",
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0
    ):
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self._client = None

    @property
    def client(self) -> httpx.Client:
        """lazy client initialization."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    @property
    def name(self) -> str:
        return f"ollama:{self.model}"

    def is_available(self) -> bool:
        """check if ollama is running and model is available."""
        try:
            resp = self.client.get(f"{self.base_url}/api/tags")
            if resp.status_code == 200:
                data = resp.json()
                models = [m["name"] for m in data.get("models", [])]
                return self.model in models
            return False
        except Exception as e:
            logger.warning(f"ollama not available: {e}")
            return False

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2000
    ) -> LLMResponse:
        """generate text using ollama."""
        import time
        start = time.time()

        # build request
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }

        if system:
            payload["system"] = system

        try:
            resp = self.client.post(
                f"{self.base_url}/api/generate",
                json=payload
            )

            if resp.status_code != 200:
                logger.error(f"ollama error: {resp.status_code} - {resp.text[:200]}")
                return LLMResponse(
                    text="",
                    model=self.model,
                    raw={"error": resp.text}
                )

            data = resp.json()
            duration = (time.time() - start) * 1000

            return LLMResponse(
                text=data.get("response", ""),
                model=self.model,
                tokens_used=data.get("eval_count", 0),
                duration_ms=duration,
                raw=data
            )

        except Exception as e:
            logger.error(f"ollama generate failed: {e}")
            return LLMResponse(
                text="",
                model=self.model,
                raw={"error": str(e)}
            )

    def generate_json(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.2
    ) -> Optional[Dict[str, Any]]:
        """generate and parse JSON response."""
        # add JSON instruction to system prompt
        json_system = (system or "") + "\n\nRespond ONLY with valid JSON. No other text."

        response = self.generate(
            prompt=prompt,
            system=json_system,
            temperature=temperature
        )

        if not response.text:
            return None

        # try to parse JSON
        text = response.text.strip()

        # handle markdown code blocks
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning(f"failed to parse JSON: {e}")
            logger.debug(f"raw response: {response.text[:500]}")
            return None

    def close(self):
        """close the client."""
        if self._client and not self._client.is_closed:
            self._client.close()
            self._client = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
