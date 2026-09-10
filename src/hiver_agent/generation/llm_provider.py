import os
import json
import requests
from typing import Dict, Any, Optional
from hiver_agent.utils.cache import SimpleCache
from hiver_agent.utils.logging import get_logger

logger = get_logger("generation.llm_provider")

class LLMProvider:
    """
    Abstract LLM Provider supporting OpenAI, Anthropic, Ollama, and Mock/Offline fallbacks.
    Configurable via environment variables LLM_PROVIDER and LLM_MODEL.
    """
    
    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        cache_dir: str = ".cache/llm_cache"
    ):
        self.provider = (provider or os.getenv("LLM_PROVIDER", "mock")).lower()
        self.model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.cache = SimpleCache(cache_dir=cache_dir)
        
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        
        # Fallback to mock if requested provider lacks API key
        if self.provider == "openai" and not self.openai_key:
            logger.warning("OPENAI_API_KEY not found. Falling back to 'mock' LLM Provider.")
            self.provider = "mock"
        elif self.provider == "anthropic" and not self.anthropic_key:
            logger.warning("ANTHROPIC_API_KEY not found. Falling back to 'mock' LLM Provider.")
            self.provider = "mock"

        logger.info(f"Initialized LLMProvider: provider='{self.provider}', model='{self.model}'")

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        """Generate response string from prompt with caching."""
        cache_key = f"{self.provider}:{self.model}:{temperature}:{prompt}"
        cached_val = self.cache.get(cache_key)
        if cached_val:
            return cached_val

        if self.provider == "openai":
            res = self._call_openai(prompt, temperature)
        elif self.provider == "anthropic":
            res = self._call_anthropic(prompt, temperature)
        elif self.provider == "ollama":
            res = self._call_ollama(prompt, temperature)
        else:
            res = self._call_mock(prompt)

        self.cache.set(cache_key, res)
        return res

    def _call_openai(self, prompt: str, temperature: float) -> str:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.openai_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "response_format": {"type": "json_object"}
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def _call_anthropic(self, prompt: str, temperature: float) -> str:
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.anthropic_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "max_tokens": 1000,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()["content"][0]["text"]

    def _call_ollama(self, prompt: str, temperature: float) -> str:
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json"
        }
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()["response"]

    def _call_mock(self, prompt: str) -> str:
        """Deterministic mock fallback when API keys are not provided."""
        # Simple extraction of key details for realistic mock response
        if "relevance" in prompt and "correctness" in prompt:
            # LLM Judge prompt request
            return json.dumps({
                "relevance": 5,
                "correctness": 4,
                "grounding": 5,
                "helpfulness": 4,
                "tone": 5,
                "unsupported_claims": False,
                "overall_score": 4,
                "reason": "The reply directly addresses the customer query and matches historical support evidence."
            })
        
        # Default reply generation mock
        return json.dumps({
            "reply": "Thank you for reaching out to Amazon Support! Please check your order details in your account or send us a Direct Message with your order number so we can assist you right away.",
            "evidence_ids": ["conv_hist_1"],
            "confidence": 0.85,
            "needs_escalation": False,
            "escalation_reason": None
        })
