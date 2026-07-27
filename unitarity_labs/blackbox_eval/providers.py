"""Providers — one uniform call surface over any model backend.

A `Provider.complete(prompt, system)` returns a `Completion` (the response text
plus light metadata). Everything downstream (runner, scoring) is provider-
agnostic, so the same suite runs against a mock, a hosted API model, or a local
checkpoint by swapping the provider.

SDK imports (`anthropic`, `openai`, `transformers`) happen lazily inside each
provider's constructor, so `import unitarity_labs.blackbox_eval` needs none of
them. Only the provider you actually instantiate pulls its dependency.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable


@dataclass
class Completion:
    """One model response plus metadata the runner records."""

    text: str
    model: str
    provider: str
    # Optional token counts if the backend reports them; None when unknown.
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    # True if the backend signalled a safety/policy refusal at the protocol
    # level (distinct from a refusal detected heuristically in the text).
    refused: bool = False
    raw: dict = field(default_factory=dict)


@runtime_checkable
class Provider(Protocol):
    """Anything that turns a prompt into a `Completion`."""

    name: str
    model: str

    def complete(self, prompt: str, system: Optional[str] = None) -> Completion:
        ...


# ---------------------------------------------------------------------------
# Mock — deterministic, offline. Used by the test suite and for dry runs.
# ---------------------------------------------------------------------------
class MockProvider:
    """Deterministic provider with no network dependency.

    `rules` maps a lowercased substring to a canned reply; the first matching
    rule wins. With no rule match it echoes the prompt, which is enough to
    exercise contains/length/regex checks. `refuse_on` substrings force a
    refusal-shaped reply so refusal checks are testable offline.
    """

    def __init__(
        self,
        model: str = "mock-1",
        rules: Optional[dict[str, str]] = None,
        refuse_on: Optional[list[str]] = None,
    ) -> None:
        self.name = "mock"
        self.model = model
        self._rules = {k.lower(): v for k, v in (rules or {}).items()}
        self._refuse_on = [s.lower() for s in (refuse_on or [])]

    def complete(self, prompt: str, system: Optional[str] = None) -> Completion:
        low = prompt.lower()
        for needle in self._refuse_on:
            if needle in low:
                return Completion(
                    text="I can't help with that request.",
                    model=self.model,
                    provider=self.name,
                    refused=True,
                )
        for needle, reply in self._rules.items():
            if needle in low:
                return Completion(text=reply, model=self.model, provider=self.name)
        return Completion(text=f"echo: {prompt}", model=self.model, provider=self.name)


# ---------------------------------------------------------------------------
# Anthropic — hosted API, works on large models with no internals exposed.
# ---------------------------------------------------------------------------
class AnthropicProvider:
    """Calls the Anthropic Messages API via the official `anthropic` SDK.

    Requires `pip install anthropic` and a credential the SDK can resolve
    (`ANTHROPIC_API_KEY`, or an `ant auth login` profile). Model defaults to
    `claude-opus-5`; override for whichever model you want to test.
    """

    def __init__(self, model: str = "claude-opus-5", max_tokens: int = 1024) -> None:
        try:
            import anthropic  # noqa: F401
        except ImportError as exc:  # pragma: no cover - exercised only without the dep
            raise ImportError(
                "AnthropicProvider needs the anthropic SDK: pip install anthropic"
            ) from exc
        self.name = "anthropic"
        self.model = model
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic()

    def complete(self, prompt: str, system: Optional[str] = None) -> Completion:
        kwargs: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        resp = self._client.messages.create(**kwargs)
        # A safety decline arrives as a normal 200 with stop_reason == "refusal".
        refused = getattr(resp, "stop_reason", None) == "refusal"
        text = "".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        )
        usage = getattr(resp, "usage", None)
        return Completion(
            text=text,
            model=getattr(resp, "model", self.model),
            provider=self.name,
            input_tokens=getattr(usage, "input_tokens", None) if usage else None,
            output_tokens=getattr(usage, "output_tokens", None) if usage else None,
            refused=refused,
        )


# ---------------------------------------------------------------------------
# OpenAI — the "bigger chatGPT models" path.
# ---------------------------------------------------------------------------
class OpenAIProvider:
    """Calls OpenAI chat completions via the official `openai` SDK.

    Requires `pip install openai` and `OPENAI_API_KEY`. Model is required —
    pass e.g. "gpt-4o", "gpt-4o-mini", "o3-mini", or whatever you have access to.
    """

    def __init__(self, model: str, max_tokens: int = 1024) -> None:
        try:
            import openai  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "OpenAIProvider needs the openai SDK: pip install openai"
            ) from exc
        self.name = "openai"
        self.model = model
        self.max_tokens = max_tokens
        self._client = openai.OpenAI()

    def complete(self, prompt: str, system: Optional[str] = None) -> Completion:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = self._client.chat.completions.create(
            model=self.model, max_tokens=self.max_tokens, messages=messages
        )
        choice = resp.choices[0]
        usage = getattr(resp, "usage", None)
        return Completion(
            text=choice.message.content or "",
            model=getattr(resp, "model", self.model),
            provider=self.name,
            input_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
            output_tokens=getattr(usage, "completion_tokens", None) if usage else None,
            refused=choice.finish_reason == "content_filter",
        )


# ---------------------------------------------------------------------------
# Local — small open-weight models via transformers (already a core dep).
# ---------------------------------------------------------------------------
class LocalTransformersProvider:
    """Runs a small open-weight causal LM locally via transformers.

    Useful for a fully offline, zero-cost smoke run of the harness itself
    (e.g. distilgpt2). Not the way to test *big* models — that's what the
    hosted-API providers are for.
    """

    def __init__(self, model: str = "distilgpt2", max_new_tokens: int = 128) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "LocalTransformersProvider needs transformers (a core dependency)"
            ) from exc
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = "local"
        self.model = model
        self.max_new_tokens = max_new_tokens
        self._tok = AutoTokenizer.from_pretrained(model)
        self._model = AutoModelForCausalLM.from_pretrained(model)

    def complete(self, prompt: str, system: Optional[str] = None) -> Completion:
        text_in = f"{system}\n\n{prompt}" if system else prompt
        inputs = self._tok(text_in, return_tensors="pt")
        out = self._model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self._tok.eos_token_id,
        )
        generated = out[0][inputs["input_ids"].shape[1]:]
        text = self._tok.decode(generated, skip_special_tokens=True)
        return Completion(text=text, model=self.model, provider=self.name)


def get_provider(name: str, model: Optional[str] = None, **kwargs) -> Provider:
    """Factory used by the CLI. `name` in {mock, anthropic, openai, local}."""
    name = name.lower()
    if name == "mock":
        return MockProvider(model=model or "mock-1", **kwargs)
    if name == "anthropic":
        return AnthropicProvider(model=model or "claude-opus-5", **kwargs)
    if name == "openai":
        if not model:
            raise ValueError("openai provider requires --model (e.g. gpt-4o-mini)")
        return OpenAIProvider(model=model, **kwargs)
    if name == "local":
        return LocalTransformersProvider(model=model or "distilgpt2", **kwargs)
    raise ValueError(f"unknown provider: {name!r} (want mock|anthropic|openai|local)")
