"""
Synthetic review generation for the review-level perturbations (Row C, Table 1).

As described in Section 4.1, the LLM and its generation parameters (model,
prompt, temperature, ...) are given in the ``llm_configuration`` block of the
injection YAML. Backends:

  * vllm         - efficient batched inference with vLLM (default when installed);
  * transformers - Hugging Face text-generation pipeline;
  * template     - NO LLM: sentences sampled from small built-in word banks.
                   Meant only for tests, demos and machines without a GPU. It has
                   to be selected explicitly with ``backend: template``.

All prompts of one perturbation are generated in a single batch.
"""

from __future__ import annotations

import os
import re

import numpy as np


class _SafeDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


def format_prompt(template: str, values: dict) -> str:
    return (template or "").format_map(_SafeDict(values))


def _clean(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^(here is|here's)[^:\n]*:\s*", "", text, flags=re.I)
    text = re.sub(r"^(review|rewritten review)\s*:\s*", "", text, flags=re.I)
    return text.strip().strip('"').strip("“”").strip()


def _truncate(text: str, max_tokens: int) -> str:
    tokens = text.split()
    return text if len(tokens) <= max_tokens else " ".join(tokens[:max_tokens])


class TextGenerator:
    def __init__(self, cfg: dict, seed: int, logger):
        self.cfg = dict(cfg or {})
        self.seed = int(seed)
        self.logger = logger
        self.backend = self.cfg.get("backend") or self._default_backend()
        self.temperature = float(self.cfg.get("temperature", 0.9))
        self.top_p = float(self.cfg.get("top_p", 0.95))
        self.max_retries = int(self.cfg.get("max_retries", 2))
        self.system_prompt = self.cfg.get("system_prompt") or (
            "You write realistic customer reviews. Answer with the review text only.")
        self._engine = None
        token = self.cfg.get("huggingface_token")
        if token and set(str(token)) != {"*"}:
            os.environ.setdefault("HF_TOKEN", str(token))
        self.rng = np.random.default_rng(self.seed)

    @staticmethod
    def _default_backend() -> str:
        for name, mod in (("vllm", "vllm"), ("transformers", "transformers")):
            try:
                __import__(mod)
                return name
            except ImportError:
                continue
        raise ImportError("No LLM backend available: install vllm or transformers, "
                          "or set llm_configuration.backend: template for an offline run.")

    # ------------------------------------------------------------ public
    def generate(self, requests: list[dict], min_len: int = 1, max_len: int = 100) -> list[str]:
        self.logger.info(f"Generating {len(requests)} reviews with backend '{self.backend}'"
                         + (f" ({self.cfg.get('model')})" if self.backend != "template" else ""))
        outputs = [None] * len(requests)
        todo = list(range(len(requests)))
        for attempt in range(self.max_retries + 1):
            texts = self._generate([requests[i] for i in todo], max_len)
            still = []
            for i, t in zip(todo, texts):
                t = _truncate(_clean(t), max_len)
                outputs[i] = t
                if len(t.split()) < min_len and attempt < self.max_retries:
                    still.append(i)
            if not still:
                break
            todo = still
        return outputs

    # ---------------------------------------------------------- backends
    def _messages(self, prompt: str) -> list[dict]:
        return [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": prompt}]

    def _generate(self, requests: list[dict], max_len: int) -> list[str]:
        if self.backend == "template":
            return [self._template(r, max_len) for r in requests]
        prompts = [r["prompt"] for r in requests]
        max_new = int(self.cfg.get("max_new_tokens") or max_len * 2 + 16)
        if self.backend == "vllm":
            return self._vllm(prompts, max_new)
        if self.backend == "transformers":
            return self._transformers(prompts, max_new)
        raise ValueError(f"Unknown llm backend '{self.backend}' (vllm | transformers | template)")

    def _vllm(self, prompts, max_new):
        from vllm import LLM, SamplingParams

        if self._engine is None:
            self._engine = LLM(model=self.cfg["model"], seed=self.seed, **(self.cfg.get("engine_args") or {}))
        params = SamplingParams(temperature=self.temperature, top_p=self.top_p, max_tokens=max_new)
        outs = self._engine.chat([self._messages(p) for p in prompts], params, use_tqdm=False)
        return [o.outputs[0].text for o in outs]

    def _transformers(self, prompts, max_new):
        import transformers
        from transformers import pipeline

        if self._engine is None:
            transformers.set_seed(self.seed)
            self._engine = pipeline("text-generation", model=self.cfg["model"],
                                    device_map=self.cfg.get("device_map", "auto"), torch_dtype="auto")
            if self._engine.tokenizer.pad_token_id is None:
                self._engine.tokenizer.pad_token_id = self._engine.tokenizer.eos_token_id
        outs = self._engine([self._messages(p) for p in prompts], max_new_tokens=max_new,
                            do_sample=self.temperature > 0, temperature=max(self.temperature, 1e-5),
                            top_p=self.top_p, batch_size=int(self.cfg.get("batch_size", 8)))
        return [o[0]["generated_text"][-1]["content"] for o in outs]

    # ------------------------------------------------------ offline bank
    _BANK = {
        "positive": ["I love it.", "Works exactly as described.", "Great quality for the price.",
                     "Would definitely buy again.", "Exceeded my expectations.", "Highly recommended.",
                     "Really happy with this purchase.", "Five stars, no complaints."],
        "negative": ["Terrible item.", "Broke after a few days.", "Complete waste of money.",
                     "Not as described at all.", "Very disappointed.", "Would not recommend it.",
                     "Awful quality.", "I returned it immediately."],
        "neutral": ["It is okay.", "Does the job.", "Average product.", "Nothing special.",
                    "Fine for the price.", "Some pros and some cons."],
        "off_topic": ["The package arrived two weeks late.", "The seller never answered my messages.",
                      "Shipping was very slow.", "The box was damaged by the courier.",
                      "Customer service was rude on the phone.", "Tracking information was never updated.",
                      "The seller sent the wrong invoice.", "Delivery was left at the wrong door."],
        "context_swap": ["Perfect to take to the beach in summer.", "Ideal for hiking in the snow.",
                         "Great as a gift for a newborn.", "My dog loves playing with it.",
                         "Excellent for cooking pasta.", "Works well for fixing the car engine."],
    }

    def _template(self, request: dict, max_len: int) -> str:
        kind = request.get("kind", "review")
        pol = request.get("polarity") or "neutral"
        if kind == "off_topic":
            bank = self._BANK["off_topic"]
        elif kind == "context_swap":
            bank = self._BANK["context_swap"]
        else:
            bank = self._BANK.get(pol, self._BANK["neutral"])
        n = int(self.rng.integers(1, 4))
        text = " ".join(self.rng.choice(bank, size=n, replace=False))
        if kind == "context_swap" and pol in ("positive", "negative"):
            text = f"{self.rng.choice(self._BANK[pol])} {text}"
        return _truncate(text, max_len)
