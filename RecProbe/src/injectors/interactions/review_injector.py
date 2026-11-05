"""
Row C of Table 1 - Review-level defects (review-aware methods).

  point       Absent or uninformative review, Off-topic content
  contextual  Context swap, Sentiment flip
  collective  Burst (e.g. review bombing), Shilling (fake reviewer group)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.config import deep_merge
from injectors.base import CONTEXTUAL, MODIFY, POINT, Perturbation
from injectors.collective import BurstPerturbation, ShillingPerturbation, prompt_request
from injectors.engine import RowNoiseInjector
from injectors.registry import register
from utils import ITEM, RATING, REVIEW

LLM_DEFAULTS = {
    "backend": None,                     # vllm | transformers | template (None: first installed)
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "huggingface_token": None,
    "temperature": 0.9,
    "top_p": 0.95,
    "max_new_tokens": None,
    "prompt": None,
}


def _has_text(s: pd.Series) -> pd.Series:
    return s.notna() & (s.astype("string").str.strip().fillna("") != "")


class _ReviewPerturbation(Perturbation):
    signal = "review"
    requires = (REVIEW,)

    @staticmethod
    def needs_text(params) -> bool:
        return True

    def prompt(self, default: str) -> str:
        return (self.p.get("llm_configuration") or {}).get("prompt") or default


@register
class UninformativeReview(_ReviewPerturbation):
    """Missing, very short, or content-free text (e.g. "ok", ".")."""

    name = "uninformative_review"
    observability = POINT
    operation = MODIFY
    default_target = "user"
    description = "Replace reviews with missing, truncated or content-free text."
    DEFAULTS = {
        "selection_strategy": "uniform",
        "min_per_node": 1,
        "max_per_node": 5,
        "modes": ["missing", "short", "content_free"],
        "mode_weights": None,
        "missing_value": "",             # "" or null
        "short_length": 2,               # short: keep only the first tokens
        "content_free_tokens": [".", "ok", "...", "-", "!!", "fine", "n/a"],
    }

    @staticmethod
    def needs_text(params) -> bool:
        return False

    def perturb(self, df):
        idx = self.select_records(df, _has_text(df[REVIEW]))
        if not idx:
            return df
        modes = list(self.p["modes"])
        w = self.p.get("mode_weights")
        w = np.asarray(w, float) / np.sum(w) if w else None
        chosen = self.rng.choice(modes, size=len(idx), p=w)
        k = int(self.p["short_length"])
        new = []
        for i, m in zip(idx, chosen):
            if m == "missing":
                mv = self.p.get("missing_value")
                new.append(pd.NA if mv is None else mv)
            elif m == "short":
                new.append(" ".join(str(df.at[i, REVIEW]).split()[:k]))
            elif m == "content_free":
                new.append(str(self.rng.choice(self.p["content_free_tokens"])))
            else:
                raise ValueError(f"[{self.name}] unknown mode '{m}'")
        self.log_modify(df, idx, REVIEW, new)
        df.loc[idx, REVIEW] = pd.array(new, dtype="string")
        self.stats["modes"] = {m: int((chosen == m).sum()) for m in modes}
        return df


@register
class OffTopicReview(_ReviewPerturbation):
    """Text that discusses aspects external to the product itself, recognizable without
    reference to the item (e.g. shipping delays or seller behaviour)."""

    name = "off_topic_review"
    observability = POINT
    operation = MODIFY
    default_target = "user"
    description = "Replace (or extend) reviews with text about shipping, delivery or the seller."
    DEFAULTS = {
        "selection_strategy": "uniform",
        "min_per_node": 1,
        "max_per_node": 5,
        "mode": "replace",               # replace | append
        "min_length_of_review": 5,
        "max_length_of_review": 60,
        "llm_configuration": deep_merge(LLM_DEFAULTS, {
            "prompt": "Write a short customer review that talks only about the shipping, the "
                      "delivery or the seller's behaviour, without saying anything about the "
                      "product itself."}),
    }

    def perturb(self, df):
        idx = self.select_records(df, _has_text(df[REVIEW]))
        if not idx:
            return df
        tpl = self.prompt(self.DEFAULTS["llm_configuration"]["prompt"])
        reqs = [prompt_request(tpl, self.ctx, df.at[i, ITEM],
                               df.at[i, RATING] if self.ctx.has_rating else None,
                               str(df.at[i, REVIEW]), kind="off_topic") for i in idx]
        texts = self.generate_texts(reqs)
        if self.p["mode"] == "append":
            texts = [f"{df.at[i, REVIEW]} {t}" for i, t in zip(idx, texts)]
        self.log_modify(df, idx, REVIEW, texts)
        df.loc[idx, REVIEW] = pd.array(texts, dtype="string")
        return df


@register
class ContextSwap(_ReviewPerturbation):
    """A well-formed review that is semantically mismatched with the reviewed item
    (e.g. hiking boots described as ideal for the beach)."""

    name = "context_swap"
    observability = CONTEXTUAL
    operation = MODIFY
    default_target = "item"
    description = "Replace reviews with well-formed reviews written for a different item/context."
    DEFAULTS = {
        "selection_strategy": "uniform",
        "min_per_node": 1,
        "max_per_node": 5,
        # donor -> a real review written for another item (another category when item
        #          metadata is available), with the same rating polarity, so that only
        #          the item/context is mismatched;
        # llm   -> a generated review describing the item for an unrelated use.
        "strategy": "donor",
        "min_length_of_review": 5,
        "max_length_of_review": 80,
        "llm_configuration": deep_merge(LLM_DEFAULTS, {
            "prompt": "Write a short {polarity} customer review of '{item}' that describes it "
                      "as ideal for a use or context completely unrelated to what a {category} "
                      "is meant for."}),
    }

    @staticmethod
    def needs_text(params) -> bool:
        return params.get("strategy", "donor") == "llm"

    def perturb(self, df):
        idx = self.select_records(df, _has_text(df[REVIEW]))
        if not idx:
            return df
        if self.p["strategy"] == "llm":
            tpl = self.prompt(self.DEFAULTS["llm_configuration"]["prompt"])
            reqs = [prompt_request(tpl, self.ctx, df.at[i, ITEM],
                                   df.at[i, RATING] if self.ctx.has_rating else None,
                                   kind="context_swap") for i in idx]
            new = self.generate_texts(reqs)
        elif self.p["strategy"] == "donor":
            new = self._donors(df, idx)
        else:
            raise ValueError(f"[{self.name}] unknown strategy '{self.p['strategy']}'")
        keep = [k for k, v in enumerate(new) if v is not None]
        idx, new = [idx[k] for k in keep], [new[k] for k in keep]
        self.stats["target_records"] = len(idx)
        if idx:
            self.log_modify(df, idx, REVIEW, new)
            df.loc[idx, REVIEW] = pd.array(new, dtype="string")
        return df

    def _donors(self, df, idx):
        ctx, rng = self.ctx, self.rng
        pool = ctx.reference[_has_text(ctx.reference[REVIEW])]
        p_items = pool[ITEM].to_numpy()
        p_text = pool[REVIEW].astype(str).to_numpy()
        p_cat = np.array([(ctx.category or {}).get(i) for i in p_items], dtype=object)
        p_pol = np.array([ctx.polarity(r) for r in pool[RATING]]) if ctx.has_rating else None
        out = []
        for i in idx:
            item = df.at[i, ITEM]
            mask = p_items != item
            if ctx.category is not None and ctx.category.get(item) is not None:
                other_cat = mask & (p_cat != ctx.category.get(item))
                mask = other_cat if other_cat.any() else mask
            if p_pol is not None and pd.notna(df.at[i, RATING]):
                same = mask & (p_pol == ctx.polarity(float(df.at[i, RATING])))
                mask = same if same.any() else mask
            cand = np.flatnonzero(mask)
            out.append(str(p_text[rng.choice(cand)]) if len(cand) else None)
        return out


@register
class SentimentFlip(_ReviewPerturbation):
    """A review whose sentiment is inverted relative to the paired rating
    (e.g. a strongly negative text on a 5-star rating). The rating is unchanged."""

    name = "sentiment_flip"
    observability = CONTEXTUAL
    operation = MODIFY
    default_target = "user"
    requires = (REVIEW, RATING)
    description = "Invert the sentiment of the review text while keeping the rating."
    DEFAULTS = {
        "selection_strategy": "uniform",
        "min_per_node": 1,
        "max_per_node": 5,
        # llm   -> the review is rewritten with the opposite sentiment;
        # donor -> replaced by a real review of opposite polarity, preferably of the same item
        "strategy": "llm",
        "polarity_thresholds": {"low": None, "high": None},   # default: lower/upper quarter of the scale
        "min_length_of_review": 3,
        "max_length_of_review": 100,
        "llm_configuration": deep_merge(LLM_DEFAULTS, {
            "prompt": "Rewrite the following product review so that it expresses a clearly "
                      "{polarity} opinion, keeping the same product and topics. Return only the "
                      "rewritten review.\n\nReview: {review}"}),
    }

    @staticmethod
    def needs_text(params) -> bool:
        return params.get("strategy", "llm") == "llm"

    def _pol(self, r):
        th = self.p.get("polarity_thresholds") or {}
        return self.ctx.polarity(float(r), th.get("low"), th.get("high"))

    def perturb(self, df):
        pol = df[RATING].map(lambda r: self._pol(r) if pd.notna(r) else "neutral")
        idx = self.select_records(df, _has_text(df[REVIEW]) & (pol != "neutral"))
        if not idx:
            return df
        target_pol = ["negative" if pol.at[i] == "positive" else "positive" for i in idx]
        if self.p["strategy"] == "llm":
            tpl = self.prompt(self.DEFAULTS["llm_configuration"]["prompt"])
            reqs = [prompt_request(tpl, self.ctx, df.at[i, ITEM], review=str(df.at[i, REVIEW]),
                                   polarity=tp, kind="sentiment_flip") for i, tp in zip(idx, target_pol)]
            new = self.generate_texts(reqs)
        elif self.p["strategy"] == "donor":
            new = self._donors(df, idx, target_pol)
        else:
            raise ValueError(f"[{self.name}] unknown strategy '{self.p['strategy']}'")
        keep = [k for k, v in enumerate(new) if v is not None]
        idx, new = [idx[k] for k in keep], [new[k] for k in keep]
        self.stats["target_records"] = len(idx)
        if idx:
            self.log_modify(df, idx, REVIEW, new)
            df.loc[idx, REVIEW] = pd.array(new, dtype="string")
        return df

    def _donors(self, df, idx, target_pol):
        ctx, rng = self.ctx, self.rng
        pool = ctx.reference[_has_text(ctx.reference[REVIEW]) & ctx.reference[RATING].notna()]
        p_items = pool[ITEM].to_numpy()
        p_text = pool[REVIEW].astype(str).to_numpy()
        p_pol = np.array([self._pol(r) for r in pool[RATING]])
        p_cat = np.array([(ctx.category or {}).get(i) for i in p_items], dtype=object)
        out = []
        for i, tp in zip(idx, target_pol):
            item = df.at[i, ITEM]
            base = p_pol == tp
            for mask in (base & (p_items == item),
                         base & (p_cat == (ctx.category or {}).get(item)) if ctx.category else base & False,
                         base):
                cand = np.flatnonzero(mask)
                if len(cand):
                    out.append(str(p_text[rng.choice(cand)]))
                    break
            else:
                out.append(None)
        return out


@register
class ReviewBurst(_ReviewPerturbation, BurstPerturbation):
    """A concentration of reviews on the same target items within a short time window,
    with the text as the vehicle of the manipulation (e.g. a review-bombing campaign).
    The reviews are generated with an LLM (Figure 3, central panel)."""

    name = "review_burst"
    observability = BurstPerturbation.observability
    operation = BurstPerturbation.operation
    default_target = "item"
    generate_reviews = True
    description = "Burst of LLM-generated reviews on target items inside a time window."
    DEFAULTS = deep_merge(BurstPerturbation.BURST_DEFAULTS, {
        "min_length_of_review": 10,
        "max_length_of_review": 100,
        "llm_configuration": deep_merge(LLM_DEFAULTS, {
            "prompt": "Write a very short, generic {polarity} review of {item}."}),
    })

    def burst_prompt(self, item, rating):
        tpl = self.prompt(self.DEFAULTS["llm_configuration"]["prompt"])
        return prompt_request(tpl, self.ctx, item, rating, kind="burst")


@register
class ReviewShilling(_ReviewPerturbation, ShillingPerturbation):
    """A group of coordinated reviewer accounts targeting the same items, with filler
    reviews that make each account plausible in isolation (fake reviewer group).
    Target reviews are generated with the same prompt, so the group shares textual and
    behavioural similarity; filler reviews are real reviews of the filler items."""

    name = "review_shilling"
    observability = ShillingPerturbation.observability
    operation = ShillingPerturbation.operation
    default_target = "item"
    generate_reviews = True
    description = "Fake reviewer group: LLM target reviews (push/nuke) plus plausible filler reviews."
    DEFAULTS = deep_merge(ShillingPerturbation.SHILLING_DEFAULTS, {
        "min_length_of_review": 10,
        "max_length_of_review": 100,
        "llm_configuration": deep_merge(LLM_DEFAULTS, {
            "prompt": "Write a short, enthusiastic {polarity} review of {item}."}),
    })

    def target_prompt(self, item):
        tpl = self.prompt(self.DEFAULTS["llm_configuration"]["prompt"])
        pol = "positive" if self.p.get("direction", "push") == "push" else "negative"
        rating = self.target_rating() if self.ctx.has_rating else None
        return prompt_request(tpl, self.ctx, item, rating, polarity=pol, kind="shilling")


class ReviewNoiseInjector(RowNoiseInjector):
    signal = "review"
