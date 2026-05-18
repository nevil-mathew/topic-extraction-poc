"""
LLM-based Topic Labeling
=========================

Generate human-readable topic labels using LLMs:
- Claude (Anthropic)
- GPT-4 (OpenAI)
- Gemini (Google)

Two output styles:
- "short"  : 3-7 word title + 1-2 sentence description (default; backward compatible)
- "theme"  : 5-8 word evocative title + 4-6 sentence narrative paragraph suitable
             for inclusion in a qualitative research report
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Literal


class LLMLabeler:
    """
    Generate topic labels using Large Language Models.

    Parameters
    ----------
    provider : str
        LLM provider: "anthropic", "openai", or "google"
    api_key : str
        API key for the provider.
    model : str, optional
        Model name. Defaults to best available model.
    max_tokens : int, optional
        Maximum tokens in response. When ``style="theme"`` and this is left at
        the default, a higher per-call cap is used automatically.
    temperature : float
        Sampling temperature. Default: 0.3
    language : str
        Output language. Default: "english"
    domain_hint : str, optional
        Short noun phrase describing the study (e.g. "education barriers in
        rural India"). Injected into every prompt to anchor register.
    style : {"short", "theme"}
        Output style. "short" produces a brief label + 1-2 sentence
        description (legacy behavior). "theme" produces a report-ready
        evocative title + 4-6 sentence narrative paragraph that quotes
        the underlying documents.
    n_docs : int, optional
        Number of representative documents included. Defaults depend on
        ``style`` (5 for short, 8 for theme).
    doc_max_chars : int, optional
        Maximum characters per document. Defaults depend on ``style``
        (500 for short, 1200 for theme).
    n_keywords : int, optional
        Number of keywords passed to the LLM (10 for short, 15 for theme).
    cache : bool
        If True, cache responses by prompt hash to avoid re-paying on retries.
    verbose : bool
        Print each label as it is generated. Default: False
    """

    def __init__(
        self,
        provider: Literal["anthropic", "openai", "google"] = "anthropic",
        api_key: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.3,
        language: str = "english",
        domain_hint: str | None = None,
        style: Literal["short", "theme"] = "short",
        n_docs: int | None = None,
        doc_max_chars: int | None = None,
        n_keywords: int | None = None,
        cache: bool = True,
        verbose: bool = False,
    ):
        self.provider = provider
        self.api_key = api_key
        self.model = model or self._default_model()
        self.temperature = temperature
        self.language = language
        self.domain_hint = domain_hint
        self.style = style
        self.verbose = verbose

        # Style-dependent defaults
        if style == "theme":
            self.max_tokens = max_tokens if max_tokens is not None else 900
            self.n_docs = n_docs if n_docs is not None else 8
            self.doc_max_chars = doc_max_chars if doc_max_chars is not None else 1200
            self.n_keywords = n_keywords if n_keywords is not None else 15
        else:
            self.max_tokens = max_tokens if max_tokens is not None else 500
            self.n_docs = n_docs if n_docs is not None else 5
            self.doc_max_chars = doc_max_chars if doc_max_chars is not None else 500
            self.n_keywords = n_keywords if n_keywords is not None else 10

        self._client = None
        self._cache: dict[str, tuple[str, str]] = {} if cache else None
        # When True, the Google path skips the label/description response_schema
        # so callers can request arbitrary JSON shapes (e.g. meta-theme proposer).
        self._raw_mode: bool = False
    
    def _default_model(self) -> str:
        """Get default model for provider."""
        if self.provider == "anthropic":
            return "claude-haiku-4-5-20251001"
        elif self.provider == "google":
            return "gemini-2.5-flash"
        else:
            return "gpt-4o-mini"
    
    def _init_client(self):
        """Initialize API client."""
        if self._client is not None:
            return
        
        if self.provider == "anthropic":
            try:
                from anthropic import Anthropic
                self._client = Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "anthropic package not installed. "
                    "Install with: pip install anthropic"
                )
        elif self.provider == "google":
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "google-genai package not installed. "
                    "Install with: pip install google-genai"
                )
        else:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "openai package not installed. "
                    "Install with: pip install openai"
                )
    
    def _call_with_retry(self, fn):
        """Call fn() up to 3 times with exponential backoff on transient errors."""
        for attempt in range(3):
            try:
                return fn()
            except Exception as e:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)

    def generate_label(
        self,
        keywords: list[str],
        representative_docs: list[str],
        domain_hint: str | None = None,
        existing_labels: list[dict] | None = None,
    ) -> tuple[str, str]:
        """
        Generate a label for a topic.

        Parameters
        ----------
        keywords : list[str]
            Topic keywords (top 10-15 recommended).
        representative_docs : list[str]
            Representative documents for the topic.
        domain_hint : str, optional
            Domain context (e.g., "education barriers in rural India").
            Overrides ``self.domain_hint`` if provided.
        existing_labels : list[dict], optional
            Labels already assigned to other topics in this run, each as
            ``{"label": str, "keywords": list[str]}``. Used to prevent
            duplicate or near-duplicate labels across topics.

        Returns
        -------
        label : str
        description : str
        """
        self._init_client()

        system_prompt, user_prompt = self._build_prompt(
            keywords, representative_docs, domain_hint, existing_labels
        )

        cache_key = self._cache_key(system_prompt, user_prompt) if self._cache is not None else None
        if cache_key is not None and cache_key in self._cache:
            return self._cache[cache_key]

        try:
            if self.provider == "anthropic":
                response = self._call_with_retry(lambda: self._call_anthropic(system_prompt, user_prompt))
            elif self.provider == "google":
                response = self._call_with_retry(lambda: self._call_google(system_prompt, user_prompt))
            else:
                response = self._call_with_retry(lambda: self._call_openai(system_prompt, user_prompt))
        except Exception as e:
            import warnings
            warnings.warn(f"LLM API call failed: {e}. Falling back to keyword label.")
            fallback_label = " & ".join(kw.title() for kw in keywords[:3])
            return fallback_label, f"Topic related to: {', '.join(keywords[:6])}"

        result = self._parse_response(response)
        if cache_key is not None:
            self._cache[cache_key] = result
        return result

    def _cache_key(self, system_prompt: str, user_prompt: str) -> str:
        h = hashlib.sha256()
        h.update(self.model.encode())
        h.update(b"\x00")
        h.update(system_prompt.encode())
        h.update(b"\x00")
        h.update(user_prompt.encode())
        return h.hexdigest()

    def call_raw(self, system_prompt: str, user_prompt: str, max_tokens: int | None = None) -> str:
        """Low-level LLM call. Used by meta-theme generation paths in TriTopic.

        Returns the raw text response. Caller is responsible for parsing.
        Honors caching and retry behavior. Falls back to raising on failure
        so callers can decide whether to recover.
        """
        self._init_client()

        cache_key = self._cache_key(system_prompt, user_prompt) if self._cache is not None else None
        if cache_key is not None and cache_key in self._cache:
            # Cache stores tuples; raw API stores str under a separate prefix
            cached = self._cache.get("raw:" + cache_key)
            if cached:
                return cached  # type: ignore[return-value]

        # Temporarily swap max_tokens if caller wants more headroom,
        # and disable the Google structured-output schema so the response
        # can take an arbitrary JSON shape (e.g. {"themes": [...]}).
        saved_tokens = self.max_tokens
        saved_raw = self._raw_mode
        try:
            if max_tokens is not None:
                self.max_tokens = max_tokens
            self._raw_mode = True
            if self.provider == "anthropic":
                response = self._call_with_retry(lambda: self._call_anthropic(system_prompt, user_prompt))
            elif self.provider == "google":
                response = self._call_with_retry(lambda: self._call_google(system_prompt, user_prompt))
            else:
                response = self._call_with_retry(lambda: self._call_openai(system_prompt, user_prompt))
        finally:
            self.max_tokens = saved_tokens
            self._raw_mode = saved_raw

        if cache_key is not None:
            self._cache["raw:" + cache_key] = response  # type: ignore[assignment]
        return response
    
    def _build_prompt(
        self,
        keywords: list[str],
        representative_docs: list[str],
        domain_hint: str | None = None,
        existing_labels: list[dict] | None = None,
    ) -> tuple[str, str]:
        """Build the labeling prompt for the configured style."""
        hint = domain_hint or self.domain_hint

        # Format documents block (shared)
        docs_text = ""
        for i, doc in enumerate(representative_docs[:self.n_docs], 1):
            truncated = doc[:self.doc_max_chars] + "..." if len(doc) > self.doc_max_chars else doc
            docs_text += f"\nDocument {i}: {truncated}\n"

        # Format "already used labels" block to prevent duplication
        existing_block = ""
        if existing_labels:
            lines = []
            for ex in existing_labels[-40:]:  # cap context to most recent 40
                kw_str = ", ".join(ex.get("keywords", [])[:5])
                lines.append(f"  - \"{ex['label']}\"  (keywords: {kw_str})")
            existing_block = (
                "\nTHEMES ALREADY ASSIGNED TO OTHER TOPICS IN THIS REPORT (do not duplicate "
                "or produce near-duplicates such as the same first three words):\n"
                + "\n".join(lines)
                + "\n\nIf your topic overlaps thematically with any above, name what makes "
                "YOUR cluster specifically different (the distinguishing mechanism or keyword), "
                "not what is shared.\n"
            )

        if self.style == "theme":
            return self._build_theme_prompt(keywords, docs_text, hint, existing_block)
        return self._build_short_prompt(keywords, docs_text, hint, existing_block)

    def _build_short_prompt(
        self, keywords: list[str], docs_text: str, hint: str | None, existing_block: str
    ) -> tuple[str, str]:
        domain_context = f"\nDomain context: This is about {hint}.\n" if hint else ""

        system_prompt = (
            "You are a qualitative research analyst skilled at surfacing higher-order themes from "
            "clusters of documents. Your task is to identify the deeper human concern, motivation, "
            "perception, or systemic pattern that unifies a cluster — not just its surface subject. "
            "You always respond with valid JSON and nothing else."
        )

        user_prompt = f"""You are analyzing a cluster of documents from a topic model. Your goal is to name the THEME — the deeper meaning — that unifies this cluster, not simply describe its subject matter.

Keywords (statistically representative of this cluster):
{', '.join(keywords[:self.n_keywords])}

Representative Documents:
{docs_text}
{domain_context}{existing_block}
Instructions:
- Identify the underlying human concern, motivation, perception, or systemic pattern that connects these documents.
- Do NOT produce surface-level category names (e.g. "Infrastructure Issues", "Teacher Feedback", "Attendance Problems").
- Do NOT use the bare phrases "Systemic Barriers", "Structural Barriers", "Geographic Barriers", or "Educational Access" — they are too generic for this domain. Name the SPECIFIC mechanism instead.
- The theme label should be 3-7 words, title case, no special characters.
- The description should explain the underlying concern in 1-2 sentences — what people are really expressing, not just what they are talking about.
- Output in {self.language}.

Respond ONLY with this exact JSON format, no other text:
{{"label": "Your Theme Label", "description": "Your insight-oriented description."}}"""

        return system_prompt, user_prompt

    def _build_theme_prompt(
        self, keywords: list[str], docs_text: str, hint: str | None, existing_block: str
    ) -> tuple[str, str]:
        domain_context = (
            f"\nDomain context: This is a qualitative research study about {hint}.\n"
            if hint else ""
        )

        system_prompt = (
            "You are a qualitative research analyst writing emerging themes for a research "
            "report. Your output will be read by program staff, funders, and the public. "
            "Each theme must be EVOCATIVE (a 5-8 word title naming the lived experience), "
            "CONCRETE (citing specific mechanisms, behaviors, or short quoted phrases from "
            "the source documents), DISTINCT (clearly different from other themes in the "
            "report), and NARRATIVE (a 4-6 sentence paragraph that synthesizes what people "
            "are really expressing — not just what they are talking about). "
            "You always respond with valid JSON and nothing else."
        )

        user_prompt = f"""You are naming an emerging theme that unifies a cluster of documents from a qualitative study. Below are the keywords statistically representative of the cluster, and a sample of representative documents.
{domain_context}
Keywords (top {self.n_keywords}):
{', '.join(keywords[:self.n_keywords])}

Representative Documents:
{docs_text}
{existing_block}
YOUR "label" (the theme title) MUST:
- Be 5-8 words, title case, no special characters
- Be EVOCATIVE — name the lived experience, not a generic abstraction
- NEVER use the bare phrases "Systemic Barriers", "Structural Barriers", "Geographic Barriers", "Educational Access", "Systemic Disadvantage" — these are too generic
- Name the SPECIFIC mechanism or actor (good: "Aadhar Card as Gatekeeper to Schooling", "Distance to Anganwadi Centre Disrupts Early Learning"; bad: "Documentation Barriers", "Geographic Barriers")

YOUR "description" (the narrative) MUST:
- Be 4-6 sentences (40-100 words)
- Include at least one short quoted phrase from the documents (e.g., parents described the school as "too far", families said admission was "denied without Aadhar")
- Describe concrete observations: what people said, did, or experienced — using specifics from the documents
- End with a sentence that names the underlying meaning or "why it matters" insight
- Be written in plain {self.language}, third person, present-or-past tense as appropriate

REFERENCE EXAMPLE of the target style (do not copy, only match register):

  Title: "A Festive and Welcoming School Environment"
  Description: "The PTM was often described as a 'celebration' or a 'festival.' Schools were decorated with student artwork, crafts, and photo booths, creating a vibrant atmosphere. Students actively participated in welcoming guests, singing songs, and organizing events — making the PTM feel like a community-led initiative. This joyful environment made parents feel proud and included."

Respond ONLY with this exact JSON format, no other text:
{{"label": "Your Evocative Title Here", "description": "Your 4-6 sentence narrative paragraph in the style above."}}"""

        return system_prompt, user_prompt
    
    def _call_anthropic(self, system_prompt: str, user_prompt: str) -> str:
        """Call Anthropic API."""
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text

    def _call_openai(self, system_prompt: str, user_prompt: str) -> str:
        """Call OpenAI API."""
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content

    def _call_google(self, system_prompt: str, user_prompt: str) -> str:
        """Call Google Gemini API via google-genai with structured JSON output."""
        from google.genai import types

        # gemini-2.5-* are thinking models; disable thinking so output tokens
        # are not crowded out by reasoning tokens, which would truncate the JSON.
        config_kwargs: dict = dict(
            system_instruction=system_prompt,
            temperature=self.temperature,
            max_output_tokens=max(self.max_tokens, 1024),
            response_mime_type="application/json",
        )
        # Only constrain to the label/description schema for standard topic
        # labeling. Raw callers (e.g. meta-theme proposer) need free-form JSON.
        if not self._raw_mode:
            config_kwargs["response_schema"] = types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "label": types.Schema(type=types.Type.STRING),
                    "description": types.Schema(type=types.Type.STRING),
                },
                required=["label", "description"],
            )
        if "2.5" in self.model:
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)

        response = self._client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        return response.text

    def _parse_response(self, response: str) -> tuple[str, str]:
        """Parse LLM response to extract label and description."""
        import re

        if not response or not response.strip():
            return "Unknown Topic", ""

        # Strategy 1: Try to parse complete JSON from response
        try:
            start = response.find("{")
            end = response.rfind("}") + 1

            if start != -1 and end > start:
                json_str = response[start:end]
                data = json.loads(json_str)

                label = data.get("label", "Unknown Topic")
                description = data.get("description", "")

                return str(label).strip(), str(description).strip()
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

        # Strategy 2: Handle truncated JSON by extracting fields with regex
        label_match = re.search(r'"label"\s*:\s*"([^"]*)"', response)
        desc_match = re.search(r'"description"\s*:\s*"([^"]*)', response)

        if label_match:
            label = label_match.group(1).strip()
            description = desc_match.group(1).strip() if desc_match else ""
            return label, description

        # Strategy 3: Fallback - extract from plain text
        lines = [line.strip() for line in response.strip().split("\n") if line.strip()]
        label = lines[0] if lines else "Unknown Topic"
        description = " ".join(lines[1:]) if len(lines) > 1 else ""

        # Clean up common artifacts
        for prefix in ["Label:", "label:", "**Label:**", "1.", "1)"]:
            if label.startswith(prefix):
                label = label[len(prefix):].strip()
        for prefix in ["Description:", "description:", "**Description:**", "2.", "2)"]:
            if description.startswith(prefix):
                description = description[len(prefix):].strip()

        label = label.strip('"\'{}')
        description = description.strip('"\'{}')

        return label or "Unknown Topic", description
    
    def generate_labels_batch(
        self,
        topics_data: list[dict],
        domain_hint: str | None = None,
        dedup: bool = True,
    ) -> list[tuple[str, str]]:
        """
        Generate labels for multiple topics, optionally with cross-topic
        deduplication context.

        Parameters
        ----------
        topics_data : list[dict]
            List of dicts with "keywords" and "representative_docs".
        domain_hint : str, optional
            Domain context.
        dedup : bool
            If True (default), each call sees the previously assigned labels
            and is instructed not to duplicate them.

        Returns
        -------
        list[tuple[str, str]]
        """
        results: list[tuple[str, str]] = []
        existing: list[dict] = []

        for topic in topics_data:
            label, desc = self.generate_label(
                keywords=topic["keywords"],
                representative_docs=topic["representative_docs"],
                domain_hint=domain_hint,
                existing_labels=existing if dedup else None,
            )
            results.append((label, desc))
            if dedup:
                existing.append({"label": label, "keywords": topic["keywords"]})

        return results


class SimpleLabeler:
    """
    Simple rule-based labeler (no LLM required).
    
    Creates labels from top keywords.
    """
    
    def __init__(self, n_words: int = 3):
        self.n_words = n_words
    
    def generate_label(
        self,
        keywords: list[str],
        **kwargs,
    ) -> tuple[str, str]:
        """Generate label from top keywords."""
        # Take top n keywords
        top_keywords = keywords[:self.n_words]
        
        # Title case
        label = " & ".join(kw.title() for kw in top_keywords)
        
        # Description from more keywords
        description = f"Topics related to: {', '.join(keywords[:6])}"
        
        return label, description
