from __future__ import annotations

from dataclasses import dataclass

from llm_service import generate_financial_story


@dataclass
class NarrativeAgent:
    """Produces story, risks, and suggestions using the LLM layer."""

    def run(self, payload: dict) -> dict:
        story = generate_financial_story(payload)
        return {
            "status": "ok",
            "message": "Narrative generated.",
            "story": story,
        }
