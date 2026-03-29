from __future__ import annotations

from dataclasses import dataclass

from .analysis_agent import FinancialAnalysisAgent
from .categorization_agent import CategorizationAgent
from .input_agent import InputAgent
from .narrative_agent import NarrativeAgent
from .planning_agent import PlanningAgent


@dataclass
class AgentPipeline:
    """Coordinates agent handoff via JSON-like dict payloads."""

    input_agent: InputAgent
    categorization_agent: CategorizationAgent
    analysis_agent: FinancialAnalysisAgent
    planning_agent: PlanningAgent
    narrative_agent: NarrativeAgent

    @classmethod
    def build_default(cls, categorization_llm_enabled: bool = False) -> "AgentPipeline":
        return cls(
            input_agent=InputAgent(),
            categorization_agent=CategorizationAgent(llm_enabled=categorization_llm_enabled),
            analysis_agent=FinancialAnalysisAgent(),
            planning_agent=PlanningAgent(),
            narrative_agent=NarrativeAgent(),
        )

    def parse_and_categorize(self, filename: str, file_bytes: bytes) -> dict:
        parsed = self.input_agent.run(filename, file_bytes)
        if parsed.get("status") != "ok":
            return parsed
        return self.categorization_agent.run(parsed)

    def analyze(self, transactions_payload: dict) -> dict:
        analysis = self.analysis_agent.run(transactions_payload)
        if analysis.get("status") != "ok":
            return analysis

        planning_input = {"metrics": analysis.get("metrics", {})}
        planning = self.planning_agent.run(planning_input)

        return {
            "status": "ok",
            "message": "Analysis and planning completed.",
            "metrics": analysis.get("metrics", {}),
            "health_score": analysis.get("health_score", {}),
            "risk_alerts": analysis.get("risk_alerts", []),
            "annual_savings_projection": analysis.get("annual_savings_projection", 0.0),
            "planner": planning.get("planner", {}),
        }

    def narrative(self, analysis_payload: dict, sample_transactions: list[dict]) -> dict:
        llm_payload = {
            "metrics": analysis_payload.get("metrics", {}),
            "health_score": analysis_payload.get("health_score", {}),
            "planner": analysis_payload.get("planner", {}),
            "risk_alerts": analysis_payload.get("risk_alerts", []),
            "annual_savings_projection": analysis_payload.get("annual_savings_projection", 0.0),
            "sample_transactions": sample_transactions[:20],
            "safety_note": "No investment advice. Keep suggestions educational and simple.",
        }
        return self.narrative_agent.run(llm_payload)
