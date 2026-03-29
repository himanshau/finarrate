from __future__ import annotations

import ast
import json
import re
from typing import Any, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph


class StoryState(TypedDict):
    payload: dict[str, Any]
    llm_output: str


DEFAULT_ALERT = "Review recent high-spend categories to avoid month-end cash stress."
DEFAULT_ACTION = "Track weekly spending and reduce one non-essential category by 10%."


def _is_investment_query(query: str) -> bool:
    q = query.lower()
    keywords = [
        "invest",
        "investment",
        "sip",
        "portfolio",
        "return",
        "high return",
        "allocation",
        "mutual fund",
        "equity",
        "debt",
        "diversif",
    ]
    return any(k in q for k in keywords)


def _looks_like_refusal(text: str) -> bool:
    t = text.lower()
    patterns = [
        "i'm sorry",
        "i am sorry",
        "i cannot provide",
        "i can't provide",
        "cannot assist",
        "can't assist",
        "unable to help",
    ]
    return any(p in t for p in patterns)


def _safe_investment_answer(context: dict[str, Any]) -> str:
    metrics = context.get("metrics", {})
    planner = context.get("planner", {})

    income = float(metrics.get("total_income", 0.0) or 0.0)
    expenses = float(metrics.get("total_expenses", 0.0) or 0.0)
    savings = max(income - expenses, 0.0)
    savings_rate = float(metrics.get("savings_rate", 0.0) or 0.0)
    emi_ratio = float(metrics.get("emi_to_income_ratio", 0.0) or 0.0)
    sip_amount = float(planner.get("suggested_monthly_sip_like_amount", 0.0) or 0.0)
    profile = str(planner.get("suitable_sip_style_for_current_portfolio", "Balanced"))

    # Allocation bands are educational and profile-based, not product recommendations.
    equity, debt, gold, liquid = 50, 30, 10, 10
    if profile.lower().startswith("conservative") or savings_rate < 15 or emi_ratio > 25:
        equity, debt, gold, liquid = 35, 40, 10, 15
    elif profile.lower().startswith("growth") and savings_rate >= 25 and emi_ratio <= 20:
        equity, debt, gold, liquid = 65, 20, 10, 5

    return (
        "You can improve investments by spreading monthly savings across multiple buckets instead of chasing a single high-return option. "
        "I cannot identify guaranteed high-return assets, but I can suggest a safer allocation approach based on your current profile.\n\n"
        f"Current context: savings rate {savings_rate:.2f}%, EMI ratio {emi_ratio:.2f}%, estimated monthly savings {savings:.2f}.\n"
        f"Suggested monthly SIP-like amount: {sip_amount:.2f} ({profile} profile).\n\n"
        "Suggested allocation split (educational):\n"
        f"- Equity bucket: {equity}%\n"
        f"- Debt/Fixed-income bucket: {debt}%\n"
        f"- Gold/hedge bucket: {gold}%\n"
        f"- Liquid/emergency bucket: {liquid}%\n\n"
        "How to improve from here:\n"
        "- Increase monthly investable amount gradually as savings rate crosses 20%.\n"
        "- Keep emergency reserves strong before increasing risk.\n"
        "- Rebalance every 6 months based on cash flow and EMI changes."
    )


def _build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are AI Money Mentor. Use only provided data. "
                "Write in plain, simple language. Be warm and personal. "
                "Do not provide investment advice. No hallucinations. "
                "Return only valid JSON. Include category concentration insight, one behavioral pattern insight, "
                "and mention annual savings projection when available. "
                "If any value is missing, explicitly say it is unavailable instead of guessing.",
            ),
            (
                "human",
                "Generate strict JSON with exact keys only: monthly_story (string), risk_alerts (list of strings), "
                "actionable_suggestions (list of strings). Use this data only: {data}",
            ),
        ]
    )


def _invoke_model(state: StoryState) -> StoryState:
    model = ChatOllama(model="gpt-oss:20b-cloud", temperature=0.2)
    prompt = _build_prompt()
    chain = prompt | model
    response = chain.invoke({"data": json.dumps(state["payload"], ensure_ascii=True)})
    content = response.content if isinstance(response.content, str) else json.dumps(response.content)
    return {"payload": state["payload"], "llm_output": content}


def _extract_json_candidates(raw_text: str) -> list[str]:
    candidates: list[str] = []
    fenced = re.findall(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", raw_text, flags=re.IGNORECASE)
    candidates.extend(fenced)

    stack = 0
    start_idx = -1
    for idx, ch in enumerate(raw_text):
        if ch == "{":
            if stack == 0:
                start_idx = idx
            stack += 1
        elif ch == "}":
            if stack > 0:
                stack -= 1
                if stack == 0 and start_idx >= 0:
                    candidates.append(raw_text[start_idx : idx + 1])

    stripped = raw_text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        candidates.append(stripped)
    return candidates


def _split_list_text(text: str) -> list[str]:
    cleaned = text.strip()
    if not cleaned:
        return []
    parts = re.split(r"\n|;|\s*\|\s*|\d+\.\s+|[-*]\s+", cleaned)
    result = [part.strip(" \t-*") for part in parts if part.strip(" \t-*.")]
    return result


def _extract_section_list(raw_text: str, heading: str) -> list[str]:
    pattern = rf"{heading}\s*:?\s*(.*?)(?:\n\s*[A-Z][A-Za-z ]+\s*:|$)"
    match = re.search(pattern, raw_text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return []
    body = match.group(1).strip()
    return _split_list_text(body)


def _coerce_to_dict(candidate: str) -> dict[str, Any] | None:
    text = candidate.strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    # Tolerate single-quoted dict-like output.
    try:
        obj = ast.literal_eval(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _normalize_story_obj(obj: dict[str, Any], raw_text: str) -> dict[str, Any]:
    story = str(
        obj.get("monthly_story")
        or obj.get("story")
        or obj.get("monthly_financial_story")
        or ""
    ).strip()

    raw_alerts = obj.get("risk_alerts") or obj.get("alerts") or []
    if isinstance(raw_alerts, str):
        alerts = _split_list_text(raw_alerts)
    elif isinstance(raw_alerts, list):
        alerts = [str(item).strip() for item in raw_alerts if str(item).strip()]
    else:
        alerts = []

    raw_actions = obj.get("actionable_suggestions") or obj.get("suggestions") or obj.get("actions") or []
    if isinstance(raw_actions, str):
        actions = _split_list_text(raw_actions)
    elif isinstance(raw_actions, list):
        actions = [str(item).strip() for item in raw_actions if str(item).strip()]
    else:
        actions = []

    if not alerts:
        alerts = _extract_section_list(raw_text, "Risk Alerts")
    if not actions:
        actions = _extract_section_list(raw_text, "Actionable Suggestions")

    if not story:
        story = raw_text.strip()

    if not alerts:
        alerts = [DEFAULT_ALERT]
    if not actions:
        actions = [DEFAULT_ACTION]

    alerts = list(dict.fromkeys(alerts))[:5]
    actions = list(dict.fromkeys(actions))[:5]

    return {
        "monthly_story": story,
        "risk_alerts": alerts,
        "actionable_suggestions": actions,
    }


def _repair_with_llm(raw_text: str) -> str:
    repair_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Convert user text into strict JSON with exact keys only: monthly_story, risk_alerts, actionable_suggestions. "
                "Do not add extra keys. Keep all facts from input only.",
            ),
            (
                "human",
                "Normalize this into valid JSON only. Input:\n{raw}",
            ),
        ]
    )
    model = ChatOllama(model="gpt-oss:20b-cloud", temperature=0)
    chain = repair_prompt | model
    response = chain.invoke({"raw": raw_text})
    return response.content if isinstance(response.content, str) else json.dumps(response.content)


def _build_graph():
    graph = StateGraph(StoryState)
    graph.add_node("generate_story", _invoke_model)
    graph.add_edge(START, "generate_story")
    graph.add_edge("generate_story", END)
    return graph.compile()


def _safe_parse_json(raw_text: str) -> dict[str, Any]:
    for candidate in _extract_json_candidates(raw_text):
        obj = _coerce_to_dict(candidate)
        if obj is not None:
            return _normalize_story_obj(obj, raw_text)

    repaired = _repair_with_llm(raw_text)
    for candidate in _extract_json_candidates(repaired):
        obj = _coerce_to_dict(candidate)
        if obj is not None:
            return _normalize_story_obj(obj, raw_text)

    return _normalize_story_obj({}, raw_text)


def generate_financial_story(payload: dict[str, Any]) -> dict[str, Any]:
    app = _build_graph()
    result = app.invoke({"payload": payload, "llm_output": ""})
    parsed = _safe_parse_json(result["llm_output"])

    parsed.setdefault("monthly_story", "No story generated.")
    parsed.setdefault("risk_alerts", [])
    parsed.setdefault("actionable_suggestions", [])
    return parsed


def answer_result_query(context: dict[str, Any], user_query: str) -> str:
    if _is_investment_query(user_query):
        return _safe_investment_answer(context)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are AI Money Mentor assistant. Answer the user question using only provided analysis context. "
                "Use simple language, be concise, and provide educational guidance only. "
                "Do not recommend specific products/funds and do not promise high returns. "
                "If answer is not in context, clearly say the data is not available.",
            ),
            (
                "human",
                "Context JSON: {context}\nQuestion: {question}",
            ),
        ]
    )
    model = ChatOllama(model="gpt-oss:20b-cloud", temperature=0.1)
    chain = prompt | model
    response = chain.invoke(
        {
            "context": json.dumps(context, ensure_ascii=True),
            "question": user_query,
        }
    )
    content = response.content if isinstance(response.content, str) else json.dumps(response.content)
    cleaned = content.strip()
    if _looks_like_refusal(cleaned):
        return (
            "I can help with a practical next-step plan from your current data, but I cannot provide guaranteed-return calls. "
            "Try asking: 'How should I split my monthly savings across equity, debt, gold, and emergency fund?'"
        )
    return cleaned
