from __future__ import annotations

import altair as alt
import pandas as pd
import requests
import streamlit as st

BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="AI Money Mentor", page_icon="💰", layout="wide")
st.title("AI Money Mentor")
st.caption("Upload your statement, analyze money health, and get a plain-language monthly story.")


with st.sidebar:
    st.header("Upload Statement")
    uploaded_file = st.file_uploader("CSV or PDF bank statement", type=["csv", "pdf"])
    if "upload_id" not in st.session_state:
        st.session_state.upload_id = None

    if st.button("Upload", use_container_width=True):
        if uploaded_file is None:
            st.warning("Please select a CSV or PDF file first.")
        else:
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            try:
                response = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=180)
                response.raise_for_status()
                payload = response.json()
                st.session_state.upload_id = payload["upload_id"]
                st.success(f"Uploaded successfully. upload_id = {payload['upload_id']}")
            except requests.RequestException as exc:
                st.error(f"Upload failed: {exc}")


upload_id = st.session_state.upload_id
if upload_id is None:
    st.info("Upload a statement from the sidebar to begin.")
    st.stop()

col_a, col_b = st.columns(2)
with col_a:
    if st.button("Run Analysis", use_container_width=True):
        try:
            analyze_resp = requests.get(f"{BACKEND_URL}/analyze/{upload_id}", timeout=60)
            analyze_resp.raise_for_status()
            st.session_state.analysis = analyze_resp.json()
        except requests.RequestException as exc:
            st.error(f"Analysis failed: {exc}")

with col_b:
    if st.button("Generate AI Story", use_container_width=True):
        try:
            story_resp = requests.post(f"{BACKEND_URL}/generate-story/{upload_id}", timeout=180)
            story_resp.raise_for_status()
            st.session_state.story = story_resp.json()["story"]
        except requests.RequestException as exc:
            st.error(f"Story generation failed: {exc}")


if "analysis" in st.session_state:
    analysis = st.session_state.analysis
    metrics = analysis["metrics"]
    health = analysis["health_score"]
    planner = analysis["planner"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Income", f"{metrics['total_income']:.2f}")
    m2.metric("Total Expenses", f"{metrics['total_expenses']:.2f}")
    m3.metric("Savings Rate", f"{metrics['savings_rate']:.2f}%")
    m4.metric("EMI / Income", f"{metrics['emi_to_income_ratio']:.2f}%")

    st.subheader("Money Health Score")
    st.progress(min(max(health["score"], 0), 100), text=f"Score: {health['score']} / 100")
    st.write(health["explanation"])

    st.subheader("Category-wise Spending")
    category_map = metrics.get("category_spending", {})
    category_df = pd.DataFrame(
        [{"Category": str(k), "Amount": abs(float(v))} for k, v in category_map.items()]
    )
    if not category_df.empty:
        category_df = category_df[category_df["Amount"] >= 0].sort_values("Amount", ascending=False)
        upper = float(category_df["Amount"].max()) if not category_df.empty else 1.0
        chart = (
            alt.Chart(category_df)
            .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
            .encode(
                x=alt.X("Category:N", sort="-y", title="Category"),
                y=alt.Y("Amount:Q", title="Amount", scale=alt.Scale(domain=[0, upper * 1.1])),
                tooltip=[alt.Tooltip("Category:N"), alt.Tooltip("Amount:Q", format=",.2f")],
            )
            .properties(height=320)
        )
        st.altair_chart(chart, use_container_width=True)
        st.dataframe(category_df.set_index("Category").style.format({"Amount": "{:.2f}"}), use_container_width=True)
    else:
        st.info("No expense categories available yet.")

    if analysis.get("risk_alerts"):
        st.subheader("Risk Alerts (Logic Agent)")
        for alert in analysis.get("risk_alerts", []):
            st.info(alert)

    st.subheader("Basic Planner")
    p1, p2, p3 = st.columns(3)
    p1.metric("Savings Target", f"{planner['monthly_savings_target']:.2f}")
    p2.metric("Emergency Fund Goal", f"{planner['emergency_fund_goal']:.2f}")
    p3.metric("Suggested SIP-like Amount", f"{planner['suggested_monthly_sip_like_amount']:.2f}")
    st.caption(planner.get("emergency_fund_explanation", ""))
    st.write(f"SIP Profile: {planner.get('suitable_sip_style_for_current_portfolio', 'Balanced')}")
    st.write(planner.get("sip_guidance", ""))
    for item in planner.get("suggestions", []):
        st.success(item)

    with st.container(border=True):
        st.subheader("Ask About Your Results")
        st.caption("Ask any question about your current analysis. Responses use your uploaded data context.")
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        user_query = st.text_input("Your question", placeholder="Why is my savings rate low this month?")
        if st.button("Ask Mentor", use_container_width=True):
            if not user_query.strip():
                st.warning("Please enter a question first.")
            else:
                try:
                    chat_resp = requests.post(
                        f"{BACKEND_URL}/chat-insight",
                        json={"upload_id": upload_id, "query": user_query.strip()},
                        timeout=120,
                    )
                    chat_resp.raise_for_status()
                    answer = chat_resp.json().get("answer", "No answer generated.")
                    st.session_state.chat_history.append({"q": user_query.strip(), "a": answer})
                except requests.RequestException as exc:
                    st.error(f"Chat failed: {exc}")

        for item in reversed(st.session_state.chat_history[-5:]):
            st.markdown(f"**Q:** {item['q']}")
            st.markdown(f"**A:** {item['a']}")


if "story" in st.session_state:
    story = st.session_state.story
    st.subheader("Monthly Financial Story")
    st.write(story.get("monthly_story", "No story generated."))

    st.subheader("Risk Alerts")
    for alert in story.get("risk_alerts", []):
        st.warning(alert)

    st.subheader("Actionable Suggestions")
    for item in story.get("actionable_suggestions", []):
        st.success(item)
