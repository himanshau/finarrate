from __future__ import annotations

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agents.pipeline import AgentPipeline
from db import init_db, load_transactions, save_transactions, save_upload
from llm_service import answer_result_query

# FastAPI application entrypoint for AI Money Mentor.
app = FastAPI(title="AI Money Mentor API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Keep upload latency predictable by default; LLM is still used in narrative and chat paths.
pipeline = AgentPipeline.build_default(categorization_llm_enabled=False)


class UploadRef(BaseModel):
    upload_id: int


class ChatQuery(BaseModel):
    upload_id: int
    query: str


@app.on_event("startup")
def startup_event() -> None:
    init_db()


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "AI Money Mentor backend is running."}


@app.post("/upload")
async def upload_statement(file: UploadFile = File(...)) -> dict:
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Uploaded file must have a filename.")
        content = await file.read()
        filename = file.filename
        result = pipeline.parse_and_categorize(filename, content)
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result.get("message", "Upload parse failed."))

        transactions = result.get("transactions", [])
        if not transactions:
            raise HTTPException(status_code=400, detail="No transactions could be extracted.")

        upload_id = save_upload(filename)
        save_transactions(upload_id, pd.DataFrame(transactions))

        return {
            "upload_id": upload_id,
            "filename": filename,
            "transactions": transactions,
            "agent_status": result.get("message", "Categorized."),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Upload failed: {str(exc)}") from exc


@app.get("/analyze/{upload_id}")
def analyze(upload_id: int) -> dict:
    try:
        df = load_transactions(upload_id)
        if df.empty:
            raise HTTPException(status_code=404, detail="No transactions found for this upload_id.")

        analysis = pipeline.analyze({"transactions": df.to_dict(orient="records")})
        if analysis.get("status") != "ok":
            raise HTTPException(status_code=500, detail=analysis.get("message", "Analysis failed."))

        return {
            "upload_id": upload_id,
            "metrics": analysis.get("metrics", {}),
            "health_score": analysis.get("health_score", {}),
            "risk_alerts": analysis.get("risk_alerts", []),
            "annual_savings_projection": analysis.get("annual_savings_projection", 0.0),
            "planner": analysis.get("planner", {}),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(exc)}") from exc


@app.post("/analyze")
def analyze_from_body(payload: UploadRef) -> dict:
    return analyze(payload.upload_id)


@app.post("/generate-story/{upload_id}")
def generate_story(upload_id: int) -> dict:
    try:
        df = load_transactions(upload_id)
        if df.empty:
            raise HTTPException(status_code=404, detail="No transactions found for this upload_id.")

        analysis = pipeline.analyze({"transactions": df.to_dict(orient="records")})
        if analysis.get("status") != "ok":
            raise HTTPException(status_code=500, detail=analysis.get("message", "Analysis failed before story generation."))

        story_result = pipeline.narrative(analysis, df.to_dict(orient="records"))
        story = story_result.get("story", {})

        return {
            "upload_id": upload_id,
            "story": story,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Story generation failed: {str(exc)}") from exc


@app.post("/generate-story")
def generate_story_from_body(payload: UploadRef) -> dict:
    return generate_story(payload.upload_id)


@app.post("/chat-insight")
def chat_insight(payload: ChatQuery) -> dict:
    try:
        df = load_transactions(payload.upload_id)
        if df.empty:
            raise HTTPException(status_code=404, detail="No transactions found for this upload_id.")

        analysis = pipeline.analyze({"transactions": df.to_dict(orient="records")})
        if analysis.get("status") != "ok":
            raise HTTPException(status_code=500, detail=analysis.get("message", "Analysis failed before chat."))

        context = {
            "metrics": analysis.get("metrics", {}),
            "health_score": analysis.get("health_score", {}),
            "risk_alerts": analysis.get("risk_alerts", []),
            "annual_savings_projection": analysis.get("annual_savings_projection", 0.0),
            "planner": analysis.get("planner", {}),
        }
        answer = answer_result_query(context, payload.query)
        return {"upload_id": payload.upload_id, "answer": answer}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chat insight failed: {str(exc)}") from exc
