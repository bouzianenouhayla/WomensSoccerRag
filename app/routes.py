from __future__ import annotations

from flask import Flask, jsonify, render_template, request

from .models import QueryRequest
from .rag_pipeline import RAGPipeline

pipeline = RAGPipeline()

def register_routes(app: Flask) -> None:
    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/ask")
    def ask():
        payload = request.get_json(force=True, silent=False) or {}
        query = QueryRequest.model_validate(payload)
        answer, _ = pipeline.answer_question(
            query.question, max_contexts=query.max_contexts
        )
        return jsonify({"answer": answer})
