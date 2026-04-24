"""FastAPI server exposing the LangGraph agent as a REST API."""

from fastapi import FastAPI
from agent.graph import graph

app = FastAPI(title="LangGraph Agent API")


@app.post("/invoke")
async def invoke(input_data: dict):
    """Invoke the LangGraph agent graph with the provided input.

    Expected input shape:
        {"changeme": "your input string"}

    Optional configurable context can be passed via LangGraph's config:
        {"changeme": "...", "config": {"configurable": {"my_configurable_param": "..."}}}
    """
    config = input_data.pop("config", None)
    result = await graph.ainvoke(input_data, config=config)
    return result


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}
