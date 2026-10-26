"""HTTP API for Decider².

    locket-serve --model models/locket-decider-2-7b [--port 8000]

POST /v1/decide
    {"input": "...", "options": ["billing", {"label": "technical", "description": "..."}],
     "task": "...", "criteria": "...", "explain": true, "min_confidence": 0.8}
->  {"choice": "A", "label": "billing", "confidence": 0.93, "probabilities": {...},
     "needs_review": false, "reason": "..."}
"""

from __future__ import annotations

import argparse
import threading

from pydantic import BaseModel, Field

from .decider import Decider
from .schema import MAX_OPTIONS, decision_from_dict


# Request models live at module level: FastAPI resolves the (postponed) annotations here.
class OptionIn(BaseModel):
    label: str
    description: str = ""
    abstain: bool = False


class DecideIn(BaseModel):
    input: str = Field(min_length=1)
    options: list[str | OptionIn] = Field(min_length=2, max_length=MAX_OPTIONS)
    task: str = ""
    criteria: str = ""
    explain: bool = True
    min_confidence: float | None = Field(default=None, ge=0, le=1)


def create_app(decider: Decider):
    from fastapi import FastAPI, HTTPException

    app = FastAPI(title="Locket Decider²")
    lock = threading.Lock()  # one GPU, one request at a time

    @app.get("/health")
    def health():
        return {"status": "ok", "temperature": decider.config.temperature, "min_confidence": decider.config.min_confidence}

    @app.post("/v1/decide")
    def decide(req: DecideIn):
        try:
            d = decision_from_dict(req.model_dump())
        except (ValueError, KeyError) as e:
            raise HTTPException(status_code=422, detail=str(e))
        with lock:
            return decider.decide(d, explain=req.explain, min_confidence=req.min_confidence)

    return app


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--qlora", action="store_true")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args(argv)

    import uvicorn

    uvicorn.run(create_app(Decider.load(args.model, args.adapter, load_in_4bit=args.qlora)), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
