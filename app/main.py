"""Single-process launcher (local dev, Hugging Face Spaces, any PaaS).

    python -m app.main            # serves API + dashboard on $PORT (default 7860)
"""
import os

import uvicorn

if __name__ == "__main__":
    uvicorn.run("backend.app.main:app", host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", "7860")))
