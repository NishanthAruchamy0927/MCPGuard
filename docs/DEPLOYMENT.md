# Deployment

MCPGuard is one FastAPI process that serves the REST API **and** the dashboard on one port (default **7860**). The trained artefacts are baked into the image and nothing is retrained at start-up.

## 1. Local (no Docker)
```bash
python -m venv .venv && source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m src.pipeline.run_all                                # ~20 min on a laptop CPU
uvicorn backend.app.main:app --port 7860                      # http://localhost:7860  ·  /docs
```

## 2. Docker
```bash
docker compose up --build        # http://localhost:7860
# or
docker build -t mcpguard . && docker run -p 7860:7860 mcpguard
```
The image runs as non-root user `appuser` (UID 1000), has a `HEALTHCHECK` on `/health`, and ships the MiniLM embedding model so it starts offline.

## 3. Hugging Face Spaces (Gradio SDK, free tier)
Hugging Face restricts the Docker SDK to paid plans, so the Space uses the free **Gradio SDK**. [`space_app.py`](../space_app.py) serves the whole FastAPI app (dashboard at `/`, REST API, `/docs`) and mounts a Gradio analyzer at `/gradio`. The `README.md` front-matter sets `sdk: gradio`, `sdk_version`, `python_version` and `app_file: space_app.py`.

1. Create a Space: **SDK: Gradio**, template **Blank**, free hardware, visibility **Public**.
2. Push the lean `hf-space` branch (one commit, serving dependencies only, binaries in Git LFS):
   ```bash
   git remote add space https://huggingface.co/spaces/<user>/MCPGuard
   git push space hf-space:main --force      # password = a Hugging Face *write* token
   ```
3. Watch the **Logs** tab; the first build installs PyTorch and takes 10–20 min. The app is then live at `https://<user>-mcpguard.hf.space` (Gradio UI at `/gradio`).
4. Optional secret: `MCPGUARD_ADMIN_TOKEN` (protects `POST /reload-model`).

Run the Space entry point locally with `python space_app.py`.

## 4. AWS / GCP (alternative)
- **AWS App Runner / ECS Fargate:** push the image to ECR (`docker tag mcpguard <acct>.dkr.ecr.<region>.amazonaws.com/mcpguard && docker push …`), create a service on port 7860 with health-check path `/health`.
- **GCP Cloud Run:** `gcloud run deploy mcpguard --source . --port 7860 --memory 2Gi`.
- For persistent prediction logs, set `DATABASE_URL` to a managed PostgreSQL instance.

## Configuration
| Variable | Default | Purpose |
|---|---|---|
| `PORT` | 7860 | HTTP port |
| `DATABASE_URL` | `sqlite:///artifacts/mcpguard.db` | Any SQLAlchemy URL |
| `MCPGUARD_ADMIN_TOKEN` | unset | If set, required as `X-Admin-Token` for `/reload-model` |
| `MCPGUARD_LOG_JSON` | 1 in Docker | Structured JSON logs |
| `MCPGUARD_CONFIG` | `configs/config.yaml` | Alternative config file |
