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

## 3. Hugging Face Spaces (Docker SDK)
1. Create a new Space → **SDK: Docker** → blank template.
2. Push this repository to the Space (the YAML front-matter at the top of `README.md` already sets `sdk: docker` and `app_port: 7860`):
   ```bash
   git remote add space https://huggingface.co/spaces/<user>/mcpguard
   git push space main
   ```
   Use `git lfs track "*.joblib" "*.npz"` if the Space rejects files larger than 10 MB.
3. The Space builds the `Dockerfile` and exposes the dashboard at `https://<user>-mcpguard.hf.space`.
4. Optional secret: `MCPGUARD_ADMIN_TOKEN` (protects `POST /reload-model`).

**Status: PENDING USER ACTION.** Pushing to a Space needs your Hugging Face account and token, so it has not been done from this environment. After deploying, record the live URL in `FINAL_REQUIREMENTS_CHECKLIST.md`.

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
