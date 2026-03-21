# Production Deployment Guide (v3.4.x)

This document describes how to deploy Image Tagger beyond a single laptop
development environment.

For typical classroom use, running `install.sh` locally with Docker is enough.
Use this guide only if you intend to:

- expose the system on a shared department server,
- run it for multiple users over the network,
- or keep a long-running instance for a research group.

The goals are:

- keep data and credentials safe,
- keep VLM / API keys out of the repository,
- make upgrades predictable.

---

## 1. Core components

Image Tagger v3.x consists of:

- **Backend API** (FastAPI + Postgres)
- **Frontend** (React app, typically served by nginx)
- **Science pipeline** (Python / OpenCV / NumPy)
- **Optional VLMs** (OpenAI / Anthropic / others via `backend/services/vlm.py`)

In the default Docker setup:

- `deploy/Dockerfile.backend` builds the backend image.
- `deploy/Dockerfile.frontend` (or equivalent) builds the frontend.
- `deploy/docker-compose.yml` defines services:
  - `backend`
  - `frontend`
  - `db` (Postgres)
  - optional reverse proxy (nginx) in front.

---

## 2. Environment variables

These are the most important environment variables for a production-like setup.

### 2.1 Security / auth

- `API_SECRET`  
  - Used to sign JWT tokens.
  - **Production rule:** never use the default development key.
  - Generate a long random string, e.g.:

    ```bash
    python - << 'EOF'
    import secrets
    print(secrets.token_urlsafe(64))
    EOF
    ```

  - Set it in your environment (or `.env`) before running Docker:

    ```bash
    export API_SECRET="your-long-random-secret"
    ```

- `ADMIN_EMAIL`, `ADMIN_PASSWORD` (if supported in seeding / setup scripts)  
  - Use non-trivial values; do not reuse personal passwords.

### 2.2 Database

- `POSTGRES_DB`
- `POSTGRES_USER`
- `POSTGRES_PASSWORD`

These are defined in `docker-compose.yml`. For production-like use:

- change the defaults,
- ensure the DB is not exposed to the public internet (bind to localhost or an internal network),
- back up the database volume regularly.

### 2.3 VLM keys (optional)

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`

Rules:

- Never commit these keys to the repository.
- Prefer `.env` files or deployment-specific secret stores (e.g. Docker secrets,
  Kubernetes secrets, or a password manager).
- Remember that VLM calls cost money. Use batch size and sampling strategies
  appropriate for your budget.

---

## 3. Volumes and persistence

In `deploy/docker-compose.yml` you will see volumes for:

- Postgres data (e.g. `pgdata:/var/lib/postgresql/data`)
- Image storage (e.g. `data_store:/app/data_store`)

For a production-like deployment:

1. Ensure these volumes are backed up.
2. Prefer **named volumes** or explicit host paths, not anonymous volumes, so
   that data survives container recreation.

Example snippet:

```yaml
services:
  db:
    image: postgres:16
    environment:
      POSTGRES_DB: itagger
      POSTGRES_USER: itagger
      POSTGRES_PASSWORD: change_me
    volumes:
      - pgdata:/var/lib/postgresql/data

  backend:
    build:
      context: ..
      dockerfile: deploy/Dockerfile.backend
    environment:
      API_SECRET: ${API_SECRET}
      OPENAI_API_KEY: ${OPENAI_API_KEY:-}
      ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY:-}
    volumes:
      - data_store:/app/data_store

volumes:
  pgdata:
  data_store:
```

---

## 4. HTTPS and reverse proxy

For any deployment exposed beyond localhost, you should terminate HTTPS in a
reverse proxy such as nginx or Caddy.

Typical pattern:

- nginx listens on ports 80/443.
- nginx proxies:
  - `/api/` → backend container
  - `/` → frontend container (static assets)

High-level nginx sketch:

```nginx
server {
    listen 80;
    server_name your.domain.edu;

    # Redirect HTTP → HTTPS
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl;
    server_name your.domain.edu;

    ssl_certificate     /etc/letsencrypt/live/your.domain.edu/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your.domain.edu/privkey.pem;

    location /api/ {
        proxy_pass         http://backend:8000/api/;
        proxy_set_header   Host $host;
        proxy_set_header   X-Real-IP $remote_addr;
    }

    location / {
        root   /usr/share/nginx/html;
        index  index.html;
        try_files $uri /index.html;
    }
}
```

In a lab environment you do not need a perfect TLS story, but:

- use HTTPS if students log in with passwords,
- restrict access to your department / VPN where possible.

---

## 5. Running in “production mode”

A simple default deployment is:

```bash
cd deploy
# Make sure .env contains API_SECRET and DB credentials
docker compose up --build -d
```

Once containers are healthy:

- Visit the frontend URL in a browser, e.g.:
  - `http://localhost:8080/` (direct)
  - or `https://your.domain.edu/` (via nginx).

Use the **Admin Cockpit** to:

- configure VLM engine,
- upload a small batch of images,
- run smoke tests.

---

## 6. Upgrade strategy

When a new Image Tagger version is released (e.g. v3.4.9 → v3.4.9):

1. Back up the database and `data_store` volume.
2. Replace the backend/frontend image build context with the new repo.
3. Rebuild containers:

   ```bash
   cd deploy
   docker compose build backend frontend
   docker compose up -d
   ```

4. Run smoke tests:
   - `scripts/smoke_science.py`
   - `scripts/smoke_frontend.py`
   - basic manual exploration in the browser.

If anything fails:

- roll back to the previous images and code,
- restore from backup if necessary.

---

## 7. Security checklist

Before exposing Image Tagger outside a development machine:

- [ ] `API_SECRET` is long, random, and not the default.
- [ ] Database credentials are not the defaults shipped in the repo.
- [ ] Postgres is not exposed directly to the public internet.
- [ ] HTTPS is configured on the external endpoint (or access is VPN-only).
- [ ] VLM API keys are stored in environment variables or secrets, not in source.
- [ ] Docker volumes for DB and image data are backed up regularly.
- [ ] The version (`VERSION` file, backend, README) matches the deployed tag.

If you can honestly tick all of these boxes, you are in reasonable shape for a
departmental / lab deployment.
