# DevOps Quickstart (v3.3.7)

This guide is for TAs and students who want to run the Image Tagger v3 stack on a laptop
or lab machine.

## 1. Prerequisites

- Docker Desktop (or an equivalent container runtime)
- Python 3.10 or 3.11 (for helper scripts and tests)
- Git (if you are cloning from a repository) or a copy of the ZIP release

You do **not** need a system-wide PostgreSQL install; the database is run in a Docker container.

## 2. First-time setup

1. Ensure Docker Desktop is running.
2. From the repository root, make the install script executable (once per clone):

   ```bash
   chmod +x install.sh
   ```

3. Run the installer:

   ```bash
   ./install.sh
   ```

   By default this will:

   - Build Docker images for the API, database and frontend (using `deploy/docker-compose.yml`).
   - Run seeding scripts:
     - `backend/scripts/seed_tool_configs.py`
     - `backend/scripts/seed_attributes.py`
   - Run smoke tests:
     - `scripts/smoke_api.py`
     - `scripts/smoke_science.py`

If any step fails, the script prints an error message and exits with a non-zero status code.

## 3. Verifying the system

After `./install.sh` completes successfully:

1. Open a browser and navigate to the frontend portal (for example):

   - `http://localhost:8000/index.html` or
   - `http://localhost:8080/index.html`

   The exact port depends on your `deploy/docker-compose.yml` configuration.

2. You should see the **Role Portal**, with links to:

   - Tagger Workbench
   - Supervisor Monitor
   - Admin Cockpit
   - Research Explorer

3. Check the API health endpoint in a browser or via `curl`:

   ```bash
   curl http://localhost:8000/health
   ```

   A simple JSON response indicates the API is up.

## 4. Authentication and roles in dev mode

The backend uses a simple header-based RBAC scheme (`backend/services/auth.py`):

- `X-User-Id` (default: `1`)
- `X-User-Role` (default: `"tagger"`)

Admin-only endpoints (e.g. Monitor and Admin routers) require `X-User-Role: admin`.

For local development, the v3.3.7 frontend wires this header into the Monitor and Admin
clients by constructing:

- `new ApiClient('/api/v1/monitor', { 'X-User-Role': 'admin' });`
- `new ApiClient('/api/v1/admin', { 'X-User-Role': 'admin' });`

If you deploy behind a reverse proxy or an auth gateway, you can remove or override these
defaults and have your infrastructure inject the appropriate headers.

## 5. Common issues

- **Docker not running**  
  Symptoms: `docker-compose` commands fail or hang.  
  Fix: Start Docker Desktop and re-run `./install.sh`.

- **Port already in use**  
  Symptoms: API or frontend container fails to bind to a port (for example 8000).  
  Fix: Stop any process using that port or adjust ports in `deploy/docker-compose.yml`.

- **Database container failing**  
  Symptoms: API logs show connection errors to the DB.  
  Fix: Check Docker Desktop logs for the DB container; ensure volumes and environment
  variables in `deploy/docker-compose.yml` are correct.

## 6. Running tests

The repository includes basic API smoketests under `tests/` plus smoke scripts under `scripts/`.

- To run Python tests (outside Docker):

  ```bash
  pip install -r requirements.txt
  pytest -q
  ```

- To run tests against the Dockerised API, you can use a pattern such as:

  ```bash
  docker-compose -f deploy/docker-compose.yml exec -T api pytest -q
  ```

## 7. CI workflow

The CI recipe (`.github/workflows/ci_v3.yml`) is designed to:

1. Install Python dependencies.
2. Run governance verification:
   - `python scripts/guardian.py verify`
3. Run API smoketests:
   - `pytest tests/test_v3_api.py`
4. Optionally run `scripts/smoke_science.py` as an advisory check.

Teams can adapt this workflow to their own CI provider. The key idea is that every
published release should at least pass basic health checks and governance verification.


## Frontend smoketest expectations

The script `scripts/smoke_frontend.py` is a lightweight end-to-end check
that the portal is reachable and rendering the main shell.

- If Playwright is installed, it will launch a headless Chromium instance
  and wait for the page to reach a network-idle state.
- Otherwise it falls back to a simple HTTP GET and inspects the HTML
  response body.

The smoketest looks for at least one of a small set of key phrases, such as
"image tagger", "tagger workbench", "admin cockpit", or "research explorer".
If you significantly rename the portal or landing page copy, update the
list of phrases in `smoke_frontend.py` to keep the smoketest aligned with
the UI text.

## 5. BN / DB Health and Legacy Migration (v3.4.63+)

These steps are intended for TAs or developers who are running a real PostgreSQL
database with non-trivial data (i.e. more than just a few test images).

### 5.1 BN / DB health check

The BN / DB health checker verifies two things:

- Every ``Validation.attribute_key`` has a corresponding row in ``attributes.key``.
- Every BN candidate key exposed by the science index catalog also has a
  corresponding Attribute row.

To run the checker inside the Docker stack:

```bash
cd deploy
docker-compose exec api python -m backend.scripts.bn_db_health
```

A healthy database will report ``"ok": true``. If you see orphan validation keys
or missing candidate keys, treat this as a configuration or data hygiene issue
before running large BN exports.

You can also wire this into the governance guardian by setting
``check_bn_db_health: true`` under ``constraints`` in ``v3_governance.yml``.
This is best done once your Postgres instance is stable and seeded.

### 5.2 Legacy FK migration for Validation.attribute_key

Databases created before v3.4.63 may lack a real foreign key from
``Validation.attribute_key`` to ``attributes.key``. For new databases created
via the current ``install.sh`` (which calls ``Base.metadata.create_all``),
the constraint should already be present.

For older databases, use the migration helper (idempotent and safe to run
multiple times):

```bash
cd deploy
docker-compose exec api python -m backend.scripts.migrate_3_4_63_add_validation_fk
```

The script will:

- Inspect the live schema to see if the FK already exists.
- If missing, issue an ``ALTER TABLE`` to add the constraint.
- Print a short, human-readable summary of what it did.

After running the migration, you may re-run the BN / DB health checker to confirm
that there are no remaining orphan keys.
