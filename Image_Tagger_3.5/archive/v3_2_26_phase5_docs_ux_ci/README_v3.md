# Image Tagger v3.0 - Enterprise Edition

This is the production-ready, micro-frontend architecture for the Image Tagger system.

## 🏗️ Architecture
* **Frontend:** Monorepo with 4 distinct React Apps (Workbench, Monitor, Admin, Explorer).
* **Backend:** Unified FastAPI service with PostgreSQL.
* **Infrastructure:** Docker Compose orchestration with Nginx Gateway.

## 🚀 Quick Start (The "Enterprise Go")

1.  **Ensure Docker is installed.**
2.  **Run:**
    ```bash
    cd deploy
    docker-compose up --build
    ```
3.  **Access the GUIs:**
    * **Research Explorer:** [http://localhost:8080/explorer](http://localhost:8080/explorer)
    * **Tagger Workbench:** [http://localhost:8080/workbench](http://localhost:8080/workbench)
    * **Supervisor Monitor:** [http://localhost:8080/monitor](http://localhost:8080/monitor)
    * **Admin Cockpit:** [http://localhost:8080/admin](http://localhost:8080/admin)

## 🧪 Running Tests

To verify the API logic without Docker:
1.  `pip install pytest httpx`
2.  `pytest tests/test_v3_api.py`

## 🤖 AI Collaboration Workflow

For guidelines on how to use LLMs (ChatGPT, Claude, Gemini, etc.) with this
repository — including ZIP + concatenated TXT expectations and Guardian
governance rules — see:

- `docs/AI_COLLAB_WORKFLOW.md`