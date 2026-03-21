# 🛠️ Technical Lead Runbook – Image Tagger 3.4.74_vlm_lab_notebook_TL_runbook

This document is for the **Technical Lead (TL)** responsible for:

- getting Image Tagger up and running,
- verifying that both teaching tracks work, and
- making the system easy to share with students and collaborators.

It assumes you are comfortable with basic command-line work and Docker.

---

## 1. What you are supporting

The repository supports two main teaching modes:

- **Track A – Full App (persistent)**  
  - Full stack with Workbench, Explorer, Admin.
  - Runs in Docker on a server, Codespace, or lab machine.
  - Suitable for multi-week projects and demos with persistent data.

- **Track B – Colab “VLM Health Lab” (ephemeral)**  
  - Google Colab notebook: `notebooks/VLM_Health_Lab.ipynb`.
  - Sets up a small DB and toy image set inside Colab.
  - Runs the science pipeline and VLM variance audit.
  - Ideal for short, self-contained labs that do not require persistence.

Your goal is to ensure that **at least one Track A deployment** and the **Track B notebook** are working
and documented before students arrive.

---

## 2. Artifacts you need

- The repository zip, for example:
  - `Image_Tagger_3.4.74_vlm_lab_notebook_TL_runbook_full.zip`
- The core docs inside the repo:
  - `STUDENT_START_HERE.md`
  - `docs/ops/Student_Quickstart_v3.4.73.md`
  - `docs/ops/Cloud_AntiGravity_Quickstart.md`
  - `docs/ops/VLM_Health_Quickstart.md` (if present)
  - `docs/ops/VLM_Health_SOP.md` (if present)
  - `docs/ops/Technical_Lead_Runbook_v3.4.74.md` (this file)

Optional but recommended external docs (if provided):

- A **Repo Overview** one-pager (DOCX or PDF).
- A **TA & Student Guide** (DOCX or PDF).

---

## 3. Track A – Full App Deployment

You have three main options: local machine, GitHub Codespaces, or cloud VM / lab server.

### 3.1 Local machine (for your own development / testing)

1. Install Docker and docker-compose.
2. Unzip the repository into a folder on your machine.
3. Open a terminal in that folder and run:

   ```bash
   ./auto_install.sh
   ```

   - First run: builds images, runs migrations and seeds.
   - Subsequent runs: should be quicker, mainly verifying that things still work.

4. Start the stack (command may vary slightly by repo config):

   ```bash
   docker compose up
   ```

5. Open the main frontend URL (often `http://localhost:8080` or as documented in the code).

**Smoke test:**

- Confirm you can:
  - open Workbench,
  - open Explorer,
  - access Admin (and see VLM Health views).

### 3.2 GitHub Codespaces (recommended for remote teaching)

1. Ensure the repository is on GitHub.
2. From the GitHub repo page:
   - Click **Code → Codespaces → Create codespace on main**.
3. Once the Codespace is ready, open the integrated terminal and run:

   ```bash
   ./auto_install.sh
   ```

4. Use the **Ports** panel in Codespaces:
   - Find the port bound to the main frontend.
   - Click the globe icon to open it in your browser.
   - Optionally make the port public for temporary sharing in class.

**Advantages:**

- All students with a browser can access the same instance (if you open the URL).
- No local installation on student machines.
- You can snapshot or re-create environments as needed.

### 3.3 Cloud VM or lab server

1. Provision an Ubuntu 22.04+ VM (or choose an existing lab server).
2. Copy the repository zip to the VM.
3. SSH into the VM and run:

   ```bash
   chmod +x infra/cloud/full_stack_vm_setup.sh
   ./infra/cloud/full_stack_vm_setup.sh
   ```

   This script will:

   - install Docker if necessary,
   - unpack the repo zip into an `image_tagger` directory (if present),
   - run `./auto_install.sh`.

4. Expose the frontend:

   - For quick demos, use `ngrok http 8080` and share the generated URL.
   - For longer-term use, configure a proper reverse proxy or load balancer.

**Checklist for Track A:**

- [ ] `./auto_install.sh` completes without errors.
- [ ] Workbench loads.
- [ ] Explorer loads.
- [ ] Admin loads and VLM Health pages are reachable.
- [ ] Your chosen sharing mechanism (Codespaces URL, VM+ngrok) is documented for students.

---

## 4. Track B – Colab VLM Health Lab

The notebook `notebooks/VLM_Health_Lab.ipynb` is a self-contained “science lab” path.

### 4.1 TL verification

1. Download `Image_Tagger_3.4.74_vlm_lab_notebook_TL_runbook_full.zip` to your local machine.
2. Open the notebook in Google Colab:
   - Go to https://colab.research.google.com/
   - Choose **File → Upload notebook**, select `notebooks/VLM_Health_Lab.ipynb`.
3. Run each cell in order:
   - Step 1: environment setup (libraries + Postgres).
   - Step 2: upload the repo zip.
   - Step 3: DB init + seeds + synthetic images.
   - Step 4: run science pipeline.
   - Step 5: run VLM variance audit and view the CSV.

4. Confirm that:
   - the notebook runs end-to-end without crashing,
   - at least one variance CSV is produced and displayed.

### 4.2 Sharing instructions with TAs and students

- Make sure the **Student Quickstart** and any lab handouts:
  - clearly label this as **Track B – Colab VLM Health Lab**,
  - mention its **ephemeral** nature,
  - and provide the correct zip and notebook to use.

- Optionally create:
  - a short screencast or screenshot sequence,
  - or a one-page PDF summarising the steps.

---

## 5. Minimal GO/NO-GO Gate for the Technical Lead

Before the course starts, the TL should be able to answer **YES** to:

1. **Governance / guards:**
   - [ ] `python scripts/syntax_guard.py` passes.
   - [ ] `python scripts/program_integrity_guard.py` passes.
   - [ ] `python scripts/critical_import_guard.py` passes.
   - [ ] `python scripts/canon_guard.py` passes.

2. **Track A:**
   - [ ] I can run `./auto_install.sh` to completion in at least one environment (local, Codespace, VM).
   - [ ] I can open Workbench, Explorer, and Admin in a browser.
   - [ ] I know what URL to give students (and under what conditions).

3. **Track B:**
   - [ ] I can run the full `VLM_Health_Lab.ipynb` notebook in Colab without errors.
   - [ ] I know which zip students should upload.
   - [ ] I have told TAs which parts of the notebook matter for their assignments.

4. **Documentation:**
   - [ ] `STUDENT_START_HERE.md` and `docs/ops/Student_Quickstart_v3.4.73.md` exist and reflect our actual teaching plan.
   - [ ] `docs/ops/Cloud_AntiGravity_Quickstart.md` is accurate for our deployment strategy.
   - [ ] TAs know where to find any external guides (Repo Overview, TA & Student Guide).

If any of these are “no,” treat that as a **pre-course bug** and resolve it before students touch the system.

---

## 6. Communication with the teaching team

Share with TAs:

- where the Track A instance lives (URL, credentials if any),
- whether Track B will be used and how,
- what *not* to change (e.g., governance files, guard scripts),
- how to escalate issues (what logs to send you, what screenshots to collect).

With this runbook and the in-repo quickstarts, the Technical Lead should be able to maintain a
stable teaching instance and support both the full application and the Colab lab.
