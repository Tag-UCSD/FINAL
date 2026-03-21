# ☁️ Cloud "Anti-Gravity" Quickstart (3.4.74_vlm_lab_notebook_TL_runbook)

This guide explains how to run **Image Tagger 3.4.74_vlm_lab_notebook_TL_runbook** in the cloud with **minimal local setup**.
It offers two tracks:

- **TRACK A – Full Stack (persistent)**: full application with Workbench / Explorer / Admin.
- **TRACK B – Science Notebook (ephemeral)**: a lightweight Colab-based lab that focuses on the
  science pipeline and VLM health checks.

---

## 1. Persistent vs. Ephemeral Environments

- **Persistent environments** (TRACK A):
  - Example: GitHub Codespaces, a cloud VM (AWS/GCP/Azure), or a lab server.
  - When you stop and restart the machine, your files and database state **are still there**
    (unless you explicitly delete them).
  - Ideal for multi-week projects, teaching labs, and anything that needs continuity.

- **Ephemeral environments** (TRACK B / Colab):
  - Example: Google Colab free-tier notebook sessions.
  - The runtime is **temporary**: when the notebook disconnects, times out, or you close it,
    anything stored on the notebook filesystem (e.g. `/content/...`) **disappears**.
  - Only things you explicitly save to **Google Drive** or download to your own computer survive.
  - Ideal for short experiments, demos, and self-contained labs.

In this guide:

- TRACK A gives you a **persistent full Image Tagger instance**.
- TRACK B gives you an **ephemeral "Science Lab"** that you can re-run quickly, but should not
  be treated as long-term storage.

---

## 2. TRACK A – Full Stack (for TAs & Admins)

**Goal:** Run the full app (Workbench, Explorer, Admin, API) with a GUI.

**Best for:**
- Architecture demos and tagging sessions.
- Admin / TA configuration and testing.
- Multi-week projects that benefit from persistent data.

**Requires:** A cloud environment with Docker (e.g., GitHub Codespaces, AWS EC2, GCP VM, lab server).

### 2.1 GitHub Codespaces (Recommended)

1. Push or upload this repo (`Image_Tagger_3.4.74_vlm_lab_notebook_TL_runbook`) to GitHub.
2. In GitHub, open the repository page.
3. Click **Code → Codespaces → Create codespace on main**.
4. When the Codespace terminal is ready, run:

   ```bash
   ./auto_install.sh
   ```

5. In the **Ports** panel in Codespaces:
   - Find the port serving the main frontend (for example `8080`).
   - Click the globe icon to open the forwarded URL in your browser.

When you stop and restart the Codespace later, your data (Postgres, files under the repo) will still be there,
unless you delete the Codespace.

### 2.2 Generic Cloud VM (AWS/GCP/Azure or lab server)

1. Provision an Ubuntu 22.04+ VM (or use an existing lab machine).
2. Copy the `Image_Tagger_3.4.74_vlm_lab_TL_runbook_full.zip` artifact onto the machine, or clone from Git.
3. SSH into the machine and run:

   ```bash
   chmod +x infra/cloud/full_stack_vm_setup.sh
   ./infra/cloud/full_stack_vm_setup.sh
   ```

   This helper script will:

   - install Docker if needed,
   - unpack the repo ZIP into an `image_tagger` directory (if the ZIP is present),
   - run `./auto_install.sh` from inside the repo.

4. Access the UI:

   - If you are working over SSH with a browser on your local machine, you can:
     - forward ports via SSH, or
     - set up a reverse proxy / ingress.
   - For simple setups, the script prints instructions to use **ngrok**:

     ```bash
     ngrok http 8080
     ```

     Once ngrok is running, it shows a public URL you can give to students for short-term demos.

**Important:** A VM is **persistent** as long as you keep it running (or stop/start it without deleting its disk).
Do not treat it as disposable unless you intend to lose all stored images and tags.

---

## 3. TRACK B – Science Notebook (for Students and Labs)

**Goal:** Run the **science pipeline + VLM health checks** on a small example dataset without
installing Docker or the full stack.

**Best for:**

- Data analysis and method demonstrations.
- Labs focused on VLM variance, psychometrics, and Turing-style evaluation.
- Students working on personal laptops that cannot run Docker.

**Requires:** A Google account and access to **Google Colab (Free Tier)**.

### 3.1 Running the VLM Health Lab notebook

1. Obtain the artifact:

   - Download the repository zip specified by your instructor, for example:
     - `Image_Tagger_3.4.74_vlm_lab_TL_runbook_full.zip`.

   - Locate the Colab notebook in the repo:
     - `notebooks/VLM_Health_Lab.ipynb`.

2. Open Colab:

   - Go to https://colab.research.google.com/
   - Click **File → Upload notebook** and select `VLM_Health_Lab.ipynb`.

3. Follow the notebook cells:

   - **Step 1: Setup environment**  
     Installs Python libraries and a lightweight PostgreSQL instance.

   - **Step 2: Upload repo zip**  
     Unpacks the repo into `/content/repo` and switches to that directory.

   - **Step 3: Seed tiny image set**  
     Creates database tables, runs seed scripts (if present), and generates synthetic architectural images.

   - **Step 4: Run science pipeline**  
     Runs the Image Tagger science pipeline on the toy images.  
     In stub mode, the VLM returns neutral outputs; this is enough to exercise the pipeline.

   - **Step 5: Run VLM health audit**  
     Runs `scripts/audit_vlm_variance.py` and loads the resulting CSV using pandas.

4. Saving results:

   - Download the variance CSV(s) and any other outputs you care about.
   - Or write them to a mounted Google Drive folder.

**Reminder:** The Colab filesystem is **ephemeral**. Do not store anything important only in `/content`.

---

## 4. Which Track Should I Use?

- **Instructors / TAs / Technical Lead:**
  - Use **TRACK A (Full Stack)** for:
    - running the live system in class,
    - letting students tag real images via Workbench/Explorer,
    - running ongoing experiments where data persistence matters.
  - Use **TRACK B (Notebook)** when:
    - you need a low-friction lab on the science/VLM side,
    - students cannot run Docker,
    - or you want a standardised small experiment.

- **Students:**
  - Follow the instructions your TA gives.
  - Track A feels like a web app; Track B feels like a notebook-based lab.

---

## 5. Technical Lead Checklist

Before the course begins, the Technical Lead should:

- [ ] Bring up at least one **TRACK A** instance (Codespaces or VM) and confirm:
      - `./auto_install.sh` completes successfully.
      - Workbench, Explorer, and Admin load.
- [ ] Run the **TRACK B** notebook once end-to-end in Colab and confirm:
      - all five steps execute without error,
      - at least one variance CSV is produced.
- [ ] Update lab handouts to:
      - specify which track is in use,
      - point to the correct URL (Track A) or notebook + zip (Track B),
      - remind students about persistence vs. ephemerality.

With these steps done, your “anti-gravity” deployment is ready for teaching.
