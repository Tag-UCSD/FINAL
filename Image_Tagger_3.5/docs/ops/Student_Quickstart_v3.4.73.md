# 🎓 Student Quickstart – Image Tagger 3.4.74_vlm_lab_notebook_TL_runbook

This page explains **how you will actually use Image Tagger in this course**.

You do not need to understand every internal detail.
You just need to know which path you are on and what to run or click.

---

## 1. What Image Tagger is (for you)

Image Tagger is a system that:

- stores architectural images,
- runs a science pipeline (visual and spatial metrics, and optionally VLMs),
- and lets you explore the results through web interfaces.

In this course you will use Image Tagger to:

- run defined experiments,
- inspect tags and metrics,
- and connect them to theoretical ideas about space, perception, and affect.

You are **not** expected to maintain the infrastructure.

---

## 2. Two tracks: Full App vs Colab Lab

Your instructor or TA will tell you which track you are using:

### Track A – Full App (persistent, Docker-based)

- You access Image Tagger as a normal web app (via a URL or localhost).
- The system runs in Docker containers on:
  - a lab machine, or
  - a cloud VM / GitHub Codespace, or
  - occasionally your own laptop (if you have Docker and are comfortable using it).
- The database and image store are **persistent**:
  - if the machine is stopped and restarted, your data remains available (unless explicitly reset).

**You will:**

- visit a URL your TA provides,
- use Workbench to launch tagging jobs (if part of your assignment),
- use Explorer to view and compare images and tags,
- possibly look at Admin views if the assignment asks you to.

### Track B – Colab Science Notebook (ephemeral VLM Health Lab)

- You work in **Google Colab** using the notebook:
  - `notebooks/VLM_Health_Lab.ipynb`
- The notebook:
  - unpacks a copy of the Image Tagger repository,
  - sets up a small local database,
  - generates a tiny synthetic image set,
  - runs the science pipeline,
  - and then runs a VLM variance audit.

**Important:** Colab is **ephemeral**.

- When the Colab runtime disconnects, everything under `/content` is wiped:
  - the unpacked repo,
  - any CSVs,
  - any generated reports.

**How to keep your work:**

- Save the notebook itself to Google Drive.
- Download CSVs and text summaries you need for your writeup.
- Optionally mount Google Drive and save outputs there.

---

## 3. Track A – How to use the Full App

If your TA says you are using **Track A**, they will usually provide a URL for you to open.

### 3.1 If you are given a URL

1. Open the URL in a modern browser (Chrome, Edge, Firefox, Safari).
2. Follow your lab or assignment sheet, which will tell you:
   - whether to start in **Workbench** (running a job),
   - or **Explorer** (inspecting existing results),
   - or a specific view in **Admin**.

You should not need to run any terminal commands in this mode.

### 3.2 If you are asked to run locally with Docker

This will only happen if you are comfortable with Docker and your machine can support it.

1. Install Docker Desktop (or Docker Engine) if you do not already have it.
2. Unzip the Image Tagger repository into a folder on your machine.
3. Open a terminal in that folder and run:

   ```bash
   ./auto_install.sh
   ```

   The first run may take several minutes.

4. When instructed (by your TA or the script), start the stack (for example):

   ```bash
   docker compose up
   ```

5. Open the URL indicated by your TA (often `http://localhost:8080`).

If you hit errors in this mode, provide the terminal output and any error messages to your TA.

---

## 4. Track B – How to use the Colab VLM Health Lab

If your TA says you are using **Track B**, you will work primarily in Google Colab.

1. Make sure you have:
   - a Google account, and
   - the Image Tagger zip specified by your instructor (for example: `Image_Tagger_3.4.74_vlm_lab_TL_runbook_full.zip`).

2. Open Google Colab:
   - Go to https://colab.research.google.com/
   - Choose **File → Upload notebook**.
   - Select `notebooks/VLM_Health_Lab.ipynb` from the repository.

3. In the notebook, run the cells in order:

   - **Step 1 – Setup Environment**  
     Installs required libraries and starts a local PostgreSQL database inside Colab.

   - **Step 2 – Upload Repository Zip**  
     You will be prompted to upload the Image Tagger zip.
     The notebook unpacks it into `/content/repo`.

   - **Step 3 – Seed Database & Generate Toy Images**  
     Creates database tables, seeds configuration (if seed scripts are present), and generates a handful of synthetic “architectural” images.

   - **Step 4 – Run Science Pipeline**  
     Runs the Image Tagger science pipeline on the toy images.
     If no API keys are set, the system uses a stub engine for VLM calls (which is fine for plumbing tests).

   - **Step 5 – Run VLM Variance Audit**  
     Runs `scripts/audit_vlm_variance.py` and then loads and displays the resulting CSV.

4. At the end of the lab:

   - Download the CSV(s) and any text summaries the lab asks you to submit.
   - Optionally save them to Google Drive.

**Reminder:** if you simply close the tab or leave the notebook idle until it disconnects, all files in `/content` vanish.

---

## 5. When something breaks

Before contacting your TA:

- Note **which track** you are using (Track A vs Track B).
- Take a screenshot of the error.
- Capture:
  - the URL and which page you were on (Track A), or
  - the notebook cell you just executed (Track B).
- Write one–two sentences describing:
  - what you were trying to do,
  - what you expected,
  - what actually happened.

Send this to your TA. This will greatly accelerate debugging.

---

## 6. One-sentence summary

- Track A: **web app**, persistent, usually accessed via a URL your TA provides.  
- Track B: **Colab notebook**, ephemeral, used for a small, scripted VLM health experiment.

If you are unsure which track you’re on, ask your TA **before** trying to install or run anything yourself.
