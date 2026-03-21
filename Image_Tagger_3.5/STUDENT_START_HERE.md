# STUDENT START HERE – Image Tagger 3.4.74_vlm_lab_notebook_TL_runbook

If you are a **student** using Image Tagger for a course, this is your entry point.

## 1. What you should do first

1. Read: `docs/ops/Student_Quickstart_v3.4.73.md`.  
   This explains, in one page:
   - what Image Tagger is,
   - the two ways you might use it in the course,
   - and exactly what you are expected to do.
2. Check with your TA which **track** your course is using:
   - **Track A – Full App (persistent, Docker-based)**  
     You will normally be given a URL to a running instance, or in some cases asked to run Docker locally.
   - **Track B – Colab Science Notebook (ephemeral)**  
     You will use the `notebooks/VLM_Health_Lab.ipynb` notebook in Google Colab to run a small “VLM Health Lab.”

Once you know your track, follow the corresponding section in the Student Quickstart.

## 2. Quick mental model

- Image Tagger is a research tool for **tagging architectural images** and analysing how different spaces are perceived.
- In the course, you will **not** be rewriting the internals.
  You will:
  - run tagging jobs (Track A), or
  - run a small, scripted lab (Track B),
  - and then interpret or analyse the outputs.

## 3. If you are on Track A (Full App)

- Your TA will either:
  - give you a URL (recommended), or
  - ask you to run the app via Docker (only if you are comfortable with that).
- Follow the “Track A – Full App” section in `docs/ops/Student_Quickstart_v3.4.73.md`.

## 4. If you are on Track B (Colab VLM Health Lab)

- You will open a Colab notebook (`VLM_Health_Lab.ipynb`),
  upload a copy of this repository zip when prompted,
  and step through a small, **self-contained** experiment that:
  - creates a tiny dataset of synthetic images,
  - runs the Image Tagger science pipeline on them,
  - runs a VLM variance audit.
- Follow the “Track B – Colab Notebook” section in `docs/ops/Student_Quickstart_v3.4.73.md`.

**Remember:** in Colab, nothing in `/content` persists after the runtime restarts.
Download or save your outputs (CSVs, text summaries) if you will need them later.

## 5. When you need help

When you contact your TA for help, always include:

- which track you are using (**Track A** or **Track B**),
- what you were trying to do in one sentence,
- a screenshot of the error, and
- either:
  - the URL/path you were on (for Track A), or
  - the notebook cell you just ran (for Track B).

With that information, your TA can usually diagnose the problem quickly.
