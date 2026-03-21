# First Dashboard Quickstart

This guide shows one minimal path from a fresh install to a useful
dashboard view in the Supervisor Monitor and Research Explorer.

1. **Bring up the stack**

   Run:

   ```bash
   ./install.sh
   ```

   This starts the database, backend API, and frontend portal. If the
   science smoketest reports that there are no images yet, that is expected
   on a fresh database.

2. **Upload a small batch of images as Admin**

   - Open the frontend portal in your browser.
   - Go to the **Admin Cockpit**.
   - Use the **Bulk Upload** panel to select a small folder of images.
   - Click **Upload**. The status line will report how many images were created.

3. **Run the science pipeline**

   Once images exist in the database, run:

   ```bash
   python3 scripts/import_harness.py
   python3 scripts/smoke_science.py
   ```

   The import harness will exercise the science pipeline on a sample of
   images; the smoketest will confirm that key indices are being written
   into `Validation` rows.

4. **Inspect team metrics in the Monitor**

   - Assign yourself or a student as a **tagger** (set `X-User-Role: tagger`
     in your API client or browser plugin).
   - Use the **Tagger Workbench** to create some validations for the
     uploaded images.
   - Open the **Supervisor Monitor** and confirm that team statistics and
     IRR metrics begin to appear. If the dashboard reports that no team
     statistics are available yet, continue tagging.

5. **Explore validated images**

   - Open the **Research Explorer**.
   - Use filters to narrow down by attributes or tool configuration.
   - Export the current selection for downstream analysis (e.g., BN or
     statistical modeling).

This sequence – upload images, run science, tag, then monitor and explore –
is the canonical “first dashboard” path for Image Tagger.
