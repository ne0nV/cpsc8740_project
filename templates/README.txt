This pack adds evaluation + reporting utilities.

1) Copy the contents of this ZIP into your project root (it will add files under scripts/, templates/, docs/).

2) Evaluate models:
   conda activate recoapp
   python scripts/evaluate.py
   # writes artifacts/metrics.json

   # If you've trained LightFM with the previous add-ons, it will also evaluate it.
   # If not, the report will include just the popularity baseline.

3) Render the DOCX report (3–4 page template):
   pip install python-docx
   python scripts/render_report.py
   # outputs docs/AI_Model_Performance_Report.docx

Optional:
- Set REPORT_AUTHOR env var to customize author name:
  REPORT_AUTHOR="Todd Schuff" python scripts/render_report.py
