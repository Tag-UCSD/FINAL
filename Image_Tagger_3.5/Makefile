

# VLM health check convenience targets
RUN_ID ?= $(shell date +%F)_$(shell cat VERSION)_main_vlm

vlm-health-init:
	mkdir -p reports/vlm_health/$(RUN_ID)/raw reports/vlm_health/$(RUN_ID)/derived
	cp reports/bn_validations_flat.csv reports/vlm_health/$(RUN_ID)/raw/ || true
	cp reports/vlm_validations.csv    reports/vlm_health/$(RUN_ID)/raw/ || true
	cp reports/human_validations.csv  reports/vlm_health/$(RUN_ID)/raw/ || true
	@echo "Initialised vlm_health run at reports/vlm_health/$(RUN_ID)"

vlm-health-audit:
	python scripts/audit_vlm_variance.py \
	  reports/vlm_health/$(RUN_ID)/raw/bn_validations_flat.csv \
	  --out reports/vlm_health/$(RUN_ID)/derived/vlm_variance_audit.csv \
	  --source-column source \
	  --attribute-column attribute_key \
	  --value-column value \
	  --source-prefix science_pipeline.vlm

vlm-health-panel:
	python scripts/vlm_turing_test_prep.py \
	  --vlm   reports/vlm_health/$(RUN_ID)/raw/vlm_validations.csv \
	  --human reports/vlm_health/$(RUN_ID)/raw/human_validations.csv \
	  --out   reports/vlm_health/$(RUN_ID)/raw/vlm_turing_panel.csv \
	  --max-trials 400 \
	  --seed 42

vlm-health-score:
	python scripts/vlm_turing_test_score.py \
	  --panel reports/vlm_health/$(RUN_ID)/raw/vlm_turing_panel_completed.csv \
	  > reports/vlm_health/$(RUN_ID)/derived/vlm_turing_summary.txt
