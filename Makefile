PYTHON ?= python3

.PHONY: test week3-benchmark week3-summary week3-failures

test:
	$(PYTHON) -m pytest tests

week3-benchmark:
	$(PYTHON) -m pageheat_app.week3_runbook --config configs/week3_eval_expansion.yaml --pageheat-predictor-path $(PREDICTOR)

week3-summary:
	$(PYTHON) -m pageheat_app.summarize_results --results-dir $(RESULTS_DIR) --format markdown

week3-failures:
	$(PYTHON) -m pageheat_app.failure_analysis --results-dir $(RESULTS_DIR) --dataset agent_traces --format markdown
