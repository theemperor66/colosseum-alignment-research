PYTHON ?= python

.PHONY: verify test source-integrity proposed verify-paper help

help:
	@echo "make verify                 CPU software tests and frozen-source integrity"
	@echo "make verify-paper EVIDENCE=/path/to/evidence.zip"

verify: source-integrity test proposed

source-integrity:
	$(PYTHON) scripts/verify_source.py

test:
	$(PYTHON) -m pytest tests

proposed:
	$(PYTHON) -m unittest discover -s proposed-code -v

verify-paper:
	@test -n "$(EVIDENCE)" || (echo 'Set EVIDENCE to the manuscript evidence.zip or extracted archive root.'; exit 1)
	$(PYTHON) scripts/verify_paper.py "$(EVIDENCE)"
