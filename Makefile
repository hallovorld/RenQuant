PYTHON ?= python3

.PHONY: subrepo-doctor subrepo-test subrepo-assemble subrepo-smoke

subrepo-doctor:
	$(PYTHON) scripts/subrepo_doctor.py

subrepo-test:
	$(PYTHON) scripts/subrepo_doctor.py --run-tests

subrepo-assemble:
	$(PYTHON) scripts/subrepo_assemble.py

subrepo-smoke:
	$(PYTHON) scripts/subrepo_smoke.py
