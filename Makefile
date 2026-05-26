PYTHON ?= python3

.PHONY: subrepo-doctor subrepo-test subrepo-assemble subrepo-smoke subrepo-daily-contract

subrepo-doctor:
	$(PYTHON) scripts/subrepo_doctor.py

subrepo-test:
	$(PYTHON) scripts/subrepo_doctor.py --run-tests

subrepo-assemble:
	$(PYTHON) scripts/subrepo_assemble.py

subrepo-smoke:
	$(PYTHON) scripts/subrepo_smoke.py

subrepo-daily-contract:
	$(PYTHON) scripts/subrepo_daily_contract.py
