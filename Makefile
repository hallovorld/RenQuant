PYTHON ?= python3
SUBREPO_DAILY_ARGS :=
ifneq ($(strip $(BROKER_TYPE)),)
SUBREPO_DAILY_ARGS += --broker-type $(BROKER_TYPE)
endif
ifneq ($(strip $(BROKER_NAME)),)
SUBREPO_DAILY_ARGS += --broker-name $(BROKER_NAME)
endif
ifneq ($(strip $(EXECUTE)),)
SUBREPO_DAILY_ARGS += --execute
endif

.PHONY: subrepo-doctor subrepo-test subrepo-assemble subrepo-smoke subrepo-daily-contract subrepo-ops-contract

subrepo-doctor:
	$(PYTHON) scripts/subrepo_doctor.py

subrepo-test:
	$(PYTHON) scripts/subrepo_doctor.py --run-tests

subrepo-assemble:
	$(PYTHON) scripts/subrepo_assemble.py

subrepo-smoke:
	$(PYTHON) scripts/subrepo_smoke.py

subrepo-daily-contract:
	$(PYTHON) scripts/subrepo_daily_contract.py $(SUBREPO_DAILY_ARGS)

subrepo-ops-contract:
	$(PYTHON) scripts/subrepo_ops_contract.py
