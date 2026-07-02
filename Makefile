PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
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

.PHONY: doctor subrepo-doctor subrepo-test subrepo-assemble subrepo-runtime-root subrepo-runtime-sanity subrepo-smoke subrepo-daily-contract subrepo-ops-contract subrepo-pin-ci-green ops-preinstall-ready ops-install-launchagents ops-deployment-ready snapshot snapshot-check

# One-command live-system health check: pin/runtime drift, lock integrity,
# bundle self-consistency, promote-backup hygiene. Exit non-zero on any RED.
doctor:
	$(PYTHON) scripts/system_doctor.py

# Regenerate the machine-generated strategy-104 production snapshot from the
# PINNED subrepo config + artifact metadata (A6 / unified-107 M9). Run this
# after any promote/rollback or lock-pin bump, then commit the result.
snapshot:
	$(PYTHON) scripts/render_strategy_104_snapshot.py

# Staleness guard: exit non-zero if regenerating the snapshot from this
# machine's pinned sources produces anything different from the committed
# doc/arch/strategy-104-snapshot.md.
snapshot-check:
	$(PYTHON) scripts/render_strategy_104_snapshot.py --check

subrepo-doctor:
	$(PYTHON) scripts/subrepo_doctor.py

subrepo-test:
	$(PYTHON) scripts/subrepo_doctor.py --run-tests

subrepo-assemble:
	$(PYTHON) scripts/subrepo_assemble.py

subrepo-runtime-root:
	$(PYTHON) scripts/subrepo_assemble.py --sync --runtime-root .subrepo_runtime/repos

subrepo-runtime-sanity:
	$(PYTHON) scripts/runtime_qp_sanity_check.py

subrepo-smoke:
	$(PYTHON) scripts/subrepo_smoke.py

subrepo-daily-contract:
	$(PYTHON) scripts/subrepo_daily_contract.py $(SUBREPO_DAILY_ARGS)

subrepo-ops-contract:
	$(PYTHON) scripts/subrepo_ops_contract.py

subrepo-pin-ci-green:
	$(PYTHON) scripts/check_lock_pins_ci_green.py

ops-preinstall-ready:
	$(PYTHON) scripts/check_ops_deployment_ready.py --skip-launchagents

ops-install-launchagents: subrepo-runtime-root
	PYTHON=$(PYTHON) bash scripts/install_launchagents.sh
	$(PYTHON) scripts/check_ops_deployment_ready.py

ops-deployment-ready:
	$(PYTHON) scripts/check_ops_deployment_ready.py
