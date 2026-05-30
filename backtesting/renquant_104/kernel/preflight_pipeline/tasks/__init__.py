"""PreflightTask implementations (Track H migration target).

Each Task corresponds to one of the legacy ``_check_*`` functions in
``kernel.preflight``. As checks lift over (one per follow-up PR), import them
here to surface a clean public symbol set.
"""
from .state import StateFileTask
from .broker import BrokerConnectTask

__all__ = ["StateFileTask", "BrokerConnectTask"]
