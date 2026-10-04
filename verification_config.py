#!/usr/bin/env python3
"""
Runtime verification configuration -- toggles for the three safety layers
(V_s schema/parameter, V_i intent consistency, V_c context-aware gating)
described in the thesis's System Architecture / Runtime Command Verification
sections. Lives in its own module so every call site (dashboard_server.py,
run_verification_benchmark.py, tests) reads the same source of truth, and so
the benchmark harness can switch between the paper's C0-C3 configurations
without restarting the dashboard process.
"""

import os
from dataclasses import dataclass


@dataclass
class VerificationConfig:
    schema_enabled: bool = True
    intent_enabled: bool = True
    context_enabled: bool = True
    # Reject out-of-range parameters outright rather than clamping them to
    # the nearest valid value. Clamping would make "what was verified" and
    # "what executed" diverge, which the true/false-rejection analysis
    # (RQ1/RQ2) depends on staying identical -- see the thesis's Section 4
    # design-decisions note.
    clamp_instead_of_reject: bool = False


# Process-wide current config. Mutated in place (attributes set on the same
# object) rather than reassigned, so any module that imported this object
# earlier (e.g. dashboard_server.py at startup) still sees later changes
# made by set_preset()/set_config() without needing to re-import.
_current = VerificationConfig(
    schema_enabled=os.getenv("VERIFY_SCHEMA", "1") != "0",
    intent_enabled=os.getenv("VERIFY_INTENT", "1") != "0",
    context_enabled=os.getenv("VERIFY_CONTEXT", "1") != "0",
    clamp_instead_of_reject=os.getenv("VERIFY_CLAMP", "0") != "0",
)

# The pre-existing action-name whitelist (action_registry.py) is NOT gated
# by any of these presets -- it is baseline infrastructure active in every
# configuration including C0. These flags control only the additional
# parameter-range check within V_s, plus V_i and V_c.
_PRESETS = {
    "C0": dict(schema_enabled=False, intent_enabled=False, context_enabled=False),
    "C1": dict(schema_enabled=True, intent_enabled=False, context_enabled=False),
    "C2": dict(schema_enabled=True, intent_enabled=True, context_enabled=False),
    "C3": dict(schema_enabled=True, intent_enabled=True, context_enabled=True),
}


def get_verification_config() -> VerificationConfig:
    return _current


def set_preset(name: str) -> VerificationConfig:
    """Switch the process-wide config to one of the paper's C0-C3 ablation
    configurations."""
    if name not in _PRESETS:
        raise ValueError(f"Unknown preset {name!r}, expected one of {sorted(_PRESETS)}")
    for k, v in _PRESETS[name].items():
        setattr(_current, k, v)
    return _current


def set_config(**kwargs) -> VerificationConfig:
    """Override individual fields directly, e.g. for ad-hoc experiments
    outside the four named presets."""
    for k, v in kwargs.items():
        if not hasattr(_current, k):
            raise ValueError(f"Unknown VerificationConfig field: {k!r}")
        setattr(_current, k, v)
    return _current
