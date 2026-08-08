"""nfr_targets.yaml loader and assertion evaluator.

This module is the ONLY place assertions come from (NFR-MAINT-004): every
threshold, comparator, statistic name, and unit is read from
docs/nfr/nfr_targets.yaml at run time. Scenario code computes metric values
and hands them to `evaluate`; it never compares against a literal.
"""

import operator

import yaml

from . import paths

_OPS = {
    "<=": operator.le,
    ">=": operator.ge,
    "==": operator.eq,
    "<": operator.lt,
    ">": operator.gt,
}


def load_targets(path=None):
    with open(path or paths.NFR_TARGETS_YAML, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    return doc


def targets_for_scenario(doc, scenario, environment="E-D",
                         include_conditional=False):
    """Select target ids applicable to a scenario run on an environment.

    Conditional targets (E-M / device-matrix) are excluded and reported as
    skipped with a reason, never silently dropped. Assertion-level
    conditionality (a target with one E-D-measurable and one E-M-only
    assertion, e.g. NFR-FLEX-001) is handled in `evaluate`.
    """
    selected, skipped = {}, {}
    for tid, t in doc["targets"].items():
        scen = t.get("scenario")
        scen_list = scen if isinstance(scen, list) else [scen]
        if scenario not in scen_list:
            continue
        envs = t.get("environment")
        env_list = envs if isinstance(envs, list) else [envs]
        target_level_emc = (t.get("e_m_conditional")
                            and not any(a.get("e_m_conditional")
                                        for a in t.get("assertions", [])))
        if target_level_emc and environment != "E-M" and not include_conditional:
            skipped[tid] = "e-m-conditional: requires an E-M-class host (TensorFlow)"
            continue
        if t.get("matrix_conditional"):
            skipped[tid] = "matrix-conditional: requires the Phase 6 device matrix"
            continue
        if environment not in env_list and env_list != ["n/a"]:
            skipped[tid] = f"environment {environment} not in {env_list}"
            continue
        selected[tid] = t
    return selected, skipped


def evaluate(doc, scenario, measured, environment="E-D", context=None):
    """Evaluate every applicable assertion against `measured`.

    `measured` maps metric name -> {statistic name -> value}. `context`
    (model, sessions from the actual run) guards against evaluating a
    target against data from the wrong model or concurrency — mismatches
    are reported, not silently compared. Metrics the run did not produce
    yield 'not-measured' outcomes rather than passes.
    """
    context = context or {}
    selected, skipped = targets_for_scenario(doc, scenario, environment)
    outcomes = []
    for tid, t in selected.items():
        t_model = t.get("model")
        if (t_model and context.get("model")
                and context["model"] != t_model):
            outcomes.append({
                "id": tid, "outcome": "condition-mismatch",
                "reason": f"target specifies model={t_model}, run used "
                          f"{context['model']}"})
            continue
        t_sessions = t.get("sessions")
        if (isinstance(t_sessions, int) and context.get("sessions")
                and context["sessions"] != t_sessions):
            outcomes.append({
                "id": tid, "outcome": "condition-mismatch",
                "reason": f"target specifies sessions={t_sessions}, run "
                          f"used {context['sessions']}"})
            continue
        for a in t.get("assertions", []):
            if a.get("e_m_conditional") and environment != "E-M":
                outcomes.append({
                    "id": tid, "metric": a["metric"],
                    "statistic": a["statistic"], "outcome": "skipped",
                    "reason": "assertion is e-m-conditional"})
                continue
            if a.get("discharge") == "inspection":
                outcomes.append({
                    "id": tid, "metric": a["metric"],
                    "statistic": a["statistic"],
                    "outcome": "inspection",
                    "reason": "discharged by inspection per the YAML "
                              "(no fabricated measured count)"})
                continue
            metric, stat = a["metric"], a["statistic"]
            got = measured.get(metric, {}).get(stat)
            if got is None:
                outcomes.append({
                    "id": tid, "metric": metric, "statistic": stat,
                    "op": a["op"], "target": a["value"], "unit": a["unit"],
                    "measured": None, "outcome": "not-measured",
                })
                continue
            ok = _OPS[a["op"]](got, a["value"])
            outcomes.append({
                "id": tid, "metric": metric, "statistic": stat,
                "op": a["op"], "target": a["value"], "unit": a["unit"],
                "measured": got, "outcome": "pass" if ok else "FAIL",
            })
        for r in t.get("report_metrics", []):
            got = measured.get(r["metric"], {})
            outcomes.append({
                "id": tid, "metric": r["metric"], "statistic": "report",
                "op": "report", "target": None, "unit": None,
                "measured": got or None, "outcome": "characterisation",
            })
    for tid, reason in skipped.items():
        outcomes.append({"id": tid, "outcome": "skipped", "reason": reason})
    return outcomes


def coverage_gaps(doc, runnable_scenarios):
    """Targets whose scenario no run.py scenario implements: these must
    surface in every summary as suite-coverage gaps, not vanish."""
    gaps = []
    for tid, t in doc["targets"].items():
        scen = t.get("scenario")
        scen_list = scen if isinstance(scen, list) else [scen]
        if not any(s in runnable_scenarios for s in scen_list):
            gaps.append({
                "id": tid, "outcome": "not-evaluable-by-this-suite",
                "reason": f"scenario(s) {scen_list} are outside the load "
                          f"suite (unit/inspection/analysis/mobile)"})
    return gaps
