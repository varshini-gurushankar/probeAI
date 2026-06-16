"""ProbeAI evaluation layer.

Two kinds of measurement, deliberately separated so a model never grades itself:

  * DETERMINISTIC ground-truth checks (heuristics.py, metrics.py, scenarios.py):
    the headline "scenario match-accuracy" compares the moderator's actual
    behavior to hand-labeled expectations — no LLM in that loop.

  * An ISOLATED LLM judge (judge.py): scoped to follow-up *relevance* only, on a
    different model from the moderator. Never used for the headline number.

Run the offline harness with:  python -m probeai.evals.run
"""
