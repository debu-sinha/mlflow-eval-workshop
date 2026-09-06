"""A portable visual report built from a checkpoint 4 result, without model calls."""

from html import escape
import json
import math
from pathlib import Path


def _text(value):
    return escape(str(value), quote=True)


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _score(row):
    values = [row.get("scores", {}).get(key) for key in ("deterministic_stack", "policy_judge")]
    return min(values) if all(_number(value) for value in values) else None


def _percent(value):
    return f"{value:.0%}" if _number(value) else "Unavailable"


def _policy(row):
    for span in row.get("spans", []):
        if span.get("name") == "retrieve_refund_policy":
            return json.dumps(span.get("outputs"), indent=2, ensure_ascii=False)
    return "No retrieval span was recorded."


def _response(row):
    output = row.get("output")
    return _text(output) if isinstance(output, str) and output.strip() else "No response recorded."


def _evidence(summary):
    from .runtime import _complete

    evaluations = summary.get("evaluations", [])
    variants = dict(zip(("baseline", "candidate", "repaired"), evaluations))
    baseline_rows = variants.get("baseline", {}).get("rows", [])
    expected = [{"inputs": row["inputs"], "expectations": row["expectations"]} for row in baseline_rows]
    expected_by_id = {row["inputs"]["case_id"]: row for row in expected}
    required = [item["name"] for item in variants.get("baseline", {}).get("scorers", [])]
    complete = bool(expected) and len(evaluations) == 3 and bool(required)
    for evaluation in evaluations:
        rows = evaluation.get("rows", [])
        complete = complete and evaluation.get("complete") is True and _complete(rows, expected, required)
        complete = complete and evaluation.get("row_count") == len(rows)
        complete = complete and all(
            {"inputs": row["inputs"], "expectations": row["expectations"]} == expected_by_id.get(row["case_id"])
            for row in rows
        )
        complete = complete and evaluation.get("scorers") == variants["baseline"].get("scorers")
        complete = complete and evaluation.get("dataset_digest") == variants["baseline"].get("dataset_digest")
    complete = complete and {"deterministic_stack", "policy_judge"}.issubset(required)
    by_id = {name: {row["case_id"]: row for row in item.get("rows", [])} for name, item in variants.items()}
    return variants, by_id, complete


def _decision(gate, complete):
    if not complete or gate.get("complete") is not True:
        return "INCOMPLETE", "pending"
    if gate.get("passed") is True and gate.get("decision") == "ship":
        return "SHIP", "ship"
    if gate.get("passed") is False and gate.get("decision") == "block":
        return "BLOCK", "block"
    return "REVIEW", "pending"


def render_release_report(summary: dict) -> str:
    """Render recorded evidence. Missing or failed traces cannot produce a ship card."""
    if summary.get("checkpoint") != 4:
        raise ValueError("The release report requires a checkpoint 4 summary.")
    variants, by_id, complete = _evidence(summary)
    gates = summary.get("gates", {})
    comparison = summary.get("repair_comparison") or {}
    means = {
        name: sum(_score(row) for row in item["rows"]) / len(item["rows"]) if complete else None
        for name, item in variants.items()
    }
    before, after = means.get("candidate"), means.get("repaired")
    count = len(by_id.get("baseline", {}))
    paired = sorted(by_id.get("baseline", {}), key=lambda key: (key != "day_45_opening", key))
    improved = sum(_score(by_id["repaired"][key]) > _score(by_id["candidate"][key]) for key in paired) if complete else None
    regressed = sum(_score(by_id["repaired"][key]) < _score(by_id["candidate"][key]) for key in paired) if complete else None
    observed = complete and summary.get("status") == "passed" and summary.get("expected_outcome_observed") is True
    status = "Recorded run · comparison complete" if observed else "Run needs review · inspect the evidence"
    change = comparison.get("changed_component", "A complete comparison has not been recorded.")
    cards = []
    for name, label in (("candidate", "Before · stale policy"), ("repaired", "After · retrieval repaired")):
        gate = gates.get(name, {})
        decision, tone = _decision(gate, complete)
        reason = gate.get("reason", "The run did not record a release decision.")
        reason = {
            "A mandatory deterministic invariant failed.": "A required policy check failed.",
            "No significant regression detected": "All release checks passed for this test set.",
        }.get(reason, reason)
        if decision == "INCOMPLETE":
            reason = "Responses or scores are missing, failed, or inconsistent."
        mean = means.get(name)
        width = min(100, max(0, mean * 100)) if _number(mean) else 0
        cards.append(f'''<article class="release {tone}">
          <div class="eyebrow">{label}</div><div class="verdict">{decision}</div>
          <div class="score"><strong>{_percent(mean)}</strong><span>mean evaluation score</span></div>
          <div class="track" role="img" aria-label="Mean evaluation score: {_percent(mean)}"><div style="width:{width}%"></div></div>
          <p>{_text(reason)}</p>
        </article>''')
    delta = f"{(after - before) * 100:+.0f} pts" if complete else "Unavailable"
    stats = f'''<div class="stats">
      <div><strong>{delta}</strong><span>score change</span></div>
      <div><strong>{improved if complete else 'Unavailable'}</strong><span>cases improved</span></div>
      <div><strong>{regressed if complete else 'Unavailable'}</strong><span>cases regressed</span></div>
      <div><strong>{count if complete else 'Incomplete'}</strong><span>cases per version</span></div>
    </div>'''
    opening_before = by_id.get("candidate", {}).get("day_45_opening", {})
    opening_after = by_id.get("repaired", {}).get("day_45_opening", {})
    question = opening_before.get("inputs", {}).get("question", "The opening case was not recorded.")
    answers = f'''<section class="customer">
      <div class="eyebrow">The customer behind the score</div><h2>“{_text(question)}”</h2>
      <div class="answers"><article><h3>Before</h3><p>{_response(opening_before)}</p></article>
      <article><h3>After</h3><p>{_response(opening_after)}</p></article></div>
    </section>'''
    case_rows = []
    for key in paired:
        b = by_id.get("baseline", {}).get(key, {})
        c = by_id.get("candidate", {}).get(key, {})
        r = by_id.get("repaired", {}).get(key, {})
        c_score, r_score = _score(c), _score(r)
        movement = "Unavailable"
        if complete:
            movement = "Improved" if r_score > c_score else "Regressed" if r_score < c_score else "Unchanged"
        details = []
        for label, row in (("Baseline", b), ("Before", c), ("After", r)):
            rationales = "".join(f'<li><strong>{_text(a.get("name"))}</strong>: {_text(a.get("rationale"))}</li>' for a in row.get("assessments", []))
            details.append(f'<h4>{label}</h4><p>{_response(row)}</p><ul>{rationales}</ul><p class="mono">Trace: {_text(row.get("trace_id", "Unavailable"))}</p>')
        cells = "".join(f'<td>{_percent(_score(row)) if complete else "Unavailable"}</td>' for row in (b, c, r))
        case_rows.append(f'''<tr><th scope="row">{_text(key.replace('_', ' '))}</th>{cells}<td>{movement}</td></tr>
        <tr><td colspan="5"><details class="case"><summary>Read the answers and scoring explanations</summary>{''.join(details)}</details></td></tr>''')
    evidence = f'''<details class="evidence"><summary>What changed in the answer's source?</summary>
      <p>{_text(change)}</p><div class="answers"><div><h3>Before: retrieved policy</h3><pre>{_text(_policy(opening_before))}</pre></div>
      <div><h3>After: retrieved policy</h3><pre>{_text(_policy(opening_after))}</pre></div></div></details>
      <details class="evidence"><summary>Inspect every case, including judge disagreements</summary>
      <p>Each case uses the lower of its deterministic score and policy-judge score. The mean is shown above. A 0 can reflect a judge disagreement; read the explanation and the answer together.</p>
      <div class="table-wrap"><table><thead><tr><th>Case</th><th>Baseline</th><th>Before</th><th>After</th><th>Change</th></tr></thead><tbody>{''.join(case_rows)}</tbody></table></div></details>'''
    run_rows = "".join(f'<tr><th>{_text(name)}</th><td class="mono">{_text(item.get("run_id", "Unavailable"))}</td></tr>' for name, item in variants.items())
    floor = gates.get("repaired", {}).get("quality_floor")
    regression = gates.get("repaired", {}).get("max_regression_rate")
    provenance = f'''<details class="evidence"><summary>Release rules and MLflow run details</summary>
      <p>Quality floor: {_percent(floor)}. Maximum regression rate versus baseline: {_percent(regression)}. All mandatory checks and complete case coverage are required.</p>
      <p>Same dataset: {_text(comparison.get("same_dataset", "Unavailable"))}. Same scorers: {_text(comparison.get("same_scorers", "Unavailable"))}.
      Judge controls passed: {_text(summary.get("judge_validation", {}).get("passed", "Unavailable"))}.</p>
      <p>Application: {_text(summary.get("application_model", "Unavailable"))}<br>Judge: {_text(summary.get("judge_model", "Unavailable"))}</p>
      <p>Open experiment <span class="mono">{_text(summary.get("experiment_id", "Unavailable"))}</span> in MLflow to inspect these runs and their traces.</p>
      <div class="table-wrap"><table><tbody>{run_rows}<tr><th>Release gate</th><td class="mono">{_text(summary.get("gate_run_id", "Unavailable"))}</td></tr></tbody></table></div>
      <p class="mono">Dataset: {_text(summary.get("dataset_digest", "Unavailable"))}<br>Source commit: {_text(summary.get("git_commit", "Unavailable"))}</p></details>'''
    error = summary.get("error") or "; ".join(summary.get("preflight", {}).get("reasons", []))
    warning = "" if observed else f'<p class="notice">The expected improvement and release outcome were not fully established. Review missing evidence or failed checks before making a release decision. {_text(error or "")}</p>'
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>Northstar Shop | Release decision</title><style>{_CSS}</style></head><body><main class="report">
    <header><div class="brand"><span class="mark">N</span> NORTHSTAR SHOP <span class="divider">/</span> RELEASE REVIEW</div><span class="recorded">{status}</span></header>
    <section class="hero"><div class="eyebrow">Support assistant · refund policy</div><h1>Would you ship<br>this assistant?</h1>
    <p>One policy change. The same customer questions. The evidence for a release decision.</p></section>
    {warning}<section class="releases" aria-label="Release decisions">{''.join(cards)}</section>{stats}{answers}
    <section class="drilldown"><h2>Follow the evidence</h2><p>Start with the result. Open the source, then the scores and release rules.</p>{evidence}{provenance}</section>
    <footer>ODSC AI West 2026 · Debu Sinha · {_text(summary.get('provider', 'Unknown provider'))}<br>
    Recorded {_text(summary.get('utc_timestamp', 'time unavailable'))}. Reopening this report makes no model calls.<br>
    Fictional customer cases; responses and scores are from the recorded run. Results apply to this teaching dataset.</footer>
    </main></body></html>'''


def write_release_report(summary: dict) -> Path:
    """Save beside the run summary so a completed run can open instantly later."""
    if not summary.get("summary_path"):
        raise ValueError("Run checkpoint 4 first to create a saved summary.")
    path = Path(summary["summary_path"]).resolve().with_name("release-report.html")
    path.write_text(render_release_report(summary), encoding="utf-8")
    return path


_CSS = """
*{box-sizing:border-box}html{color-scheme:dark}body{margin:0;background:#0c1424;color:#f4f7fc;font:16px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
.report{max-width:1180px;margin:auto;padding:36px 44px}header{display:flex;align-items:center;justify-content:space-between;gap:20px;padding-bottom:32px;border-bottom:1px solid #2b3850}
.brand{font-size:12px;font-weight:700;letter-spacing:1.4px;display:flex;align-items:center;gap:12px}.mark{display:inline-grid;place-items:center;width:30px;height:30px;background:#b7f3e4;border-radius:8px;color:#102a2b;font-size:20px}.divider{color:#7385a0}.recorded{color:#b9c6dc;font-size:12px}
.hero{padding:34px 0 26px}.eyebrow{text-transform:uppercase;letter-spacing:2px;font-size:12px;font-weight:700;color:#bbcbe2}h1{font-size:clamp(38px,5vw,64px);letter-spacing:-2px;line-height:1.04;margin:14px 0 18px}h2{font-size:24px;line-height:1.3;letter-spacing:-.5px}h3{font-size:14px;text-transform:uppercase;letter-spacing:1px;color:#bdd1eb}p{color:#c0cce0}.hero p{max-width:660px;margin:0;font-size:18px}.releases,.answers{display:grid;grid-template-columns:1fr 1fr;gap:22px}
.release{padding:25px 28px;border:1px solid #4b3a4a;border-radius:16px;background:#221e2c;border-top:4px solid #ff998e}.release.ship{background:#132e30;border-color:#366b63;border-top-color:#8de6c8}.release.pending{background:#302c20;border-color:#8f7847}.verdict{font-size:46px;line-height:1.1;font-weight:800;letter-spacing:2px;margin:16px 0;color:#ffafa3}.ship .verdict{color:#9defd0}.pending .verdict{color:#f2d293;font-size:30px}.score{display:flex;align-items:baseline;gap:12px}.score strong{font-size:30px}.score span{font-size:14px;color:#c4cedd}.track{height:7px;border-radius:8px;background:#ffffff16;margin-top:10px;overflow:hidden}.track div{height:100%;background:#f1a297}.ship .track div{background:#91e5c8}.release p{font-size:13px;margin:15px 0 0;min-height:20px}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;padding:24px 0 30px}.stats div{padding:0 20px;border-left:1px solid #334159}.stats div:first-child{padding-left:0;border:0}.stats strong{display:block;font-size:28px;font-weight:650}.stats span{font-size:13px;color:#aebed5}.customer{padding:26px 28px;background:#172337;border:1px solid #33445e;border-radius:16px}.customer h2{margin:10px 0 18px;font-size:23px}.answers p{font-size:15px;white-space:pre-wrap;margin-bottom:0}.answers article+article{border-left:1px solid #394d67;padding-left:22px}.answers h3{margin:0 0 8px}.drilldown{padding-top:28px}.drilldown>p{font-size:14px;margin-top:-6px}.evidence{border-top:1px solid #35435a;padding:17px 0}.evidence>summary{font-weight:650;font-size:17px;cursor:pointer}.evidence[open]>summary{color:#9defd0;margin-bottom:18px}summary:focus-visible{outline:2px solid #9defd0;outline-offset:5px}.case{padding:6px 0;font-size:14px}.case summary{cursor:pointer;color:#b5dcd2}.case p,.case li{color:#c4cee0}h4{font-size:16px;margin-bottom:6px}pre{white-space:pre-wrap;overflow-wrap:anywhere;color:#cfdbed;background:#111c2d;padding:16px;border-radius:10px;font-size:13px}.mono{font-family:ui-monospace,Consolas,monospace;font-size:12px;overflow-wrap:anywhere}.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px;text-align:left}th,td{border-bottom:1px solid #304058;padding:12px 10px;vertical-align:top}th{font-weight:600}thead{color:#b8c7dc}.notice{border:1px solid #ab8a4a;background:#332b1d;padding:16px;border-radius:10px;color:#f8dca7}footer{font-size:12px;color:#91a5c3;border-top:1px solid #35435a;margin-top:20px;padding:22px 0 8px;line-height:1.8}
@media(max-width:650px){.report{padding:22px 18px}header{align-items:flex-start;flex-direction:column;gap:12px;padding-bottom:22px}.brand{font-size:10px;gap:8px}.hero{padding-top:25px}.releases,.answers{grid-template-columns:1fr}.release{padding:22px}.stats{grid-template-columns:repeat(2,1fr);gap:22px}.stats div:nth-child(3){border:0;padding-left:0}.customer{padding:22px}.answers article+article{border-left:0;border-top:1px solid #394d67;padding:18px 0 0}.score{flex-wrap:wrap}h1{letter-spacing:-1px}.stats strong{font-size:24px}}
@media print{body{background:white;color:#152136}.report{max-width:none;padding:0}.recorded,p,.eyebrow,footer{color:#394c66}.release,.release.ship,.customer{background:#f5f8fa;color:#152136}.verdict{color:#a52a21}.ship .verdict{color:#14644a}.score span,.release p,.answers p{color:#344961}.release,.customer{break-inside:avoid}}
"""
