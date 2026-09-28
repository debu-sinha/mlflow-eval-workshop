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


def _points(value):
    return f"{value * 100:+.0f} pts" if _number(value) else "Unavailable"


def _p_value(value):
    if not _number(value):
        return "Unavailable"
    return "p < 0.001" if value < 0.001 else f"p = {value:.3f}" if value < 0.1 else f"p = {value:.2f}"


def _policy(row):
    for span in row.get("spans", []):
        if span.get("name") == "retrieve_refund_policy":
            return json.dumps(span.get("outputs"), indent=2, ensure_ascii=False)
    return "No retrieval span was recorded."


def _response(row):
    output = row.get("output")
    return _text(output) if isinstance(output, str) and output.strip() else "No response recorded."


def _case_name(key):
    return key.replace("_", " ")


def _count(number):
    words = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve")
    return words[number] if isinstance(number, int) and 0 <= number < len(words) else str(number)


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


def paired_comparison(summary: dict, before: str, after: str) -> dict | None:
    """Recompute the paired evidence from the recorded rows, or None when incomplete."""
    from eval_gate import paired_evidence

    _, by_id, complete = _evidence(summary)
    if not complete:
        return None
    first = {key: _score(row) for key, row in by_id[before].items()}
    second = {key: _score(row) for key, row in by_id[after].items()}
    return paired_evidence(first, second)


def score_breakdown(summary: dict) -> list[dict]:
    """Count recorded passes without treating missing evidence as a success."""
    variants, _, complete = _evidence(summary)
    result = []
    for name in ("baseline", "candidate", "repaired"):
        rows = variants.get(name, {}).get("rows", [])
        policy_complete = complete and all(_number(row.get("scores", {}).get("policy_decision")) for row in rows)
        result.append({
            "version": name,
            "cases": len(rows),
            "correct_eligibility": sum(row["scores"]["policy_decision"] == 1 for row in rows) if policy_complete else None,
            "combined_passes": sum(_score(row) == 1 for row in rows) if complete else None,
        })
    return result


def judge_review_cases(summary: dict) -> list[dict]:
    """Find cases needing semantic review. A disagreement alone is not a judge error."""
    variants, _, complete = _evidence(summary)
    if not complete:
        return []
    return [dict(row, version=name)
            for name in ("baseline", "candidate", "repaired")
            for row in variants.get(name, {}).get("rows", [])
            if row.get("scores", {}).get("policy_decision") == 1
            and row.get("scores", {}).get("policy_judge") == 0]


def _questions(summary, count):
    """The three questions every release decision has to answer."""
    repair = paired_comparison(summary, "candidate", "repaired")
    release = paired_comparison(summary, "baseline", "repaired")
    counts = {item["version"]: item for item in score_breakdown(summary)}
    if repair is None or release is None:
        unavailable = "The comparison is incomplete, so this question cannot be answered from this run."
        answers = [("Unavailable", unavailable)] * 3
    else:
        before, after = counts["candidate"], counts["repaired"]
        better = (f"Cases passing every required check went from {_percent(repair['baseline_mean'])} "
                  f"to {_percent(repair['candidate_mean'])}. Correct eligibility went from "
                  f"{before['correct_eligibility']}/{count} to {after['correct_eligibility']}/{count}.")
        names = sorted(set(repair["regressed"]) | set(release["regressed"]))
        regressed_headline = "None" if not names else f"{len(names)} case" + ("" if len(names) == 1 else "s")
        regressed_detail = (f"{len(repair['regressed'])} of {count} against the stale candidate. "
                            f"{len(release['regressed'])} of {count} against the current baseline.")
        if names:
            regressed_detail += " Read " + ", ".join(_case_name(name) for name in names) + " below."
        low, high = repair["interval"]
        real = (f"{len(repair['improved'])} improved and {len(repair['regressed'])} regressed "
                f"({repair['test']} test). 95% bootstrap interval for the change: {_points(low)} to {_points(high)}. ")
        real += ("A change this large is unlikely to be noise. Ten cases still cannot rule out a small regression."
                 if repair["p_value"] < 0.05 and repair["mean_change"] > 0 else
                 f"{_count(count).capitalize()} cases cannot separate this change from noise. Add reviewed cases before claiming it.")
        answers = [(_points(repair["mean_change"]), better), (regressed_headline, regressed_detail),
                   (_p_value(repair["p_value"]), real)]
    titles = ("Did it get better?", "Which cases regressed?", "Is it real, or noise?")
    cards = "".join(
        f'<article><div class="question"><span>{number}</span>{_text(title)}</div>'
        f'<strong>{_text(headline)}</strong><p>{_text(detail)}</p></article>'
        for number, title, (headline, detail) in zip((1, 2, 3), titles, answers)
    )
    return f'<section class="questions" aria-label="Three release questions">{cards}</section>'


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
    count = len(by_id.get("baseline", {}))
    paired = sorted(by_id.get("baseline", {}), key=lambda key: (key != "day_45_opening", key))
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
            "No significant regression detected": "Every release rule passed for this test set.",
            "Candidate or baseline is below the absolute quality floor.": "The score is below the quality floor.",
        }.get(reason, reason)
        if decision == "INCOMPLETE":
            reason = "Responses or scores are missing, failed, or inconsistent."
        mean = means.get(name)
        width = min(100, max(0, mean * 100)) if _number(mean) else 0
        cards.append(f'''<article class="release {tone}">
          <div class="eyebrow">{label}</div><div class="verdict">{decision}</div>
          <div class="score"><strong>{_percent(mean)}</strong><span>of cases pass every required check</span></div>
          <div class="track" role="img" aria-label="Cases passing every required check: {_percent(mean)}"><div style="width:{width}%"></div></div>
          <p>{_text(reason)}</p>
        </article>''')
    breakdown_rows = []
    for item in score_breakdown(summary):
        counts = [f'{item[key]}/{item["cases"]}' if item[key] is not None else "Unavailable"
                  for key in ("correct_eligibility", "combined_passes")]
        breakdown_rows.append(f'<tr><th scope="row">{_text(item["version"].title())}</th><td>{counts[0]}</td><td>{counts[1]}</td></tr>')
    breakdown = f'''<section class="breakdown" aria-label="What the scores measure">
      <h2>What actually improved?</h2><p>Correct eligibility counts answers with the right refund, credit, or review label.
      Every required check also needs the deterministic checks and the policy judge to pass.
      A correct label can still come with a wrong explanation.</p>
      <div class="table-wrap"><table><thead><tr><th>Version</th><th>Correct eligibility</th><th>Every required check</th></tr></thead>
      <tbody>{''.join(breakdown_rows)}</tbody></table></div>
      <p class="note">Each case scores the lower of its deterministic stack and its policy judge. Read the answers before you interpret a change in score.</p></section>'''
    review_rows = judge_review_cases(summary)
    review_items = []
    for row in review_rows:
        rationale = next((item.get("rationale") for item in row.get("assessments", []) if item.get("name") == "policy_judge"), "No rationale recorded.")
        review_items.append(f'''<details class="case"><summary>{_text(row['version'].title())} · {_text(_case_name(row['case_id']))}</summary>
          <p>Correct eligibility: 1 · Policy judge: 0 · Deterministic stack: {_text(row['scores'].get('deterministic_stack', 'Unavailable'))}</p>
          <h4>Actual answer</h4><p>{_response(row)}</p><h4>The judge's explanation</h4><p>{_text(rationale)}</p>
          <p class="mono">Trace: {_text(row.get('trace_id', 'Unavailable'))}</p></details>''')
    review_count = str(len(review_rows)) if complete else "unavailable"
    review = f'''<details class="evidence"><summary>Review the judge: {review_count} correct labels with rejected replies</summary>
      <p>A correct label with a rejected reply can mean a wrong policy explanation or a judge mistake.
      Read the exact claim and the rubric together. The rubric permits customer next steps. It rejects claims or promises that the assistant carries out a transaction.
      Passing the eight rubric controls does not settle these individual cases.</p>
      {''.join(review_items) if complete else '<p>The comparison is incomplete. Check its evidence first.</p>'}
      {'<p>No case matches this pattern in this complete run. That does not establish general judge accuracy.</p>' if complete and not review_rows else ''}
      </details>'''
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
        case_rows.append(f'''<tr><th scope="row">{_text(_case_name(key))}</th>{cells}<td>{movement}</td></tr>
        <tr><td colspan="5"><details class="case"><summary>Read the answers and scoring explanations</summary>{''.join(details)}</details></td></tr>''')
    evidence = f'''<details class="evidence"><summary>What changed in the answer's source?</summary>
      <p>{_text(change)}</p><div class="answers"><div><h3>Before: retrieved policy</h3><pre>{_text(_policy(opening_before))}</pre></div>
      <div><h3>After: retrieved policy</h3><pre>{_text(_policy(opening_after))}</pre></div></div></details>
      <details class="evidence"><summary>Inspect every case, including judge disagreements</summary>
      <p>Each case scores the lower of its deterministic stack and its policy judge. A 0 can reflect a judge disagreement, so read the explanation and the answer together.</p>
      <div class="table-wrap"><table><thead><tr><th>Case</th><th>Baseline</th><th>Before</th><th>After</th><th>Change</th></tr></thead><tbody>{''.join(case_rows)}</tbody></table></div></details>'''
    run_rows = "".join(f'<tr><th>{_text(name)}</th><td class="mono">{_text(item.get("run_id", "Unavailable"))}</td></tr>' for name, item in variants.items())
    repaired_gate = gates.get("repaired", {})
    release = paired_comparison(summary, "baseline", "repaired")
    release_line = ("Against the current baseline, the repaired assistant improved "
                    f"{len(release['improved'])} and regressed {len(release['regressed'])} of {count} cases "
                    f"({release['test']} test, {_p_value(release['p_value'])}). " if release else "")
    provenance = f'''<details class="evidence"><summary>Release rules and MLflow run details</summary>
      <p>Every case needs a successful answer and every required score. Every deterministic check must pass.
      Both the baseline and the evaluated version need at least {_percent(repaired_gate.get("quality_floor"))} of cases passing,
      and no more than {_percent(repaired_gate.get("max_regression_rate"))} of cases may regress against the baseline.
      A significant paired loss also blocks the release.</p>
      <p>{_text(release_line)}With {count} cases, a significance test needs at least 6 cases moving in the same direction before it can call a change significant. The regression limit and mandatory checks catch smaller problems.</p>
      <p>Same dataset: {_text(comparison.get("same_dataset", "Unavailable"))}. Same scorers: {_text(comparison.get("same_scorers", "Unavailable"))}.
      Judge controls passed: {_text(summary.get("judge_validation", {}).get("passed", "Unavailable"))}.</p>
      <p>Application: {_text(summary.get("application_model", "Unavailable"))}<br>Judge: {_text(summary.get("judge_model", "Unavailable"))}</p>
      <p>Open experiment <span class="mono">{_text(summary.get("experiment_id", "Unavailable"))}</span> in MLflow to inspect these runs and their traces.</p>
      <div class="table-wrap"><table><tbody>{run_rows}<tr><th>Release gate</th><td class="mono">{_text(summary.get("gate_run_id", "Unavailable"))}</td></tr></tbody></table></div>
      <p class="mono">Dataset: {_text(summary.get("dataset_digest", "Unavailable"))}<br>Source commit: {_text(summary.get("git_commit", "Unavailable"))}</p></details>'''
    error = summary.get("error") or " ".join(summary.get("preflight", {}).get("reasons", []))
    warning = "" if observed else f'<p class="notice">This run did not establish the expected improvement and release outcome. Review missing evidence or failed checks before making a release decision. {_text(error or "")}</p>'
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>Northstar Shop | Release decision</title><style>{_CSS}</style></head><body><main class="report">
    <header><div class="brand"><span class="mark">N</span> NORTHSTAR SHOP <span class="divider">/</span> RELEASE REVIEW</div><span class="recorded">{status}</span></header>
    <section class="hero"><div class="eyebrow">Support assistant · refund policy</div><h1>Would you ship<br>this assistant?</h1>
    <p>One retrieval fix. The same {_count(count) if count else "named"} customer cases. Three questions every release has to answer.</p></section>
    {warning}<section class="releases" aria-label="Release decisions">{''.join(cards)}</section>{_questions(summary, count)}{answers}{breakdown}
    <section class="drilldown"><h2>Follow the evidence</h2><p>Start with the result. Open the source, then the scores and the release rules.</p>{review}{evidence}{provenance}</section>
    <footer>ODSC AI West 2026 · Debu Sinha · {_text(summary.get('provider', 'Unknown provider'))}<br>
    Recorded {_text(summary.get('utc_timestamp', 'time unavailable'))}. Reopening this report makes no model calls.<br>
    Fictional customer cases. Responses and scores come from the recorded run, and results apply to this teaching dataset.</footer>
    </main></body></html>'''


def write_release_report(summary: dict) -> Path:
    """Save beside the run summary so a completed run can open instantly later."""
    if not summary.get("summary_path"):
        raise ValueError("Run checkpoint 4 first to create a saved summary.")
    path = Path(summary["summary_path"]).resolve().with_name("release-report.html")
    path.write_text(render_release_report(summary), encoding="utf-8")
    return path


_CSS = """
*{box-sizing:border-box}html{color-scheme:dark}body{margin:0;background:#0c1424;color:#f4f7fc;font:17px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
.report{max-width:1180px;margin:auto;padding:36px 44px}header{display:flex;align-items:center;justify-content:space-between;gap:20px;padding-bottom:28px;border-bottom:1px solid #2b3850}
.brand{font-size:13px;font-weight:700;letter-spacing:1.4px;display:flex;align-items:center;gap:12px}.mark{display:inline-grid;place-items:center;width:30px;height:30px;background:#b7f3e4;border-radius:8px;color:#102a2b;font-size:20px}.divider{color:#8a9ab3}.recorded{color:#c7d2e4;font-size:14px}
.hero{padding:34px 0 26px}.eyebrow{text-transform:uppercase;letter-spacing:2px;font-size:13px;font-weight:700;color:#c3d1e6}h1{font-size:clamp(40px,5.2vw,68px);letter-spacing:-2px;line-height:1.04;margin:14px 0 18px}h2{font-size:26px;line-height:1.3;letter-spacing:-.5px}h3{font-size:15px;text-transform:uppercase;letter-spacing:1px;color:#c6d7ee}p{color:#ccd6e6}.hero p{max-width:720px;margin:0;font-size:20px}.releases,.answers{display:grid;grid-template-columns:1fr 1fr;gap:22px}
.release{padding:26px 28px;border:1px solid #4b3a4a;border-radius:16px;background:#221e2c;border-top:4px solid #ff998e}.release.ship{background:#132e30;border-color:#366b63;border-top-color:#8de6c8}.release.pending{background:#302c20;border-color:#8f7847}.verdict{font-size:50px;line-height:1.1;font-weight:800;letter-spacing:2px;margin:14px 0;color:#ffafa3}.ship .verdict{color:#9defd0}.pending .verdict{color:#f2d293;font-size:32px}.score{display:flex;align-items:baseline;gap:12px;flex-wrap:wrap}.score strong{font-size:32px}.score span{font-size:15px;color:#d2dbe8}.track{height:8px;border-radius:8px;background:#ffffff1c;margin-top:10px;overflow:hidden}.track div{height:100%;background:#f1a297}.ship .track div{background:#91e5c8}.release p{font-size:15px;margin:15px 0 0;min-height:20px}
.questions{display:grid;grid-template-columns:repeat(3,1fr);gap:18px;padding:26px 0 30px}.questions article{padding:22px 24px;border:1px solid #33445e;border-radius:16px;background:#131d30}.question{display:flex;align-items:center;gap:10px;font-size:15px;font-weight:700;color:#dbe6f5}.question span{display:inline-grid;place-items:center;width:26px;height:26px;border-radius:50%;background:#26354f;color:#b7f3e4;font-size:13px}.questions strong{display:block;font-size:34px;font-weight:750;margin:14px 0 8px;letter-spacing:-.5px}.questions p{font-size:15px;margin:0;line-height:1.5}
.customer{padding:26px 28px;background:#172337;border:1px solid #33445e;border-radius:16px}.customer h2{margin:10px 0 18px;font-size:24px}.answers p{font-size:16px;white-space:pre-wrap;margin-bottom:0}.answers article+article{border-left:1px solid #394d67;padding-left:22px}.answers h3{margin:0 0 8px}.breakdown .note{font-size:15px}.drilldown{padding-top:28px}.drilldown>p{font-size:15px;margin-top:-6px}.evidence{border-top:1px solid #35435a;padding:17px 0}.evidence>summary{font-weight:650;font-size:18px;cursor:pointer}.evidence[open]>summary{color:#9defd0;margin-bottom:18px}summary:focus-visible{outline:2px solid #9defd0;outline-offset:5px}.case{padding:6px 0;font-size:15px}.case summary{cursor:pointer;color:#bfe3d9}.case p,.case li{color:#d0d9e8}h4{font-size:16px;margin-bottom:6px}pre{white-space:pre-wrap;overflow-wrap:anywhere;color:#d7e2f2;background:#111c2d;padding:16px;border-radius:10px;font-size:14px}.mono{font-family:ui-monospace,Consolas,monospace;font-size:13px;overflow-wrap:anywhere}.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:15px;text-align:left}th,td{border-bottom:1px solid #304058;padding:12px 10px;vertical-align:top}th{font-weight:600}thead{color:#c3d1e6}.notice{border:1px solid #ab8a4a;background:#332b1d;padding:16px;border-radius:10px;color:#f8dca7}footer{font-size:13px;color:#a9b8cf;border-top:1px solid #35435a;margin-top:20px;padding:22px 0 8px;line-height:1.8}
@media(max-width:900px){.questions{grid-template-columns:1fr}}
@media(max-width:650px){.report{padding:22px 16px}header{align-items:flex-start;flex-direction:column;gap:12px;padding-bottom:22px}.brand{font-size:11px;gap:8px}.hero{padding-top:25px}.releases,.answers{grid-template-columns:1fr}.release{padding:22px}.customer{padding:22px}.answers article+article{border-left:0;border-top:1px solid #394d67;padding:18px 0 0}h1{letter-spacing:-1px}}
@media print{body{background:white;color:#152136}.report{max-width:none;padding:0}.recorded,p,.eyebrow,footer,.question{color:#394c66}.release,.release.ship,.customer,.questions article{background:#f5f8fa;color:#152136}.verdict{color:#a52a21}.ship .verdict{color:#14644a}.score span,.release p,.answers p{color:#344961}.release,.customer,.questions article{break-inside:avoid}}
"""
