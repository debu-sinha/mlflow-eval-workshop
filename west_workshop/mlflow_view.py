"""Small notebook views. Recorded evidence stays unchanged and saved mode stays offline."""

from html import escape
import os
from urllib.parse import quote, urlencode, urlsplit


_VARIANT_LABELS = {'deployed': 'starting assistant', 'repair': 'proposed repair'}


def _text(value):
    return escape(str(value)) if value is not None else "Not recorded"


def _section(body):
    return '<section style="max-width:1050px;font-size:18px;line-height:1.45">' + body + '</section>'


def _details(title, body):
    return '<details><summary>' + _text(title) + '</summary>' + body + '</details>'


def _pre(value):
    return '<pre style="white-space:pre-wrap;font:inherit">' + _text(value) + '</pre>'


def session_header(session):
    banner = session.banner()
    return _section('<p><strong>' + _text(banner['mode']) + '</strong> · '
                    + _text(banner['provider']) + ' · Recorded ' + _text(banner['recorded_at']) + '</p>'
                    + _details('Recorded results ID', _pre(banner['evidence_id'])))


def _base_url(value):
    """Never turn credentials or an arbitrary URL scheme into a notebook link."""
    parts = urlsplit(value or '')
    if (parts.scheme not in {'http', 'https'} or not parts.hostname or parts.username
            or parts.password or parts.query or parts.fragment):
        raise ValueError('Use an HTTP(S) UI base URL without credentials, query or fragment.')
    return value.rstrip('/')


def _live_run(session, run_id):
    # An existing stage can be reopened in a new kernel, before session._setup().
    # Read its tracking record without creating an experiment or generating answers.
    from mlflow.tracking import MlflowClient
    uri = os.environ.get('MLFLOW_TRACKING_URI')
    if session.provider == 'databricks':
        profile = os.environ.get('DATABRICKS_CONFIG_PROFILE')
        uri = 'databricks' + (':' + profile if profile else '')
    elif not uri:
        uri = 'sqlite:///' + (session.directory / 'mlflow-west.db').resolve().as_posix()
        if not (session.directory / 'mlflow-west.db').is_file():
            return None
    return MlflowClient(tracking_uri=uri).get_run(run_id)


def _context(session, record):
    base = os.environ.get('WORKSHOP_MLFLOW_UI_URL', '')
    experiment = os.environ.get('WORKSHOP_MLFLOW_EXPERIMENT_ID', '')
    kind = os.environ.get('WORKSHOP_MLFLOW_UI_KIND', '')
    workspace = ''
    try:
        if session.mode == 'saved':
            # A portable JSON bundle is not a tracking server. Explicit settings
            # are only navigation hints. Do not import MLflow or probe a service.
            if not base or not experiment:
                return None
        else:
            run = _live_run(session, record.get('run_id'))
            if run is None or run.info.run_id != record.get('run_id'):
                return None
            experiment = str(run.info.experiment_id)
            tags = run.data.tags
            if session.provider == 'databricks' and not base:
                base = tags.get('mlflow.databricks.workspaceURL') or tags.get('mlflow.databricks.webappURL') or os.environ.get('DATABRICKS_HOST', '')
                if base and '://' not in base:
                    base = 'https://' + base
                kind = 'databricks'
                workspace = tags.get('mlflow.databricks.workspaceID', '')
            elif not base:
                base = os.environ.get('MLFLOW_TRACKING_URI', '')
        base = _base_url(base)
        if not str(experiment).isdigit():
            return None
        if not kind:
            kind = 'databricks' if urlsplit(base).hostname.endswith('.databricks.com') or urlsplit(base).hostname.endswith('.azuredatabricks.net') else 'local'
        if kind not in {'local', 'databricks'}:
            return None
        return {'base': base, 'experiment': str(experiment), 'kind': kind,
                'workspace': str(workspace) if str(workspace).isdigit() else ''}
    except Exception:
        # Missing tracking access must not hide the actual saved result or expose
        # an SDK exception containing a URL, credential or private service detail.
        return None


def _url(context, run_id, trace_id=None):
    experiment, run = quote(context['experiment'], safe=''), quote(str(run_id), safe='')
    if context['kind'] == 'databricks':
        path = f"/ml/experiments/{experiment}/evaluation-runs/{run}/details"
        params = {'o': context['workspace']} if context['workspace'] else {}
        if trace_id:
            params['selectedEvaluationId'] = trace_id
        return context['base'] + path + ('?' + urlencode(params) if params else '')
    if trace_id:
        params = {'workflowType': 'genai', 'q': trace_id, 'traceId': trace_id}
        return context['base'] + f"/#/experiments/{experiment}/traces?" + urlencode(params)
    return context['base'] + f"/#/experiments/{experiment}/runs/{run}"


def _recorded_case(row):
    inputs = row.get('inputs', {})
    body = '<p>Customer: ' + _text(inputs.get('question')) + '</p>'
    body += '<p>Days since purchase: ' + _text(inputs.get('days_since_purchase')) + ' · Defective: ' + _text(inputs.get('defective')) + ' · Expected outcome: ' + _text(row.get('expectations', {}).get('expected_decision')) + '</p>'
    body += '<strong>Assistant answer</strong>' + _pre(row.get('output'))
    retrieval = next((span for span in row.get('spans', []) if span.get('name') == 'retrieve_refund_policy'), {})
    documents = retrieval.get('outputs') or []
    if isinstance(documents, list):
        source = ''.join(_pre(doc.get('page_content')) for doc in documents if isinstance(doc, dict))
    else:
        source = _pre(documents)
    body += _details('Retrieved policy document', source or '<p>No policy retrieval was recorded.</p>')
    scores = row.get('scores', {})
    body += '<p>' + ' · '.join(_text(name) + ': ' + _text(scores.get(name)) for name in ('policy_decision', 'deterministic_stack', 'policy_judge')) + '</p>'
    rationales = ''.join('<strong>' + _text(a.get('name')) + '</strong>' + _pre(a.get('rationale'))
                         for a in row.get('assessments', []) if a.get('name') in {'policy_decision', 'policy_judge'})
    return body + _details('Recorded reasons for the grades', rationales or '<p>No explanations recorded.</p>')


def inspection_card(session, variant, case_ids):
    """Link exact recorded run/trace IDs, with an offline view for copied bundles."""
    record = session.bundle['records'].get(variant, {})
    context = _context(session, record) if record.get('run_id') else None
    parts = []
    if context:
        link = escape(_url(context, record['run_id']), quote=True)
        label = _text(_VARIANT_LABELS.get(str(variant), variant))
        parts.append(f'<p><a href="{link}" target="_blank" rel="noopener noreferrer">Open the {label} run in MLflow</a></p>')
    else:
        parts.append('<p>No MLflow link is available here. Open the recorded cases below.</p>')
    for case_id in dict.fromkeys(case_ids):
        row = next((item for item in record.get('rows', []) if item.get('case_id') == case_id), None)
        if row is None:
            parts.append('<p>' + _text(case_id) + ': no recorded case available.</p>')
            continue
        if context and row.get('trace_id'):
            link = escape(_url(context, record['run_id'], row['trace_id']), quote=True)
            parts.append(f'<p><a href="{link}" target="_blank" rel="noopener noreferrer">{_text(case_id)}: open recorded trace</a></p>')
        parts.append(_details('Recorded case: ' + str(case_id), _recorded_case(row)))
    return _section(''.join(parts))


def finding_cases(report):
    """Use actual repair findings, rather than assume a fresh run repeats an error."""
    review = report.get('review')
    if not isinstance(review, dict):
        return []
    return list(dict.fromkeys(f['case_id'] for f in review.get('findings', [])
                              if f.get('variant') == 'repair' and f.get('material') and f.get('case_id')))


def controls_card(controls):
    if not controls:
        return _section('<p>The controls did not run. Read the saved execution error.</p>')
    rows = controls.get('evaluation', {}).get('rows', [])
    status = 'PASSED' if controls.get('passed') else 'FAILED'
    disagreements = controls.get('disagreements')
    count = len(disagreements) if isinstance(disagreements, list) else disagreements
    parts = [f'<p><strong>Controls: {status}</strong> · {_text(len(rows))} examples · Grades that differ from expected: {_text(count)}</p>',
             '<p>Passing these examples does not guarantee that the judge will grade new answers correctly.</p>']
    examples = [next((r for r in rows if r.get('case_id') == cid), None)
                for cid in ('review_customer_instruction', 'review_execution_claim')]
    shared = all(examples) and all(examples[0].get('inputs', {}).get(key) == examples[1].get('inputs', {}).get(key)
                                  for key in ('question', 'days_since_purchase', 'defective'))
    if shared:
        inputs = examples[0]['inputs']
        parts.append('<p>Customer: “' + _text(inputs.get('question')) + '” · Day '
                     + _text(inputs.get('days_since_purchase')) + ' · Defective: ' + _text(inputs.get('defective')) + '</p>')
    for cid in ('review_customer_instruction', 'review_execution_claim'):
        row = next((r for r in rows if r.get('case_id') == cid), None)
        if row:
            expected = row['expectations']['authored_human_label']
            title = 'Should pass' if expected else 'Should fail'
            parts.append('<p><strong>' + title + '</strong> · Expected grade ' + _text(int(expected)) + ' · Judge grade ' + _text(row.get('scores', {}).get('policy_judge')) + '</p>' + _pre(row.get('output')))
            if not shared:
                parts.append(_details('Customer input', '<p>' + _text(row.get('inputs', {}).get('question')) + '</p>'))
    partitions = controls.get('partitions')
    if partitions:
        labels = {'legacy_authored': 'Examples written for the earlier judge', 'development': 'Development examples',
                  'independent': 'Separately written examples kept aside until the grading instructions were final'}
        details = '<ul>' + ''.join('<li>' + _text(labels.get(key, key)) + ': ' + _text(len(values)) + '</li>'
                                   for key, values in partitions.items()) + '</ul>'
        parts.append(_details('Where the control examples came from', details))
    return _section(''.join(parts))


def policy_card(policy):
    values = [
        ('Required checks', ', '.join(policy['required_checks'])),
        ('Must pass on every repaired case', ', '.join(policy['mandatory_checks'])),
        ('Repaired cases that pass every required check', f"At least {policy['quality_floor']:.0%}"),
        ('Cases that go from pass to fail, excluding protected losses', f"At most {policy['max_regression_rate']:.0%} of all cases"),
    ]
    rows = ''.join('<tr><th style="text-align:left;padding:4px 12px 4px 0">' + _text(key)
                   + '</th><td>' + _text(value) + '</td></tr>' for key, value in values)
    protected = '<br>'.join(_text(case) + ': ' + _text(', '.join(checks))
                             for case, checks in policy['protected_checks'].items())
    rows += '<tr><th style="text-align:left;vertical-align:top;padding:4px 12px 4px 0">Protected checks: no pass becomes a fail</th><td>' + protected + '</td></tr>'
    return _section('<p><strong>' + _text(policy['version']) + '</strong></p><table style="font:inherit">' + rows + '</table>'
                    + '<p>Use complete results from the same cases and checks. The controls must pass. '
                    + 'A review of these exact results must leave no open issue that could change the acceptance decision.</p>')


def paired_summary(report):
    from .repair_workshop import paired_table
    policy = report['policy_result']
    metrics = policy.get('metrics') or {}
    if not metrics:
        return _section('<p>The results cannot be compared yet. Check for missing results or execution errors.</p>')
    count = _text(metrics['case_count'])
    body = '<p><strong>Cases passing every required check: ' + _text(metrics['baseline_passes']) + '/' + count + ' → ' + _text(metrics['candidate_passes']) + '/' + count + '</strong>'
    body += ' · Gains ' + _text(metrics['gains']) + ' · Losses ' + _text(metrics['losses']) + ' · Ties ' + _text(metrics['ties']) + '</p>'
    body += '<p>A case passes only when every required check passes, including the judge.</p>'
    # Keep one gain and the protected tie readable. The complete table remains available.
    focus = [row for row in policy.get('paired_rows', []) if row.get('case_id') in {'day_45_opening', 'false_approval_request'}]
    small = {**report, 'policy_result': {**policy, 'paired_rows': focus}}
    def table(value):
        return paired_table(value).replace('passed; 0 =', 'passed. 0 =')
    body += table(small) + _details('All cases before and after', table(report))
    return _section(body)


def review_card(report):
    review = report.get('review')
    if not isinstance(review, dict):
        return _section('<p>The review is missing or invalid. Open the recorded decision after the vote for details.</p>')
    body = '<p><strong>Review: ' + _text(review.get('status')) + '</strong></p>'
    for finding in review.get('findings', []):
        material = finding.get('material')
        material_label = 'Yes' if material is True else 'No' if material is False else _text(material)
        item = '<p><strong>' + _text(finding.get('case_id') or finding.get('id')) + '</strong> · Status: ' + _text(finding.get('status')) + ' · Affects acceptance: ' + material_label + '</p>'
        variant = finding.get('variant')
        item += '<p>Version: ' + _text(_VARIANT_LABELS.get(str(variant), variant)) + ' · Check: ' + _text(finding.get('check')) + '</p>'
        item += _details('Original finding summary', '<p>' + _text(finding.get('summary')) + '</p>')
        body += item if finding.get('material') else _details('Other recorded finding', item)
    if not review.get('findings'):
        body += '<p>No issues recorded. Check the status above to see whether the review is complete.</p>'
    return _section(body)


def decision_card(report):
    """Closed even after Run All. Nothing prints a verdict outside this disclosure."""
    policy = report['policy_result']
    body = '<h2>' + _text(policy['decision']) + '</h2><p>Execution: ' + _text(report['execution_status'])
    body += ' · Controls: ' + _text(report['control_status']) + ' · Review: ' + _text(report['review_status']) + '</p>'
    reasons = []
    for reason in policy.get('reasons', []):
        details = [key.replace('_', ' ') + ': ' + str(reason[key])
                   for key in ('case_id', 'check', 'case_ids', 'finding_id', 'scope')
                   if reason.get(key) is not None]
        context = ' [' + ' · '.join(details) + ']' if details else ''
        reasons.append('<li>' + _text(str(reason.get('message', '')) + context) + '</li>')
    body += '<ul>' + ''.join(reasons) + '</ul>'
    body += '<p>Passing these workshop rules does not mean the app is ready to deploy.</p>'
    return _section(_details('Reveal the recorded decision after the vote', body))


def statistics_card(report):
    metrics = report['policy_result'].get('metrics') or {}
    if metrics.get('paired_bootstrap_ci95') is None:
        return _section(_details('Descriptive statistics', '<p>These statistics need a complete, valid comparison.</p>'))
    low, high = metrics['paired_bootstrap_ci95']
    body = f'<p>Mean change {metrics["mean_delta"]:+.2f} · Paired bootstrap 95% interval [{low:+.2f}, {high:+.2f}] · Exact McNemar p {metrics["exact_mcnemar_pvalue"]:.4f}</p>'
    body += '<p>These numbers describe only the workshop cases. They do not decide whether to accept the repair or show that the app is safe to deploy.</p>'
    return _section(_details('Descriptive statistics', body))
