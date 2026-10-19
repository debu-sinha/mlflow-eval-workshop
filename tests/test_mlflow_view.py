"""Navigation and notebook presentation must preserve the recorded evidence."""

import copy
from html import unescape
from html.parser import HTMLParser
import json
from pathlib import Path
import socket
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest

from west_workshop import mlflow_view as views
from west_workshop.repair_workshop import WorkshopSession


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def saved(monkeypatch):
    for name in ('WORKSHOP_MLFLOW_UI_URL', 'WORKSHOP_MLFLOW_EXPERIMENT_ID', 'WORKSHOP_MLFLOW_UI_KIND'):
        monkeypatch.delenv(name, raising=False)
    return WorkshopSession('saved', evidence_dir=ROOT / 'evidence/workshop/local/final')


class VisibleText(HTMLParser):
    """Text outside closed disclosures, without executing JavaScript."""
    def __init__(self):
        super().__init__()
        self.depth = 0
        self.text = []

    def handle_starttag(self, tag, attrs):
        if tag == 'details':
            assert 'open' not in dict(attrs)
            self.depth += 1

    def handle_endtag(self, tag):
        if tag == 'details':
            self.depth -= 1

    def handle_data(self, data):
        if not self.depth:
            self.text.append(data)


def test_saved_inspection_is_offline_and_has_exact_fallback(saved, monkeypatch):
    monkeypatch.setattr(socket, 'socket', lambda *a, **kw: pytest.fail('Saved view contacted a server'))
    monkeypatch.setattr(views, '_live_run', lambda *a: pytest.fail('Saved view read tracking'))
    before = copy.deepcopy(saved.bundle)
    card = views.inspection_card(saved, 'deployed', ['day_45_opening'])
    assert 'href=' not in card
    assert 'within 90 days' in card and 'Recorded assessment explanations' in card
    assert 'retrieve' in card.lower() and 'policy_decision: 0.0' in card
    assert saved.bundle == before


def test_saved_links_require_explicit_backend_and_keep_exact_trace(saved, monkeypatch):
    monkeypatch.setenv('WORKSHOP_MLFLOW_UI_URL', 'http://127.0.0.1:5000')
    assert views._context(saved, saved.recorded('deployed')) is None
    monkeypatch.setenv('WORKSHOP_MLFLOW_EXPERIMENT_ID', '42')
    record = saved.recorded('deployed')
    row = next(row for row in record['rows'] if row['case_id'] == 'day_45_opening')
    context = views._context(saved, record)
    url = views._url(context, record['run_id'], row['trace_id'])
    fragment = urlsplit(url).fragment
    assert fragment.startswith('/experiments/42/traces?')
    assert parse_qs(fragment.split('?', 1)[1]) == {
        'workflowType': ['genai'], 'q': [row['trace_id']], 'traceId': [row['trace_id']],
    }


@pytest.mark.parametrize('url', ['javascript:alert(1)', 'https://user:secret@example.test',
                                'https://example.test?token=secret', 'https://example.test/#token=secret'])
def test_unsafe_ui_url_falls_back_without_exposing_it(saved, monkeypatch, url):
    monkeypatch.setenv('WORKSHOP_MLFLOW_UI_URL', url)
    monkeypatch.setenv('WORKSHOP_MLFLOW_EXPERIMENT_ID', '42')
    card = views.inspection_card(saved, 'deployed', ['day_45_opening'])
    assert 'href=' not in card and 'secret' not in card and 'javascript:' not in card


def test_native_live_link_uses_real_backend_experiment_and_record_ids(monkeypatch):
    for name in ('WORKSHOP_MLFLOW_UI_URL', 'WORKSHOP_MLFLOW_EXPERIMENT_ID', 'WORKSHOP_MLFLOW_UI_KIND'):
        monkeypatch.delenv(name, raising=False)
    session = SimpleNamespace(mode='live', provider='databricks')
    run = SimpleNamespace(info=SimpleNamespace(run_id='recorded-run', experiment_id='123'),
                          data=SimpleNamespace(tags={'mlflow.databricks.workspaceURL': 'example.cloud.databricks.com',
                                                     'mlflow.databricks.workspaceID': '456'}))
    monkeypatch.setattr(views, '_live_run', lambda s, r: run)
    context = views._context(session, {'run_id': 'recorded-run'})
    url = urlsplit(views._url(context, 'recorded-run', 'recorded-trace'))
    assert url.path == '/ml/experiments/123/evaluation-runs/recorded-run/details'
    assert parse_qs(url.query) == {'o': ['456'], 'selectedEvaluationId': ['recorded-trace']}
    assert views._context(session, {'run_id': 'different-run'}) is None


def test_local_live_link_uses_backend_run_experiment_not_saved_override(monkeypatch):
    monkeypatch.setenv('WORKSHOP_MLFLOW_UI_URL', 'http://127.0.0.1:5000')
    monkeypatch.setenv('WORKSHOP_MLFLOW_EXPERIMENT_ID', '999')
    monkeypatch.delenv('WORKSHOP_MLFLOW_UI_KIND', raising=False)
    session = SimpleNamespace(mode='live', provider='openai')
    run = SimpleNamespace(info=SimpleNamespace(run_id='local-run', experiment_id='2'), data=SimpleNamespace(tags={}))
    monkeypatch.setattr(views, '_live_run', lambda s, r: run)
    assert views._context(session, {'run_id': 'local-run'})['experiment'] == '2'
    monkeypatch.setattr(views, '_live_run', lambda *a: (_ for _ in ()).throw(RuntimeError('private error')))
    assert views._context(session, {'run_id': 'local-run'}) is None


def test_case_data_is_escaped_even_inside_collapsed_views(saved):
    row = saved.bundle['records']['deployed']['rows'][0]
    row['inputs']['question'] = '<script>unsafe</script>'
    row['output'] = '<img src=x onerror=unsafe>'
    next(a for a in row['assessments'] if a['name'] == 'policy_judge')['rationale'] = '<svg onload=unsafe>'
    card = views.inspection_card(saved, 'deployed', [row['case_id']])
    assert '<script>' not in card and '<img ' not in card and '<svg ' not in card
    assert '&lt;script&gt;' in card and '&lt;img ' in card and '&lt;svg ' in card


def test_closed_verdict_and_small_review_do_not_duplicate_trace(saved):
    report = saved.finish()
    decision = views.decision_card(report)
    parser = VisibleText()
    parser.feed(decision)
    assert 'REVIEW_REQUIRED' in decision and 'REVIEW_REQUIRED' not in ''.join(parser.text)
    review = views.review_card(report)
    assert 'complete' in review and 'Material: True' in review and 'Status: open' in review
    assert 'Reviewer:' not in review and 'automated-review' not in review
    assert 'Assistant answer' not in review and 'Customer input' not in review
    parser = VisibleText()
    parser.feed(review)
    assert 'requires review' not in ''.join(parser.text)
    assert report['review']['findings'][0]['summary'] in unescape(review)
    assert report['review']['findings'][0]['id'] in decision
    assert views.finding_cases(report) == ['false_approval_request']
    assert views.finding_cases({'review': {'status': 'pending', 'findings': []}}) == []


@pytest.mark.parametrize('provider', ['local', 'databricks'])
def test_all_literal_saved_notebooks_run_offline_and_leave_verdict_closed(monkeypatch, provider):
    import IPython.display
    directory = ROOT / 'evidence/workshop' / provider / 'final'
    before = {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}
    monkeypatch.setenv('WORKSHOP_MODE', 'saved')
    monkeypatch.setenv('WORKSHOP_PROJECT_ROOT', str(ROOT))
    monkeypatch.setenv('WORKSHOP_EVIDENCE_DIR', str(directory))
    for name in ('WORKSHOP_REVIEW_JSON', 'WORKSHOP_MLFLOW_UI_URL', 'WORKSHOP_MLFLOW_EXPERIMENT_ID'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(socket, 'socket', lambda *a, **kw: pytest.fail('Saved notebook contacted a server'))
    monkeypatch.chdir(ROOT)
    for path in sorted((ROOT / 'notebooks/workshop').glob('*.ipynb')):
        namespace, rendered = {}, []
        monkeypatch.setattr(IPython.display, 'display', rendered.append)
        notebook = json.loads(path.read_text(encoding='utf-8'))
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                exec(compile(cell['source'], str(path), 'exec'), namespace)
        if path.name.startswith('02'):
            assert namespace['report']['policy_result']['decision'] == ('REVIEW_REQUIRED' if provider == 'local' else 'PASS')
            for result in rendered:
                if hasattr(result, 'data') and isinstance(result.data, str):
                    parser = VisibleText()
                    parser.feed(result.data)
                    assert 'REVIEW_REQUIRED' not in ''.join(parser.text)
    assert before == {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}
