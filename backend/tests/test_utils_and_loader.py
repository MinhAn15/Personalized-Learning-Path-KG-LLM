import os
import sys
import pytest

# Ensure repo root is on sys.path so 'backend' package can be imported during tests
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from backend.src import data_loader as dl


def test_jaccard_similarity_basic():
    assert dl.jaccard_similarity(['a', 'b'], ['b', 'c']) == pytest.approx(1/3)
    assert dl.jaccard_similarity([], []) == 0.0


def test_merge_properties_basic():
    existing = {'A': '1', dl.Config.PROPERTY_SEMANTIC_TAGS: 'x;y'}
    new = {'A': '2', dl.Config.PROPERTY_SEMANTIC_TAGS: 'y;z'}
    merged = dl.merge_properties(existing, new)
    assert merged['A'] == '2'
    # tags merged and deduped
    tags = set(merged[dl.Config.PROPERTY_SEMANTIC_TAGS].split(';'))
    assert tags == {'x', 'y', 'z'}


def test_calculate_learning_speed():
    perf = ['n1:90:30', 'n2:80:45']
    assert dl.calculate_learning_speed(perf) == pytest.approx((30+45)/2)
    # malformed entries are ignored
    perf2 = ['badformat', 'n1:90:20']
    assert dl.calculate_learning_speed(perf2) == pytest.approx(20)


def test_check_and_load_kg_local_fallback(monkeypatch, tmp_path):
    # Force _get_github_file_content to raise so code falls back to local files
    monkeypatch.setattr(dl, '_get_github_file_content', lambda path: (_ for _ in ()).throw(Exception('no github')))

    # Create small local CSV files in expected location
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    target_dir = os.path.join(repo_root, 'backend', 'data', 'github_import')
    os.makedirs(target_dir, exist_ok=True)
    nodes_path = os.path.join(target_dir, dl.Config.IMPORT_NODES_FILE)
    rels_path = os.path.join(target_dir, dl.Config.IMPORT_RELATIONSHIPS_FILE)
    with open(nodes_path, 'w', encoding='utf-8') as f:
        f.write('Node_ID,Sanitized_Concept,Priority,Time_Estimate\n1,Intro,1,10')
    with open(rels_path, 'w', encoding='utf-8') as f:
        f.write('Source_ID,Target_ID,Relationship_Type,Weight,Dependency\n1,1,RELATED_TO,1.0,0')

    # Patch execute_cypher_query to a no-op and provide a dummy driver/session for relationships
    monkeypatch.setattr(dl, 'execute_cypher_query', lambda driver, query, params=None: [])

    class DummyTx:
        def __init__(self):
            self.runs = []
        def run(self, cypher, **kwargs):
            self.runs.append((cypher, kwargs))
        def commit(self):
            return None

    class DummySession:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            return False
        def begin_transaction(self):
            return DummyTx()

    class DummyDriver:
        def session(self, database=None):
            return DummySession()

    res = dl.check_and_load_kg(DummyDriver())
    assert isinstance(res, dict)
    assert res.get('status') == 'success'
