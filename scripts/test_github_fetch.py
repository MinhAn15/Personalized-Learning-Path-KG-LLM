import os
import sys
# ensure repo root in path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from backend.src import data_loader as dl

fp = 'backend/data/github_import/' + dl.Config.IMPORT_NODES_FILE
print('Attempting to fetch from GitHub path:', fp)
try:
    content = dl._get_github_file_content(fp)
    print('Fetched length:', len(content))
    print('First 200 chars:\n', content[:200])
except Exception as e:
    print('GitHub fetch failed:', repr(e))
