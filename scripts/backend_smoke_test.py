import compileall
import importlib
import pkgutil
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'backend' / 'src'
print(f"Running backend smoke tests on: {ROOT}")

# 1) compileall
print('\n1) Running compileall...')
res = compileall.compile_dir(str(ROOT), force=True, quiet=1)
print('compileall result:', 'PASS' if res else 'FAIL')

# 2) attempt to import top-level package modules under backend.src
# Add repository root to sys.path so that 'backend' is importable as a package
repo_root = ROOT.parent.parent
sys.path.insert(0, str(repo_root))

errors = []
print('\n2) Importing modules under backend.src...')
for finder, name, ispkg in pkgutil.walk_packages([str(ROOT)], prefix='backend.src.'):
    print('Importing', name)
    try:
        importlib.import_module(name)
    except Exception as e:
        tb = traceback.format_exc()
        errors.append((name, tb))
        print(f'ERROR importing {name}:', e)

print('\nSummary:')
if errors:
    print('IMPORT FAIL - errors detected in the following modules:')
    for name, tb in errors:
        print('---', name)
        print(tb)
else:
    print('IMPORT PASS - no import-time errors detected.')

# Exit code
if errors or not res:
    raise SystemExit(1)
