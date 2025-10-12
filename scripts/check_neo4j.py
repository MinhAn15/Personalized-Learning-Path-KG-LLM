import os
import sys
from neo4j import GraphDatabase

# Ensure repo root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from backend.src.config import NEO4J_CONFIG

cfg = NEO4J_CONFIG
print('NEO4J URL:', cfg.get('url'))

drv = None
try:
    drv = GraphDatabase.driver(cfg.get('url'), auth=(cfg.get('username'), cfg.get('password')))
    print('Created driver, verifying connectivity...')
    drv.verify_connectivity()
    print('verify_connectivity: OK')
    with drv.session(database='neo4j') as s:
        res = s.run('RETURN 1 AS v')
        print('Cypher test result:', [r['v'] for r in res])
except Exception as e:
    print('Neo4j connectivity test failed:', e)
finally:
    if drv:
        drv.close()
        print('Driver closed')
