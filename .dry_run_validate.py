import pandas as pd
import os
import traceback

NODE_PATH = os.path.join('backend', 'data', 'github_import', 'master_nodes.csv')
RELS_PATH = os.path.join('backend', 'data', 'github_import', 'master_relationships.csv')

print('Reading:', NODE_PATH)
print('Reading:', RELS_PATH)

try:
    nodes = pd.read_csv(NODE_PATH, dtype=str).fillna('')
    rels = pd.read_csv(RELS_PATH, dtype=str).fillna('')
except Exception as e:
    traceback.print_exc()
    print('FAILED_TO_READ')
    raise SystemExit(1)

print('\nBasic counts:')
print('  nodes:', len(nodes))
print('  relationships:', len(rels))

# Check required columns
required_node_cols = ['Node_ID']
required_rel_cols = ['Source_ID', 'Target_ID']

missing_node_cols = [c for c in required_node_cols if c not in nodes.columns]
missing_rel_cols = [c for c in required_rel_cols if c not in rels.columns]

print('\nMissing columns:')
print('  nodes missing:', missing_node_cols)
print('  rels missing:', missing_rel_cols)

# Unique Node_ID check
dup_nodes = nodes[nodes['Node_ID'].duplicated()]['Node_ID'].tolist() if 'Node_ID' in nodes.columns else []
print('\nDuplicate Node_IDs count:', len(dup_nodes))
if dup_nodes:
    print('  examples:', dup_nodes[:10])

# Relationship integrity
if 'Node_ID' in nodes.columns and 'Source_ID' in rels.columns and 'Target_ID' in rels.columns:
    node_set = set(nodes['Node_ID'])
    missing_src = set(rels['Source_ID']) - node_set
    missing_tgt = set(rels['Target_ID']) - node_set
    print('\nRelationship referential integrity:')
    print('  missing sources:', len(missing_src))
    print('  missing targets:', len(missing_tgt))
    if missing_src:
        print('   examples:', list(missing_src)[:10])
    if missing_tgt:
        print('   examples:', list(missing_tgt)[:10])

# Numeric column checks
numeric_checks = {
    'Priority': float,
    'Time_Estimate': float,
}
print('\nNumeric parsing checks:')
for col, typ in numeric_checks.items():
    if col in nodes.columns:
        def can_convert(x):
            try:
                if x == '' or x == 'Not Available':
                    return True
                typ(x)
                return True
            except Exception:
                return False
        bad = [v for v in nodes[col].unique() if not can_convert(v)]
        print(f'  {col}: unique values={len(nodes[col].unique())}, unparsable examples={bad[:5]}')
    else:
        print(f'  {col}: NOT PRESENT')

# Relationship numeric
rel_numeric_checks = {'Weight': float, 'Dependency': float}
for col, typ in rel_numeric_checks.items():
    if col in rels.columns:
        bad = []
        for v in rels[col].unique():
            try:
                if v == '' or v == 'Not Available':
                    continue
                typ(v)
            except Exception:
                bad.append(v)
        print(f'  rel {col}: unique={len(rels[col].unique())}, unparsable examples={bad[:5]}')
    else:
        print(f'  rel {col}: NOT PRESENT')

# Summaries: top contexts
if 'Context' in nodes.columns:
    top_contexts = nodes['Context'].value_counts().head(10)
    print('\nTop contexts:')
    print(top_contexts.to_string())

print('\nDry-run validation complete.')
