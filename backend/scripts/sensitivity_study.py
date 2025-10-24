"""
Sensitivity study for A* heuristic weights (One-At-a-Time ±10% perturbations).

This script connects to the project's Neo4j instance (reads credentials from
`backend/src/config.py` / .env), samples a set of nodes (or start-goal pairs),
computes baseline heuristic values using the existing `heuristic()` function,
then perturbs each selected weight by +10% and -10% and measures the effect on:

- mean_abs_delta_h: average absolute change in h(n)
- rel_change_h: mean relative change (delta / |baseline|)
- avg_rank_move: average rank position movement across sampled nodes
- spearman_rho: Spearman correlation between baseline and perturbed rankings

Results are saved to `backend/data/output/sensitivity_results.csv` and a
human-readable summary is printed.

Usage:
  python backend/scripts/sensitivity_study.py --sample-size 300 --student-id <id>

Note: this is intended to run against a test/dev Neo4j instance.
"""

import os
import csv
import argparse
import statistics
from math import isclose
from collections import defaultdict

from neo4j import GraphDatabase

from backend.src.config import Config
from backend.src.data_loader import execute_cypher_query
from backend.src.path_generator import heuristic, calculate_dynamic_weights

try:
    from scipy.stats import spearmanr, ttest_rel
except Exception:
    spearmanr = None
    ttest_rel = None

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'output')
os.makedirs(OUT_DIR, exist_ok=True)
OUT_CSV = os.path.join(OUT_DIR, 'sensitivity_results.csv')


def get_driver():
    cfg = Config.NEO4J_CONFIG
    if not all(cfg.values()):
        raise RuntimeError('NEO4J_CONFIG incomplete; set NEO4J_URL, NEO4J_USER, NEO4J_PASSWORD in .env')
    uri = cfg['url']
    user = cfg['username']
    password = cfg['password']
    return GraphDatabase.driver(uri, auth=(user, password))


def sample_node_ids(driver, limit=300):
    q = f"""
    MATCH (n:KnowledgeNode)
    RETURN n.{Config.PROPERTY_ID} AS id
    LIMIT $limit
    """
    rows = execute_cypher_query(driver, q, params={'limit': limit})
    return [r['id'] for r in rows]


def compute_baseline_h(driver, node_ids, student_id=None, context=''):
    dyn = calculate_dynamic_weights(student_id)
    baseline = {}
    for nid in node_ids:
        h = heuristic(nid, driver, goal_tags=[], dynamic_weights=dyn, Config=Config, context=context)
        baseline[nid] = float(h)
    return baseline


def perturb_and_measure(driver, node_ids, baseline_h, weight_name, multiplier, student_id=None, context=''):
    # Prepare dynamic weights copy (we perturb the dynamic scalar where applicable)
    dyn = calculate_dynamic_weights(student_id)
    # If perturbing skill_level default, handle nested dict
    if weight_name == 'skill_level.default':
        sl = dyn.get('skill_level', {})
        if not isinstance(sl, dict):
            sl = {'default': float(sl)}
        sl_default = float(sl.get('default', Config.ASTAR_HEURISTIC_WEIGHTS['skill_level'].get('default', 0.5)))
        sl['default'] = sl_default * multiplier
        dyn['skill_level'] = sl
    else:
        # either weights at top-level or time_estimate
        dyn[weight_name] = float(dyn.get(weight_name, Config.ASTAR_HEURISTIC_WEIGHTS.get(weight_name, 1.0))) * multiplier

    pert_h = {}
    for nid in node_ids:
        h = heuristic(nid, driver, goal_tags=[], dynamic_weights=dyn, Config=Config, context=context)
        pert_h[nid] = float(h)

    # Compute metrics
    node_list = node_ids
    deltas = [abs(pert_h[n] - baseline_h[n]) for n in node_list]
    rel_changes = []
    for n in node_list:
        b = baseline_h[n]
        if isclose(b, 0.0):
            rel_changes.append(0.0)
        else:
            rel_changes.append((pert_h[n] - b) / abs(b))

    mean_abs_delta_h = statistics.mean(deltas) if deltas else 0.0
    mean_rel_change = statistics.mean(rel_changes) if rel_changes else 0.0

    # Ranking metrics
    baseline_sorted = sorted(node_list, key=lambda x: baseline_h[x])
    pert_sorted = sorted(node_list, key=lambda x: pert_h[x])
    pos_baseline = {nid: i for i, nid in enumerate(baseline_sorted)}
    pos_pert = {nid: i for i, nid in enumerate(pert_sorted)}
    avg_rank_move = statistics.mean([abs(pos_baseline[n] - pos_pert[n]) for n in node_list])

    spearman = None
    if spearmanr is not None:
        try:
            b_vals = [baseline_h[n] for n in node_list]
            p_vals = [pert_h[n] for n in node_list]
            rho, pval = spearmanr(b_vals, p_vals)
            spearman = float(rho)
        except Exception:
            spearman = None

    # Basic paired t-test on h values
    t_stat = None
    p_val = None
    if ttest_rel is not None:
        try:
            import numpy as _np
            b_arr = _np.array([baseline_h[n] for n in node_list])
            p_arr = _np.array([pert_h[n] for n in node_list])
            t, pv = ttest_rel(b_arr, p_arr)
            t_stat = float(t)
            p_val = float(pv)
        except Exception:
            t_stat = None
            p_val = None

    return {
        'weight': weight_name,
        'multiplier': multiplier,
        'mean_abs_delta_h': mean_abs_delta_h,
        'mean_rel_change': mean_rel_change,
        'avg_rank_move': avg_rank_move,
        'spearman_rho': spearman,
        't_stat': t_stat,
        'p_val': p_val
    }


def run_oat_study(driver, sample_size=300, student_id=None, context=''):
    print(f"Sampling {sample_size} nodes from KG...")
    node_ids = sample_node_ids(driver, limit=sample_size)
    print(f"Sampled {len(node_ids)} nodes")

    baseline_h = compute_baseline_h(driver, node_ids, student_id=student_id, context=context)

    # We will test these weight keys. For skill_level we perturb default scalar.
    weight_keys = ['priority', 'difficulty_standard', 'difficulty_advanced', 'skill_level.default', 'time_estimate']
    multipliers = [0.9, 1.1]  # -10%, +10%

    results = []
    for w in weight_keys:
        for m in multipliers:
            print(f"Perturbing {w} by multiplier {m}...")
            res = perturb_and_measure(driver, node_ids, baseline_h, w, m, student_id=student_id, context=context)
            results.append(res)
            print(f"  mean_abs_delta_h={res['mean_abs_delta_h']:.6f}, mean_rel_change={res['mean_rel_change']:.4f}, avg_rank_move={res['avg_rank_move']:.2f}, spearman={res['spearman_rho']}")

    # Save CSV
    write_header = not os.path.exists(OUT_CSV)
    with open(OUT_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        if write_header:
            writer.writeheader()
        for r in results:
            writer.writerow(r)

    print('\nSensitivity study complete. Results appended to:', OUT_CSV)
    return results


def summarize_results(results):
    print('\nSummary:')
    for r in results:
        print(f"{r['weight']} * {r['multiplier']:+.2f}: mean_abs_delta_h={r['mean_abs_delta_h']:.6f}, mean_rel={r['mean_rel_change']:.4f}, avg_rank_move={r['avg_rank_move']:.2f}, rho={r['spearman_rho']}, p={r['p_val']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample-size', type=int, default=300)
    parser.add_argument('--student-id', type=str, default=None)
    parser.add_argument('--context', type=str, default='')
    args = parser.parse_args()

    driver = get_driver()
    try:
        results = run_oat_study(driver, sample_size=args.sample_size, student_id=args.student_id, context=args.context)
        summarize_results(results)
    finally:
        driver.close()


if __name__ == '__main__':
    main()
