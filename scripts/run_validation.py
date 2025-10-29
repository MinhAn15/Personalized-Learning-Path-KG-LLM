#!/usr/bin/env python
"""Basic validation helper for SPR-generated CSV files.

Checks required columns on nodes/relationships CSVs and reports missing values.
Optionally prints a snippet of the prompt templates to remind how to run LLM validation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

NODES_REQUIRED = [
    "Node_ID",
    "Sanitized_Concept",
    "Context",
    "Definition",
    "Learning_Objective",
    "Skill_Level",
    "Time_Estimate",
    "Difficulty",
    "Priority",
    "Semantic_Tags",
]

RELS_REQUIRED = [
    "Source_ID",
    "Target_ID",
    "Relationship_Type",
]

PROMPT_SNIPPET_LINES = 12


def validate_columns(df: pd.DataFrame, required: List[str]) -> List[str]:
    return [col for col in required if col not in df.columns]


def blank_counts(df: pd.DataFrame, required: List[str]) -> Dict[str, int]:
    return {col: int(df[col].isna().sum()) for col in required if col in df.columns}


def load_prompt_snippet(prompt_path: Path) -> str:
    if not prompt_path.is_file():
        return ""
    lines = prompt_path.read_text(encoding="utf-8").splitlines()[:PROMPT_SNIPPET_LINES]
    return "\n".join(lines)


def write_report(report_path: Path, payload: Dict) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate SPR CSV files")
    parser.add_argument("--nodes", type=Path, required=True, help="Path to nodes CSV")
    parser.add_argument("--relationships", type=Path, required=True, help="Path to relationships CSV")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("backend/data/output/validation_report.json"),
        help="Where to write JSON validation report",
    )
    parser.add_argument(
        "--show-prompts",
        action="store_true",
        help="Print first lines of SPR generator/validation prompts as a reminder",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    nodes_df = pd.read_csv(args.nodes)
    rels_df = pd.read_csv(args.relationships)

    nodes_missing = validate_columns(nodes_df, NODES_REQUIRED)
    rels_missing = validate_columns(rels_df, RELS_REQUIRED)

    report = {
        "nodes_file": str(args.nodes),
        "relationships_file": str(args.relationships),
        "nodes_missing_columns": nodes_missing,
        "relationships_missing_columns": rels_missing,
        "nodes_blank_counts": blank_counts(nodes_df, NODES_REQUIRED),
        "relationships_blank_counts": blank_counts(rels_df, RELS_REQUIRED),
        "total_nodes": int(len(nodes_df)),
        "total_relationships": int(len(rels_df)),
    }

    write_report(args.report, report)
    print(f"Validation report written to {args.report}")

    if args.show_prompts:
        generator_snippet = load_prompt_snippet(Path("prompts/spr_generator_prompt.txt"))
        validation_snippet = load_prompt_snippet(Path("prompts/validation_prompt.txt"))
        if generator_snippet:
            print("\nSPR Generator prompt snippet:\n" + generator_snippet)
        if validation_snippet:
            print("\nSPR Validation prompt snippet:\n" + validation_snippet)
        # Outline how to integrate LLM manually
        print(
            "\nTo perform semantic validation, feed the CSV rows into the SPR Validation prompt above "
            "using your preferred LLM client and review the model feedback."
        )


if __name__ == "__main__":
    main()
