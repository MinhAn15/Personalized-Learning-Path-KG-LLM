#!/usr/bin/env python
"""
Repair a DOCX by re-saving and normalizing fonts for readability.
- Sets default font to 'Times New Roman' for runs that don't have an explicit font
- Preserves Consolas for code blocks and Cambria Math for equations if already set

Usage:
  python scripts/repair_docx_fonts.py --doc backend/DECUONGLUANVAN.docx [--out backend/DECUONGLUANVAN_fixed.docx]
"""
from __future__ import annotations

import argparse
from pathlib import Path
from docx import Document  # type: ignore

SAFE_FONT = "Times New Roman"
CODE_FONT = "Consolas"
MATH_FONT = "Cambria Math"


def normalize_fonts(doc: Document) -> None:
    for p in doc.paragraphs:
        for r in p.runs:
            name = (r.font.name or "").strip() if r.font is not None else ""
            # Keep known code/math fonts; otherwise set a safe default
            if name in (CODE_FONT, MATH_FONT):
                continue
            # If font not set, enforce a safe Latin/Vietnamese-friendly font
            if not name:
                r.font.name = SAFE_FONT


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", required=True)
    ap.add_argument("--out", required=False)
    args = ap.parse_args()

    inp = Path(args.doc)
    if not inp.exists():
        raise SystemExit(f"Document not found: {inp}")

    out = Path(args.out) if args.out else inp.with_name(inp.stem + "_fixed" + inp.suffix)

    doc = Document(str(inp))
    normalize_fonts(doc)
    doc.save(str(out))
    print(f"Saved repaired document to {out}")
