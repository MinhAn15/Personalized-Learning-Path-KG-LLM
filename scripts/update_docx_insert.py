#!/usr/bin/env python
"""
Insert a UTF-8 text snippet into a .docx document after an anchor paragraph.

Example:
  python scripts/update_docx_insert.py \
      --doc DECUONGLUANVAN.docx \
      --text backend/docs/updates/prompt01_ch1_1.1.txt \
      --anchor "Bối cảnh nghiên cứu" \
      --occurrence 1 \
      --inplace

If the anchor is not found, the text is appended at the end. When --inplace is
not provided, the script writes a new file with suffix .updated.docx
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from docx import Document  # type: ignore


def _normalize(s: str) -> str:
    import re
    return re.sub(r"\s+", " ", s or "").strip().lower()


def insert_after_anchor(doc: Document, text: str, anchor: str, occurrence: int = 1) -> bool:
    """Insert a new paragraph with `text` after the `occurrence`-th paragraph containing `anchor`.
    Returns True if inserted after anchor, False if anchor not found and appended at end.
    """
    idx = -1
    count = 0
    for i, p in enumerate(doc.paragraphs):
        if anchor.lower() in p.text.lower():
            count += 1
            if count == occurrence:
                idx = i
                break

    # Append if anchor not found
    if idx < 0 or idx >= len(doc.paragraphs):
        doc.add_paragraph(text)
        return False

    # Insert after idx using low-level xml API
    anchor_p = doc.paragraphs[idx]
    new_p = anchor_p.insert_paragraph_before("")  # creates a paragraph before
    # Move the new paragraph after by swapping elements
    anchor_p._element.addnext(new_p._element)
    new_p.text = text
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Insert text into a DOCX after a matching anchor paragraph.")
    parser.add_argument("--doc", required=True, help="Path to .docx file to update.")
    parser.add_argument("--text", required=True, help="Path to UTF-8 text file to insert.")
    parser.add_argument("--anchor", default="Bối cảnh nghiên cứu", help="Anchor phrase to search for (case-insensitive).")
    parser.add_argument("--occurrence", type=int, default=1, help="Which occurrence of the anchor to target (1-based).")
    parser.add_argument("--inplace", action="store_true", help="Overwrite the input .docx instead of writing a new file.")
    parser.add_argument("--dedupe", action="store_true", help="Remove duplicate occurrences of the inserted text, keeping the earliest.")

    args = parser.parse_args(argv)

    doc_path = Path(args.doc)
    text_path = Path(args.text)
    if not doc_path.exists():
        print(f"DOCX file not found: {doc_path}")
        return 1
    if not text_path.exists():
        print(f"Text file not found: {text_path}")
        return 1

    # Read text with robust encoding fallback
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
    last_err = None
    for enc in encodings:
        try:
            text = text_path.read_text(encoding=enc)
            break
        except Exception as e:  # pragma: no cover
            last_err = e
            text = None
            continue
    if text is None:
        raise SystemExit(f"Failed to read text file {text_path}: {last_err}")
    doc = Document(str(doc_path))

    # Skip if content already present (simple containment check)
    joined = "\n".join(p.text for p in doc.paragraphs)
    if text.strip() and text.strip() in joined:
        inserted = True
    else:
        inserted = insert_after_anchor(doc, text, args.anchor, args.occurrence)

    if args.dedupe and text.strip():
        norm_text = _normalize(text)
        key = norm_text[:80]
        matches = []
        for i, p in enumerate(doc.paragraphs):
            if key and key in _normalize(p.text):
                matches.append(i)
        if len(matches) > 1:
            # Keep the first, remove the rest (from the end to keep indices stable)
            for i in reversed(matches[1:]):
                elm = doc.paragraphs[i]._element
                parent = elm.getparent()
                parent.remove(elm)

    if args.inplace:
        out_path = doc_path
    else:
        out_path = doc_path.with_name(doc_path.stem + ".updated" + doc_path.suffix)

    doc.save(str(out_path))
    where = "after anchor" if inserted else "at end"
    print(f"Saved {out_path} ({where}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
