#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path

try:
    import win32com.client as win32  # type: ignore
except Exception as e:
    print("ERROR: pywin32 not installed or Word COM not available:", e)
    raise SystemExit(1)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", required=True)
    args = ap.parse_args()

    p = str(Path(args.doc))
    word = win32.Dispatch("Word.Application")
    word.Visible = False
    doc = word.Documents.Open(p)
    try:
        # wdStatisticPages = 2
        pages = doc.ComputeStatistics(2)
        print({"pages": int(pages)})
    finally:
        doc.Close(False)
        word.Quit()
