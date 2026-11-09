# docs/

`TECHNICAL.md` is the technical document (source of truth). To make the PDF for submission:

    pip install -r requirements-docs.txt
    python docs/build_pdf.py --team nirakshak          # writes docs/out/nirakshak.pdf
    python docs/build_pdf.py --team nirakshak --png    # also renders docs/out/page-N.png to eyeball

Pure Python (reportlab), no network or system tools needed; output is deterministic.
Supports the markdown subset TECHNICAL.md uses (headings, lists, tables, fenced and inline code, bold).
`docs/out/` and `*.pdf` are git-ignored. `TODO(result)` markers are printed as-is: fill them in
`TECHNICAL.md` first, then rebuild.

`TECHNICAL.md` describes the champion pipeline in `mtah/v2/`. Constants are tabled in its section 3
with the reason recorded in the briefs or code comments; `UNJUSTIFIED` and `TODO` markers must be
resolved (or deliberately kept) before the PDF is submitted. The build prints the page count (aim: 4).
