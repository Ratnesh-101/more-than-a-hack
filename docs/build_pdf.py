"""Build the technical-document PDF from docs/TECHNICAL.md.

    python docs/build_pdf.py --team nirakshak            -> docs/out/nirakshak.pdf
    python docs/build_pdf.py --team nirakshak --png      -> also docs/out/page-N.png (needs pymupdf)

Pure Python (reportlab, see requirements-docs.txt); no network, no system tools. Handles the
markdown subset TECHNICAL.md uses: headings, paragraphs, bullet and numbered lists (with wrapped
continuation lines), pipe tables, fenced code, `inline code`, **bold**, and figures `![caption](path)`.
Output is deterministic (reportlab invariant mode: fixed dates and ids).
"""
import argparse
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus import (Image, KeepTogether, ListFlowable, ListItem, Paragraph,
                                Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle)

DOCS = Path(__file__).resolve().parent
MARGIN = 16 * mm
WIDTH = A4[0] - 2 * MARGIN

BASE = ParagraphStyle("base", fontName="Helvetica", fontSize=8.8, leading=11.2, alignment=TA_LEFT)
H1 = ParagraphStyle("h1", parent=BASE, fontName="Helvetica-Bold", fontSize=14, leading=17, spaceAfter=4)
H2 = ParagraphStyle("h2", parent=BASE, fontName="Helvetica-Bold", fontSize=11, leading=13.5,
                    spaceBefore=8, spaceAfter=3, textColor=colors.HexColor("#14365d"))
P = ParagraphStyle("p", parent=BASE, spaceAfter=3.5)
CELL = ParagraphStyle("cell", parent=BASE, fontSize=7.4, leading=9)
CELLH = ParagraphStyle("cellh", parent=CELL, fontName="Helvetica-Bold")
CODE = ParagraphStyle("code", parent=BASE, fontName="Courier", fontSize=7.3, leading=9,
                      backColor=colors.HexColor("#f2f2f2"), borderPadding=4, spaceBefore=3,
                      spaceAfter=7, leftIndent=4)
CAPTION = ParagraphStyle("caption", parent=BASE, fontSize=7.6, leading=9.5,
                         spaceBefore=2.5, spaceAfter=5.5, textColor=colors.HexColor("#333333"))


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def inline(text):
    text = text.encode("latin-1", "replace").decode("latin-1")
    out = []
    for part in re.split(r"(`[^`]*`)", text):
        if len(part) > 1 and part.startswith("`") and part.endswith("`"):
            out.append('<font name="Courier">%s</font>' % esc(part[1:-1]))
        else:
            out.append(re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", esc(part)))
    return "".join(out)


def table(rows):
    header, body = rows[0], rows[1:]
    n = len(header)
    data = [[Paragraph(inline(c), CELLH) for c in header]]
    for r in body:
        data.append([Paragraph(inline(c), CELL) for c in (r + [""] * n)[:n]])
    # column widths follow the longest cell (clamped), so narrow columns stay narrow
    # Every column gets room for its longest unbreakable word (code names are Courier, which is
    # wider), then the spare width is shared in proportion to how much text the column holds.
    def longest_word(i):
        w = 0
        for r in rows:
            for word in (r[i].split() if i < len(r) else []):
                font = "Courier" if "`" in word else "Helvetica"
                w = max(w, stringWidth(word.replace("`", ""), font, CELL.fontSize))
        return w + 8
    mins = [longest_word(i) for i in range(n)]
    size = [max(1, sum(len(r[i]) for r in rows if i < len(r))) for i in range(n)]
    spare = max(0, WIDTH - sum(mins))
    widths = [m + spare * z / float(sum(size)) for m, z in zip(mins, size)]
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dde6f0")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    return [t, Spacer(1, 6)]


def wrap_code(lines, maxchars):
    out = []
    for ln in lines:
        while len(ln) > maxchars:
            cut = ln.rfind(" ", 0, maxchars)
            cut = cut if cut > maxchars // 2 else maxchars
            out.append(ln[:cut])
            ln = "    " + ln[cut:].lstrip()
        out.append(ln)
    return out


def listing(items, ordered):
    flow = [ListItem(Paragraph(inline(t), BASE), leftIndent=14) for t in items]
    kw = dict(bulletType="1", bulletFormat="%s.") if ordered else dict(bulletType="bullet", start="-")
    return [ListFlowable(flow, bulletFontName="Helvetica", bulletFontSize=8.8, leftIndent=14, **kw),
            Spacer(1, 3)]


LIST_RE = r"(\d+\.|-)\s"
IMAGE_RE = re.compile(r"^!\[(.*?)\]\s*\((.*?)\)\s*$")


def parse(md, base_dir=DOCS):
    flow, lines, i = [], md.splitlines(), 0
    maxchars = int((WIDTH - 12) / (0.6 * CODE.fontSize))
    while i < len(lines):
        ln = lines[i]
        if not ln.strip():
            i += 1
        elif ln.startswith("```"):
            i += 1
            code = []
            while i < len(lines) and not lines[i].startswith("```"):
                code.append(lines[i])
                i += 1
            i += 1
            flow.append(Preformatted("\n".join(wrap_code(code, maxchars)), CODE))
        elif ln.startswith("# "):
            flow.append(Paragraph(inline(ln[2:]), H1))
            i += 1
        elif ln.startswith("## "):
            flow.append(Paragraph(inline(ln[3:]), H2))
            i += 1
        elif ln.lstrip().startswith("|"):
            rows = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                if not re.fullmatch(r"[\s|:-]+", lines[i]):
                    rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            flow.extend(table(rows))
        elif IMAGE_RE.match(ln.strip()):
            m = IMAGE_RE.match(ln.strip())
            caption, img_str = m.group(1), m.group(2).strip().strip("'\"")
            img_path = Path(img_str)
            if not img_path.is_absolute():
                candidates = [
                    (base_dir / img_str).resolve(),
                    (DOCS / img_str).resolve(),
                    (Path.cwd() / img_str).resolve(),
                    (Path.cwd() / "docs" / img_str).resolve(),
                ]
                for cand in candidates:
                    if cand.exists():
                        img_path = cand
                        break
                else:
                    img_path = (base_dir / img_str).resolve()
            if not img_path.exists():
                raise FileNotFoundError("Image not found: %s (resolved as %s)" % (img_str, img_path))
            img = Image(str(img_path))
            if img.imageWidth > 0:
                w = WIDTH
                h = img.imageHeight * (w / img.imageWidth)
                img.drawWidth = w
                img.drawHeight = h
            items = [img]
            if caption:
                items.append(Paragraph(inline(caption), CAPTION))
            else:
                items.append(Spacer(1, 4))
            flow.append(KeepTogether(items))
            i += 1
        elif re.match(LIST_RE, ln):
            ordered = ln[0].isdigit()
            pat = r"\d+\.\s+" if ordered else r"-\s+"
            items = []
            while i < len(lines) and lines[i].strip():
                m = re.match(pat, lines[i])
                if m:
                    items.append(lines[i][m.end():])
                elif re.match(LIST_RE, lines[i]) or not lines[i].startswith(" "):
                    break
                else:
                    items[-1] += " " + lines[i].strip()
                i += 1
            flow.extend(listing(items, ordered))
        else:
            para = [ln.strip()]
            i += 1
            while i < len(lines) and lines[i].strip() and not re.match(
                    r"(```|#|\||\d+\.\s|-\s|!\[)", lines[i]):
                para.append(lines[i].strip())
                i += 1
            flow.append(Paragraph(inline(" ".join(para)), P))
    return flow


def footer(team):
    def draw(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawString(MARGIN, 9 * mm, "Team %s - More Than a Hack 2027, Phase 1 - technical document" % team)
        canvas.drawRightString(A4[0] - MARGIN, 9 * mm, "Page %d" % doc.page)
        canvas.restoreState()
    return draw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True)
    ap.add_argument("--src", default=str(DOCS / "TECHNICAL.md"))
    ap.add_argument("--out-dir", default=str(DOCS / "out"))
    ap.add_argument("--png", action="store_true", help="also render page PNGs (needs pymupdf)")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pdf = out / ("%s.pdf" % a.team)
    doc = SimpleDocTemplate(str(pdf), pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN,
                            topMargin=14 * mm, bottomMargin=16 * mm, invariant=1,
                            title="Technical document - team %s" % a.team, author=a.team)
    src_path = Path(a.src).resolve()
    doc.build(parse(src_path.read_text(encoding="utf-8"), base_dir=src_path.parent),
              onFirstPage=footer(a.team), onLaterPages=footer(a.team))
    print("wrote", pdf)
    if a.png:
        import pymupdf
        d = pymupdf.open(str(pdf))
        for n, page in enumerate(d, 1):
            page.get_pixmap(dpi=110).save(str(out / ("page-%d.png" % n)))
        print("pages:", len(d))


if __name__ == "__main__":
    main()
