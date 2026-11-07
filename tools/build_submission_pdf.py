"""Builds submission/ML_Assignment_2_Submission.pdf = fill-in template pages + the complete README.

Usage (all options optional; omitted ones stay as visible placeholders):
  python tools/build_submission_pdf.py --name "..." --student-id "..." --email "..." \
      --github URL --app URL --screenshot screenshots/bits_virtual_lab.png
"""
import argparse, re, sys
from pathlib import Path

import matplotlib
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, Preformatted, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
for nm, f in [("DV", "DejaVuSans.ttf"), ("DV-B", "DejaVuSans-Bold.ttf"), ("DV-I", "DejaVuSans-Oblique.ttf"),
              ("DV-BI", "DejaVuSans-BoldOblique.ttf"), ("DV-M", "DejaVuSansMono.ttf")]:
    pdfmetrics.registerFont(TTFont(nm, str(FONT_DIR / f)))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")

PH = "#B45309"
ap = argparse.ArgumentParser()
for a in ("name", "student-id", "email", "github", "app", "screenshot"):
    ap.add_argument(f"--{a}")
ap.add_argument("--out", default=str(ROOT / "submission" / "ML_Assignment_2_Submission.pdf"))
args = ap.parse_args()

st = {
    "title": ParagraphStyle("t", fontName="DV-B", fontSize=20, leading=25, spaceAfter=4, textColor=colors.HexColor("#1B5E20")),
    "sub": ParagraphStyle("s", fontName="DV", fontSize=10.5, leading=14, textColor=colors.HexColor("#444444"), spaceAfter=10),
    "h1": ParagraphStyle("h1", fontName="DV-B", fontSize=14, leading=18, spaceBefore=12, spaceAfter=5, textColor=colors.HexColor("#1B5E20")),
    "h2": ParagraphStyle("h2", fontName="DV-B", fontSize=11.5, leading=15, spaceBefore=8, spaceAfter=3),
    "p": ParagraphStyle("p", fontName="DV", fontSize=8.8, leading=12.4, spaceAfter=4, alignment=TA_LEFT),
    "li": ParagraphStyle("li", fontName="DV", fontSize=8.8, leading=12.4, leftIndent=14, bulletIndent=3, spaceAfter=1.5),
    "cell": ParagraphStyle("c", fontName="DV", fontSize=7.4, leading=9.6),
    "cellh": ParagraphStyle("ch", fontName="DV-B", fontSize=7.4, leading=9.6, textColor=colors.white),
    "code": ParagraphStyle("code", fontName="DV-M", fontSize=6.9, leading=8.6, backColor=colors.HexColor("#F3F4F6"),
                           borderPadding=4, leftIndent=4, spaceBefore=3, spaceAfter=6),
}


def _esc(t: str) -> str:
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _fmt(t: str) -> str:
    t = _esc(t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"<i>\1</i>", t)
    t = re.sub(r"(https?://[^\s<>)\]]+)", lambda m: f'<font color="#1D4ED8">{m.group(1)}</font>', t)
    t = re.sub(r"(&lt;YOUR_[A-Z_]+&gt;)", rf'<font color="{PH}"><b>\1</b></font>', t)
    return t


def inline(t: str) -> str:
    """Markdown inline -> ReportLab markup. Code spans are handled first so their
    contents (e.g. reports/*.csv) are never touched by the bold/italic rules."""
    parts = t.split("`")
    out = []
    for k, seg in enumerate(parts):
        if k % 2 == 1 and k < len(parts) - 1 or (k % 2 == 1 and len(parts) % 2 == 1):
            out.append(f'<font face="DV-M" color="#9A3412">{_esc(seg)}</font>')
        else:
            out.append(_fmt(seg))
    return "".join(out)


def make_table(rows, width):
    ncol = len(rows[0])
    wts = []
    for j in range(ncol):
        L = max(len(r[j]) for r in rows if j < len(r))
        wts.append(min(max(L, 5), 70) ** 0.8)
    tot = sum(wts); cw = [width * w / tot for w in wts]
    data = [[Paragraph(inline(c), st["cellh"] if i == 0 else st["cell"]) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=cw, repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2E7D32")),
                           ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B8C4B5")),
                           ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F8F2")]),
                           ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                           ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]))
    return t


def md_to_flow(md: str, width: float):
    md = re.sub(r"<!--.*?-->", "", md, flags=re.S)
    flow, lines, i = [], md.splitlines(), 0
    pend = None            # (kind, text, bullet) buffered paragraph / list item

    def flush():
        nonlocal pend
        if pend:
            kind, text, bullet = pend
            if kind == "li":
                flow.append(Paragraph(inline(text), st["li"], bulletText=bullet))
            elif kind == "q":
                flow.append(Paragraph(f"<i>{inline(text)}</i>", st["p"]))
            else:
                flow.append(Paragraph(inline(text), st["p"]))
        pend = None

    while i < len(lines):
        ln = lines[i]
        if ln.startswith("```"):
            flush(); j = i + 1; buf = []
            while j < len(lines) and not lines[j].startswith("```"):
                buf.append(lines[j]); j += 1
            flow.append(Preformatted("\n".join(buf), st["code"])); i = j + 1; continue
        if ln.startswith("|"):
            flush(); rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            n = len(rows[0])
            bad = [r for r in rows if len(r) != n]
            assert not bad, f"malformed table row (cell count != header): {bad[0]}"
            flow += [make_table(rows, width), Spacer(1, 6)]; continue
        m = re.match(r"!\[(.*?)\]\((.+?)\)", ln.strip())
        if m:
            flush(); p = ROOT / m.group(2)
            if p.exists():
                from PIL import Image as PI
                w, h = PI.open(p).size; sc = min(width / w, 9 * cm / h)
                flow += [Image(str(p), w * sc, h * sc), Spacer(1, 6)]
            i += 1; continue
        if ln.startswith("### "):
            flush(); flow.append(Paragraph(inline(ln[4:]), st["h2"]))
        elif ln.startswith("## "):
            flush(); flow.append(Paragraph(inline(ln[3:]), st["h1"]))
        elif ln.startswith("# "):
            flush(); flow.append(Paragraph(inline(ln[2:]), st["title"]))
        elif re.match(r"^[-*] ", ln):
            flush(); pend = ("li", ln[2:], "•")
        elif re.match(r"^\d+\. ", ln):
            flush(); n, t = ln.split(". ", 1); pend = ("li", t, f"{n}.")
        elif ln.strip() in ("---", ""):
            flush()
        elif ln.startswith("> "):
            if pend and pend[0] == "q":
                pend = ("q", pend[1] + " " + ln[2:], None)
            else:
                flush(); pend = ("q", ln[2:], None)
        else:   # plain text: continuation of the current paragraph / list item, or a new paragraph
            if pend:
                pend = (pend[0], pend[1] + " " + ln.strip(), pend[2])
            else:
                pend = ("p", ln.strip(), None)
        i += 1
    flush()
    return flow


readme = (ROOT / "README.md").read_text()
if args.github:
    readme = re.sub(r"<YOUR_GITHUB_REPO_URL>[^\n]*", args.github, readme)
if args.app:
    readme = re.sub(r"<YOUR_STREAMLIT_APP_URL>[^\n]*", args.app, readme)
sections = {m.group(1): m.group(2).strip() for m in re.finditer(r"^## ([a-p])\. .*?\n(.*?)(?=^## [a-p]\. |\Z)", readme, re.S | re.M)}

W = A4[0] - 3.2 * cm
S = []
S.append(Paragraph("Machine Learning – Assignment 2", st["title"]))
S.append(Paragraph("Multiclass classification of dry beans with five classifiers and a Streamlit evaluation app · "
                   "M.Tech Artificial Intelligence &amp; Machine Learning (AIML)", st["sub"]))


def ph(v, label):
    return Paragraph(inline(v) if v else f'<font color="{PH}"><b>[ {label} ]</b></font>', st["cell"])


S.append(Table([[Paragraph("<b>Name</b>", st["cell"]), ph(args.name, "Your full name")],
                [Paragraph("<b>Student ID / Roll no.</b>", st["cell"]), ph(args.student_id, "Your BITS ID")],
                [Paragraph("<b>Email</b>", st["cell"]), ph(args.email, "Your email")],
                [Paragraph("<b>Programme / Course</b>", st["cell"]), Paragraph("M.Tech AIML · Machine Learning", st["cell"])],
                [Paragraph("<b>Submission date</b>", st["cell"]), ph(None, "dd-mm-yyyy")]],
               colWidths=[W * .3, W * .7],
               style=TableStyle([("GRID", (0, 0), (-1, -1), .4, colors.grey), ("TOPPADDING", (0, 0), (-1, -1), 5),
                                 ("BOTTOMPADDING", (0, 0), (-1, -1), 5)])))

# --- order recommended by the assignment guide: GitHub link -> app link -> VM screenshot -> README
S.append(Paragraph("1. GitHub repository link", st["h1"])); S += md_to_flow(sections["d"], W)
S.append(Paragraph("2. Live Streamlit application link", st["h1"])); S += md_to_flow(sections["l"], W)
h3 = Paragraph("3. BITS Virtual Lab / Prayogshala screenshot", st["h1"])
shot = Path(args.screenshot) if args.screenshot else None
if shot and shot.exists():
    from PIL import Image as PI
    w, h = PI.open(shot).size; sc = min(W / w, 12 * cm / h)
    S.append(KeepTogether([h3, Image(str(shot), w * sc, h * sc)]))
else:
    box = Table([[Paragraph(f'<font color="{PH}"><b>[ INSERT YOUR BITS VIRTUAL LAB SCREENSHOT HERE ]</b><br/><br/>'
                            'It must visibly show: the Virtual Lab environment, the live app URL, the uploaded test CSV, the application heading and the results '
                            '(see screenshots/README.md). Re-run this script with --screenshot to embed it.</font>', st["p"])]],
                colWidths=[W], rowHeights=[5.2 * cm],
                style=TableStyle([("BOX", (0, 0), (-1, -1), 1, colors.HexColor(PH)), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                  ("ALIGN", (0, 0), (-1, -1), "CENTER")]))
    S.append(KeepTogether([h3, box]))
S.append(PageBreak())
S.append(Paragraph("4. README / report content", st["h1"]))
S += md_to_flow(readme, W)


def footer(c, d):
    c.saveState(); c.setFont("DV", 7); c.setFillColor(colors.grey)
    c.drawString(1.6 * cm, 1 * cm, "ML Assignment 2 – Dry Bean classification"); c.drawRightString(A4[0] - 1.6 * cm, 1 * cm, f"Page {d.page}")
    c.restoreState()

out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
SimpleDocTemplate(str(out), pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm, topMargin=1.5 * cm, bottomMargin=1.6 * cm,
                  title="ML Assignment 2 - Submission", author=args.name or "Student").build(S, onFirstPage=footer, onLaterPages=footer)
print("wrote", out)
