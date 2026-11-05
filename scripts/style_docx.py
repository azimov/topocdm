"""Style a pandoc-produced .docx for reading. Usage: style_docx.py in.docx out.docx"""
import sys
import docx
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

SRC, DST = sys.argv[1], sys.argv[2]
INK = RGBColor(0x1F, 0x29, 0x33)
ACCENT = RGBColor(0x1F, 0x4E, 0x79)
MUTED = RGBColor(0x52, 0x62, 0x72)
d = docx.Document(SRC)

for s in d.sections:
    s.page_width, s.page_height = Cm(21.0), Cm(29.7)
    s.top_margin = Cm(2.2); s.bottom_margin = Cm(2.2)
    s.left_margin = Cm(2.4); s.right_margin = Cm(2.4)

def set_font(style, name, size, bold=None, color=None):
    f = style.font
    f.name = name; f.size = Pt(size)
    if bold is not None: f.bold = bold
    if color is not None: f.color.rgb = color
    rpr = style.element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts"); rpr.append(rf)
    for a in ("w:ascii", "w:hAnsi", "w:cs"):
        rf.set(qn(a), name)

st = d.styles
set_font(st["Normal"], "Georgia", 10.5, color=INK)
st["Normal"].paragraph_format.line_spacing = 1.28
st["Normal"].paragraph_format.space_after = Pt(6)

for name, size, color, before, after in [("Heading 1", 22, ACCENT, 0, 10),
                                         ("Heading 2", 13.5, ACCENT, 18, 6),
                                         ("Heading 3", 11.5, INK, 12, 4)]:
    s = st[name]
    set_font(s, "Georgia", size, bold=True, color=color)
    s.paragraph_format.space_before = Pt(before)
    s.paragraph_format.space_after = Pt(after)
    s.paragraph_format.keep_with_next = True

set_font(st["Source Code"], "Menlo", 8.5, color=INK)
st["Source Code"].paragraph_format.space_before = Pt(8)
st["Source Code"].paragraph_format.space_after = Pt(10)
st["Source Code"].paragraph_format.line_spacing = 1.0

def shade(p, fill="F4F6F8"):
    pPr = p._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), fill)
    pPr.append(shd)

def hairline(p, colour="C9D1D9"):
    pPr = p._p.get_or_add_pPr()
    bdr = OxmlElement("w:pBdr")
    for side in ("top", "left", "bottom", "right"):
        e = OxmlElement(f"w:{side}")
        e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "4")
        e.set(qn("w:space"), "4"); e.set(qn("w:color"), colour)
        bdr.append(e)
    pPr.append(bdr)

for p in d.paragraphs:
    if p.style.name == "Source Code":
        p.paragraph_format.left_indent = Cm(0.0)
        p.paragraph_format.right_indent = Cm(0.0)
        shade(p); hairline(p)
    elif p.style.name in ("Body Text", "First Paragraph", "Compact"):
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_after = Pt(7)

for p in d.paragraphs[1:3]:
    if p.text.startswith("A specification") or p.text.startswith("A restatement") or p.text.startswith("Written for"):
        for r in p.runs:
            r.italic = True; r.font.size = Pt(10); r.font.color.rgb = MUTED
        p.paragraph_format.space_after = Pt(14)

d.save(DST)
print("styled ->", DST)
