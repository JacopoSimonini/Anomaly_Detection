"""
Build presentation.pptx for the Water-Pump Audio Anomaly Detection project.

Style imitates presentation-template.pdf (cream background, bold headings,
lavender section chips, big stat tiles, 3-card rows, two-up stats, blank dividers).
Charts are pulled from images/. Run:  python build_deck.py
"""
from pathlib import Path
from PIL import Image

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn

ROOT = Path(__file__).resolve().parent
IMG = ROOT / "images"
OUT = ROOT / "presentation.pptx"

# ---- design system (sampled from template) -------------------------------
CREAM = RGBColor(0xFF, 0xFA, 0xFA)
INK = RGBColor(0x1A, 0x1A, 0x1A)
GRAY = RGBColor(0x44, 0x44, 0x44)
MUTE = RGBColor(0x77, 0x77, 0x77)
LAV = RGBColor(0xD5, 0xDC, 0xF6)
BLUE = RGBColor(0x60, 0x94, 0xEA)
RED = RGBColor(0xE2, 0x33, 0x43)
GREEN = RGBColor(0x2E, 0x9E, 0x5B)
CARD_BORDER = RGBColor(0xCF, 0xD3, 0xEA)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT = "Poppins"          # geometric sans; PowerPoint substitutes if missing
FONT_BODY = "Poppins"

EMU_IN = 914400
SW, SH = 13.333, 7.5      # slide size (inches, 16:9)

prs = Presentation()
prs.slide_width = Inches(SW)
prs.slide_height = Inches(SH)
BLANK = prs.slide_layouts[6]


# ---- primitives ----------------------------------------------------------
def new_slide():
    s = prs.slides.add_slide(BLANK)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = CREAM
    return s


def _set_font(run, size, color, bold=False, font=FONT_BODY, italic=False):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.name = font
    run.font.color.rgb = color


def textbox(slide, l, t, w, h, lines, align=PP_ALIGN.LEFT,
            anchor=MSO_ANCHOR.TOP, wrap=True):
    """lines: list of dicts or list of [runs]. Each paragraph = list of run-dicts
    {text,size,color,bold,font,italic,space_after,space_before,line}."""
    tb = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = wrap
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, para in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = para.get("align", align)
        if "space_after" in para:
            p.space_after = Pt(para["space_after"])
        if "space_before" in para:
            p.space_before = Pt(para["space_before"])
        if "line" in para:
            p.line_spacing = para["line"]
        for r in para["runs"]:
            run = p.add_run()
            run.text = r["text"]
            _set_font(run, r.get("size", 18), r.get("color", GRAY),
                      r.get("bold", False), r.get("font", FONT_BODY),
                      r.get("italic", False))
    return tb


def chip(slide, text):
    w, h = 0.2 + 0.11 * len(text), 0.36
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                 Inches(0.62), Inches(0.5), Inches(w), Inches(h))
    shp.fill.solid()
    shp.fill.fore_color.rgb = LAV
    shp.line.fill.background()
    shp.shadow.inherit = False
    tf = shp.text_frame
    tf.margin_left = tf.margin_right = Inches(0.05)
    tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = text.upper()
    _set_font(r, 11, RGBColor(0x3A, 0x3A, 0x55), bold=True, font=FONT)
    return shp


def header(slide, chip_text, title, sub=None):
    if chip_text:
        chip(slide, chip_text)
    paras = [{"runs": [{"text": title, "size": 33, "color": INK, "bold": True,
                        "font": FONT}], "line": 1.02}]
    textbox(slide, 0.62, 1.0, 12.1, 1.2, paras)
    if sub:
        textbox(slide, 0.64, 1.78, 12.0, 0.5,
                [{"runs": [{"text": sub, "size": 15, "color": MUTE}]}])


def add_image_fit(slide, path, l, t, max_w, max_h, align="center", valign="center"):
    iw, ih = Image.open(path).size
    ratio = min(max_w / iw, max_h / ih)
    w, h = iw * ratio, ih * ratio
    lx = l + (max_w - w) / 2 if align == "center" else l
    ty = t if valign == "top" else t + (max_h - h) / 2
    return slide.shapes.add_picture(str(path), Inches(lx), Inches(ty),
                                    Inches(w), Inches(h))


def rrect(slide, l, t, w, h, fill=WHITE, line=CARD_BORDER, line_w=1.0):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                 Inches(l), Inches(t), Inches(w), Inches(h))
    shp.fill.solid(); shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line; shp.line.width = Pt(line_w)
    shp.shadow.inherit = False
    return shp


def rect(slide, l, t, w, h, fill, line=None):
    shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                                 Inches(l), Inches(t), Inches(w), Inches(h))
    shp.fill.solid(); shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
    shp.shadow.inherit = False
    return shp


# ---- composite archetypes ------------------------------------------------
def slide_title(title, subtitle, author, course):
    s = new_slide()
    textbox(s, 1.0, 1.85, 11.33, 1.9,
            [{"runs": [{"text": title, "size": 46, "color": INK, "bold": True,
                        "font": FONT}], "align": PP_ALIGN.CENTER, "line": 1.0}])
    textbox(s, 1.0, 4.0, 11.33, 0.9,
            [{"runs": [{"text": subtitle, "size": 22, "color": GRAY,
                        "font": FONT}], "align": PP_ALIGN.CENTER, "line": 1.05}])
    textbox(s, 1.0, 5.5, 11.33, 1.0,
            [{"runs": [{"text": author, "size": 15, "color": MUTE}],
              "align": PP_ALIGN.CENTER, "space_after": 6},
             {"runs": [{"text": course, "size": 15, "color": MUTE}],
              "align": PP_ALIGN.CENTER}])
    return s


def slide_divider(title):
    s = new_slide()
    textbox(s, 0.8, 0, 11.73, SH,
            [{"runs": [{"text": title, "size": 40, "color": INK, "bold": True,
                        "font": FONT}], "align": PP_ALIGN.CENTER}],
            anchor=MSO_ANCHOR.MIDDLE)
    return s


def slide_cards(chip_text, title, sub, cards):
    s = new_slide()
    header(s, chip_text, title, sub)
    n = len(cards)
    gap, margin = 0.5, 0.62
    total_w = SW - 2 * margin
    cw = (total_w - gap * (n - 1)) / n
    top, ch = 2.7, 3.4
    for i, (ct, cd) in enumerate(cards):
        l = margin + i * (cw + gap)
        rrect(s, l, top, cw, ch)
        textbox(s, l + 0.25, top + 0.45, cw - 0.5, 1.0,
                [{"runs": [{"text": ct, "size": 20, "color": INK, "bold": True,
                            "font": FONT}], "align": PP_ALIGN.CENTER, "line": 1.05}],
                anchor=MSO_ANCHOR.TOP)
        textbox(s, l + 0.3, top + 1.7, cw - 0.6, ch - 1.9,
                [{"runs": [{"text": cd, "size": 14.5, "color": GRAY}],
                  "align": PP_ALIGN.CENTER, "line": 1.15}], anchor=MSO_ANCHOR.TOP)
    return s


def slide_diagnostics_recap(chip_text, title, sub, checks, verdict):
    s = new_slide()
    header(s, chip_text, title, sub)
    margin, gap, row_gap = 0.62, 0.4, 0.3
    cw = (SW - 2 * margin - gap) / 2
    top, ch = 2.05, 1.65
    for i, (name, desc) in enumerate(checks):
        l = margin + (i % 2) * (cw + gap)
        t = top + (i // 2) * (ch + row_gap)
        rrect(s, l, t, cw, ch)
        textbox(s, l + 0.3, t + 0.18, cw - 0.6, 0.4,
                [{"runs": [{"text": "✓  ", "size": 18, "color": GREEN,
                            "bold": True},
                           {"text": name, "size": 18, "color": INK,
                            "bold": True, "font": FONT}], "line": 1.05}])
        textbox(s, l + 0.3, t + 0.62, cw - 0.6, ch - 0.75,
                [{"runs": [{"text": desc, "size": 13.5, "color": GRAY}],
                  "line": 1.15}], anchor=MSO_ANCHOR.TOP)
    band_top = top + 2 * ch + row_gap + 0.25
    rrect(s, margin, band_top, SW - 2 * margin, 0.95,
          fill=RGBColor(0xEA, 0xF0, 0xFC), line=BLUE)
    textbox(s, margin + 0.4, band_top, SW - 2 * margin - 0.8, 0.95,
            [{"runs": [{"text": "⟶  ", "size": 16, "color": BLUE,
                        "bold": True},
                       {"text": verdict, "size": 16, "color": INK,
                        "bold": True, "font": FONT}],
              "align": PP_ALIGN.CENTER, "line": 1.1}],
            anchor=MSO_ANCHOR.MIDDLE)
    return s


def slide_stat_tiles(chip_text, title, sub, tiles, figure=None, note=None):
    s = new_slide()
    header(s, chip_text, title, sub)
    n = len(tiles)
    margin = 0.62
    total_w = SW - 2 * margin
    tw = total_w / n
    top = 2.15
    for i, (num, lab) in enumerate(tiles):
        l = margin + i * tw
        textbox(s, l, top, tw, 0.7,
                [{"runs": [{"text": num, "size": 34, "color": INK, "bold": True,
                            "font": FONT}], "align": PP_ALIGN.CENTER}])
        textbox(s, l, top + 0.72, tw, 0.7,
                [{"runs": [{"text": lab, "size": 13, "color": GRAY}],
                  "align": PP_ALIGN.CENTER, "line": 1.05}])
    if figure:
        add_image_fit(s, figure, 0.62, 3.35, 12.1, 3.45)
    if note:
        textbox(s, 0.62, 6.9, 12.1, 0.5,
                [{"runs": [{"text": note, "size": 13, "color": MUTE,
                            "italic": True}], "align": PP_ALIGN.CENTER}])
    return s


def slide_chart_caption(chip_text, title, figure, cap_head, cap_bullets,
                        img_side="left"):
    s = new_slide()
    header(s, chip_text, title)
    iw, ih = Image.open(figure).size
    aspect = iw / ih
    head_para = {"runs": [{"text": cap_head, "size": 21, "color": INK,
                           "bold": True, "font": FONT}], "line": 1.05}

    def bullets(size):
        return [{"runs": [{"text": "•  ", "size": size, "color": BLUE,
                           "bold": True},
                          {"text": b, "size": size, "color": GRAY}],
                 "space_after": 7, "line": 1.12} for b in cap_bullets]

    if aspect >= 2.4:
        # wide figure: span the full slide width, caption band underneath
        # (caption band aligned to the image's actual horizontal extent)
        pic = add_image_fit(s, figure, 0.62, 1.9, 12.1, 3.6, valign="top")
        pw = pic.width / EMU_IN
        px = 0.62 + (12.1 - pw) / 2
        cap_top = 1.9 + pic.height / EMU_IN + 0.2
        cap_h = 7.25 - cap_top
        # narrow images leave the aligned band too cramped — use full width
        bx, bw = (px, pw) if pw >= 10 else (0.62, 12.1)
        textbox(s, bx, cap_top, 3.3, cap_h, [head_para],
                anchor=MSO_ANCHOR.MIDDLE)
        textbox(s, bx + 3.55, cap_top, bw - 3.55, cap_h, bullets(14.5),
                anchor=MSO_ANCHOR.MIDDLE)
    else:
        # squarer figure: side-by-side, image on the left margin,
        # caption centered in the remaining space
        img_top, max_h, max_w, gap = 2.1, 4.95, 8.0, 0.45
        w = min(max_w, max_h * aspect)
        h = w / aspect
        rem = SW - 0.62 - (0.62 + w + gap)
        cap_w = min(4.35, rem)
        cx = 0.62 + w + gap + (rem - cap_w) / 2
        s.shapes.add_picture(str(figure), Inches(0.62),
                             Inches(img_top + (max_h - h) / 2),
                             Inches(w), Inches(h))
        head_para["space_after"] = 12
        textbox(s, cx, img_top, cap_w, max_h,
                [head_para] + bullets(15), anchor=MSO_ANCHOR.MIDDLE)
    return s


def slide_two_up(chip_text, title, left, right, footnote):
    s = new_slide()
    header(s, chip_text, title)
    # vertical divider
    ln = slide_line = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(6.63),
                                         Inches(3.0), Pt(1.4), Inches(2.5))
    ln.fill.solid(); ln.fill.fore_color.rgb = RGBColor(0x22, 0x2A, 0x45)
    ln.line.fill.background(); ln.shadow.inherit = False
    for (num, lab, sub, color), x in ((left, 0.62), (right, 6.9)):
        textbox(s, x, 3.05, 5.8, 1.3,
                [{"runs": [{"text": num, "size": 66, "color": color, "bold": True,
                            "font": FONT}], "align": PP_ALIGN.CENTER}])
        textbox(s, x, 4.45, 5.8, 0.8,
                [{"runs": [{"text": lab, "size": 20, "color": INK, "bold": True,
                            "font": FONT}], "align": PP_ALIGN.CENTER, "line": 1.05}])
        textbox(s, x, 5.25, 5.8, 0.6,
                [{"runs": [{"text": sub, "size": 13, "color": MUTE}],
                  "align": PP_ALIGN.CENTER}])
    if footnote:
        textbox(s, 0.62, 6.75, 12.1, 0.5,
                [{"runs": [{"text": footnote, "size": 13, "color": MUTE,
                            "italic": True}], "align": PP_ALIGN.CENTER}])
    return s


def slide_two_columns(chip_text, title, sub, left, right):
    """left/right = (header, color, [items])"""
    s = new_slide()
    header(s, chip_text, title, sub)
    top, ch = 2.75, 4.0
    for (hd, color, items), l in ((left, 0.62), (right, 6.95)):
        cw = 5.75
        rrect(s, l, top, cw, ch)
        bar = rect(s, l, top, cw, 0.72, color)
        bar.adjustments  # keep top corners; acceptable rounded fallback
        textbox(s, l + 0.3, top + 0.14, cw - 0.6, 0.5,
                [{"runs": [{"text": hd, "size": 17, "color": WHITE, "bold": True,
                            "font": FONT}]}])
        paras = []
        for it in items:
            paras.append({"runs": [{"text": "•  ", "size": 15, "color": color,
                                    "bold": True},
                                   {"text": it, "size": 15, "color": GRAY}],
                          "space_after": 10, "line": 1.1})
        textbox(s, l + 0.32, top + 1.0, cw - 0.6, ch - 1.2, paras)
    return s


def slide_conclusion(chip_text, title, ach, fut, figure):
    s = new_slide()
    header(s, chip_text, title)
    if figure and figure.exists():
        add_image_fit(s, figure, 0.62, 1.85, 12.1, 3.4)
    col_top = 5.5
    for head, color, items, x, w in (("Achieved", BLUE, ach, 0.62, 6.2),
                                     ("Future Work", RED, fut, 7.1, 5.6)):
        paras = [{"runs": [{"text": head, "size": 18, "color": color,
                            "bold": True, "font": FONT}], "space_after": 8}]
        for b in items:
            paras.append({"runs": [{"text": "•  ", "size": 14, "color": color,
                                    "bold": True},
                                   {"text": b, "size": 14, "color": GRAY}],
                          "space_after": 5, "line": 1.1})
        textbox(s, x, col_top, w, 1.8, paras)
    return s


def slide_results_table(chip_text, title, figure, rows):
    """rows: list of (method, balacc, auc, kind) kind in {win,unsup,ext,sup}"""
    s = new_slide()
    header(s, chip_text, title)
    img_w, gap, tw = 4.6, 0.55, 5.45
    lx = (SW - (img_w + gap + tw)) / 2
    add_image_fit(s, figure, lx, 2.35, img_w, 4.6)
    # table on the right
    tl, tt = lx + img_w + gap, 2.45
    n = len(rows) + 1
    rh = 0.44
    from pptx.util import Inches as I
    tbl_shape = s.shapes.add_table(n, 3, I(tl), I(tt), I(tw), I(rh * n))
    table = tbl_shape.table
    table.columns[0].width = I(3.35)
    table.columns[1].width = I(1.05)
    table.columns[2].width = I(1.05)
    hdr = ["Method", "Bal Acc", "AUC"]
    for c, htxt in enumerate(hdr):
        cell = table.cell(0, c)
        cell.fill.solid(); cell.fill.fore_color.rgb = RGBColor(0x2A, 0x2A, 0x3A)
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        cell.margin_top = cell.margin_bottom = Pt(1)
        p = cell.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER
        r = p.add_run(); r.text = htxt
        _set_font(r, 10.5, WHITE, bold=True, font=FONT)
    palette = {"win": (RGBColor(0xE7, 0xEF, 0xFD), INK, True),
               "unsup": (WHITE, GRAY, False),
               "ext": (RGBColor(0xF4, 0xF4, 0xF7), MUTE, False),
               "sup": (RGBColor(0xF4, 0xF4, 0xF7), MUTE, False)}
    for ri, (m, ba, au, kind) in enumerate(rows, start=1):
        fill, fg, bold = palette[kind]
        vals = [m, ba, au]
        for c in range(3):
            cell = table.cell(ri, c)
            cell.fill.solid(); cell.fill.fore_color.rgb = fill
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.margin_left = cell.margin_right = Pt(4)
            cell.margin_top = cell.margin_bottom = Pt(1)
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER
            r = p.add_run(); r.text = vals[c]
            col = BLUE if (kind == "win" and c > 0) else fg
            _set_font(r, 9.5, col, bold=(bold or (kind == "win")), font=FONT)
    textbox(s, tl, tt + rh * n + 0.08, tw, 0.5,
            [{"runs": [{"text": "Grey rows use anomaly labels (diagnostic baselines).",
                        "size": 10.5, "color": MUTE, "italic": True}]}])
    return s


def slide_architecture(chip_text, title):
    s = new_slide()
    header(s, chip_text, title)
    # encoder blocks (decreasing), bottleneck, decoder blocks (increasing)
    enc = [("32", 2.3), ("64", 1.9), ("128", 1.5), ("256", 1.1)]
    y_mid = 3.75
    x = 0.9
    bw = 0.95
    gap = 0.28
    for lab, bh in enc:
        rrect(s, x, y_mid - bh / 2, bw, bh, fill=RGBColor(0xE9, 0xEE, 0xFB),
              line=BLUE, line_w=1.25)
        textbox(s, x, y_mid - 0.18, bw, 0.4,
                [{"runs": [{"text": lab, "size": 14, "color": INK, "bold": True,
                            "font": FONT}], "align": PP_ALIGN.CENTER}])
        x += bw + gap
    # bottleneck
    rrect(s, x, y_mid - 0.55, 1.15, 1.1, fill=BLUE, line=None)
    textbox(s, x, y_mid - 0.28, 1.15, 0.6,
            [{"runs": [{"text": "z∈ℝ⁶⁴", "size": 14, "color": WHITE, "bold": True,
                        "font": FONT}], "align": PP_ALIGN.CENTER}])
    x += 1.15 + gap
    for lab, bh in reversed(enc):
        rrect(s, x, y_mid - bh / 2, bw, bh, fill=RGBColor(0xF3, 0xE9, 0xEC),
              line=RED, line_w=1.25)
        textbox(s, x, y_mid - 0.18, bw, 0.4,
                [{"runs": [{"text": lab, "size": 14, "color": INK, "bold": True,
                            "font": FONT}], "align": PP_ALIGN.CENTER}])
        x += bw + gap
    textbox(s, 0.9, y_mid + 1.35, 5.3, 0.4,
            [{"runs": [{"text": "Encoder  (spectrogram → latent)", "size": 13,
                        "color": BLUE, "bold": True, "font": FONT}],
              "align": PP_ALIGN.CENTER}])
    textbox(s, 7.1, y_mid + 1.35, 5.3, 0.4,
            [{"runs": [{"text": "Decoder  (latent → reconstruction)", "size": 13,
                        "color": RED, "bold": True, "font": FONT}],
              "align": PP_ALIGN.CENTER}])
    # stat tiles below
    tiles = [("8192 → 64", "input values → latent"),
             ("128×", "compression ratio"),
             ("NORMAL only", "training data")]
    tw = (SW - 1.24) / 3
    for i, (num, lab) in enumerate(tiles):
        l = 0.62 + i * tw
        textbox(s, l, 6.05, tw, 0.55,
                [{"runs": [{"text": num, "size": 24, "color": INK, "bold": True,
                            "font": FONT}], "align": PP_ALIGN.CENTER}])
        textbox(s, l, 6.62, tw, 0.4,
                [{"runs": [{"text": lab, "size": 12.5, "color": MUTE}],
                  "align": PP_ALIGN.CENTER}])
    return s


def slide_pipeline(chip_text, title, steps, figure):
    s = new_slide()
    header(s, chip_text, title)
    n = len(steps)
    gap, margin = 0.35, 0.62
    bw = (SW - 2 * margin - gap * (n - 1)) / n
    top, bh = 2.1, 1.05
    for i, st in enumerate(steps):
        l = margin + i * (bw + gap)
        rrect(s, l, top, bw, bh, fill=RGBColor(0xEF, 0xF2, 0xFB), line=CARD_BORDER)
        textbox(s, l + 0.18, top, 0.55, bh,
                [{"runs": [{"text": str(i + 1), "size": 24, "color": BLUE,
                            "bold": True, "font": FONT}]}], anchor=MSO_ANCHOR.MIDDLE)
        textbox(s, l + 0.78, top, bw - 0.95, bh,
                [{"runs": [{"text": st, "size": 13, "color": INK, "font": FONT}],
                  "line": 1.08}], anchor=MSO_ANCHOR.MIDDLE)
    add_image_fit(s, figure, 0.62, 3.5, 12.1, 3.55)
    return s


def slide_closing(text, sub):
    s = new_slide()
    textbox(s, 0.8, 0, 11.73, SH,
            [{"runs": [{"text": text, "size": 44, "color": INK, "bold": True,
                        "font": FONT}], "align": PP_ALIGN.CENTER, "space_after": 10},
             {"runs": [{"text": sub, "size": 16, "color": MUTE}],
              "align": PP_ALIGN.CENTER}], anchor=MSO_ANCHOR.MIDDLE)
    return s


# =========================================================================
# BUILD
# =========================================================================
def build():
    # 1 title
    slide_title(
        "Audio Anomaly Detection on Water Pumps",
        "Learning a ConvAE Latent Space for Unsupervised Fault Detection",
        "Jacopo Simonini",
        "Digital Forensics and Biometrics — A.A. 2025/2026")

    # 2 motivation cards
    slide_cards("Motivation", "What is Acoustic Anomaly Detection?", None, [
        ("Condition Monitoring",
         "Detect machine faults early, directly from the sound they emit."),
        ("Unsupervised / One-Class",
         "Real deployments only have healthy audio; faults are rare and unlabeled."),
        ("Represent → Detect",
         "First learn a feature space, then flag samples that fall outside it."),
    ])

    # 3 dataset stat tiles
    slide_stat_tiles("Dataset", "Water-Pump Audio Dataset", None, [
        ("4", "machine IDs (00 / 02 / 04 / 06)"),
        ("16 kHz", "mono audio"),
        ("9", "windows per 10 s clip"),
        ("200 / 320", "normal / anomaly test files"),
    ], figure=IMG / "sample_log_mel_spectrograms.png",
        note="ID 06 is never seen by the AE or the detectors — it appears only in tuning and test data. File-level splits, seed 42.")

    # 4 divider
    slide_divider("Data & Preprocessing")

    # 5 preprocessing pipeline
    slide_pipeline("Preprocessing", "From Waveform to Spectrogram", [
        "Load audio — mono, 16 kHz",
        "Slice — 2 s windows, 1 s hop (9 / clip)",
        "Log-mel spectrogram — 128 × 64",
        "Normalize to [0, 1]",
    ], IMG / "sample_log_mel_spectrograms.png")

    # 6 dB normalization
    slide_chart_caption("Preprocessing", "A Normalization Choice That Matters",
        IMG / "db_distribution_fixed_reference.png",
        "Fixed reference preserves loudness", [
            "Convert to dB with ref = 1.0, not per-image max.",
            "A per-image max would erase absolute intensity across files.",
            "Clip to 1st / 99th percentile (≈ −46 / −10 dB), then map to [0, 1].",
        ])

    # 7 divider
    slide_divider("The ConvAE Representation")

    # 8 architecture
    slide_architecture("Model", "Convolutional Autoencoder")

    # 9 training history
    slide_chart_caption("Model", "Training Converges Cleanly",
        IMG / "convae_training_history.png", "Clean convergence", [
            "52 epochs, best validation MSE ≈ 0.0073.",
            "Early stopping + reduce-on-plateau scheduler.",
            "Train and validation track closely — no training failure.",
        ])

    # 10 reconstruction quality
    slide_chart_caption("Model", "Reconstruction Quality",
        IMG / "convae_reconstruction_examples.png", "The AE rebuilds spectrograms well", [
            "Faithful reconstructions on normal windows, as expected.",
            "But it also reconstructs many anomalies acceptably…",
            "…foreshadowing why plain reconstruction error is a weak score.",
        ])

    # 11 divider
    slide_divider("Detecting Anomalies")

    # 12 two families
    slide_two_columns("Methodology", "Two Families of Detectors",
        "All operate on the frozen ConvAE (latent vector or reconstruction error).",
        ("Unsupervised — the real claim", BLUE, [
            "Reconstruction Error (per-window MSE)",
            "Isolation Forest (latent)",
            "Local Outlier Factor — LOF (latent)",
            "One-Class SVM (latent)",
            "No anomaly labels used.",
        ]),
        ("Supervised — diagnostic baselines", RED, [
            "SVM-RBF (latent)",
            "XGBoost (latent)",
            "SVM on MFCC statistics",
            "Use anomaly labels in training…",
            "…so interpreted as baselines, not the headline.",
        ]))

    # 13 three method cards
    slide_cards("Methodology", "Detectors on the Latent Space", None, [
        ("One-Class SVM", "Learns a single rigid boundary around normal data."),
        ("Local Outlier Factor", "Density-based: flags points in low-density regions."),
        ("Isolation Forest", "Isolates outliers with random splits in feature space."),
    ])

    # 14 thresholding
    slide_chart_caption("Methodology", "Thresholding & Evaluation",
        IMG / "convae_threshold_tuning_distributions.png",
        "Honest, leakage-free protocol", [
            "Window scores → file scores by mean aggregation.",
            "Threshold tuned only on the tuning split (balanced accuracy).",
            "Final test untouched; report AUC-ROC / PR (threshold-free).",
        ])

    # 15 divider
    slide_divider("Results & Analysis")

    # 16 winner two-up
    slide_two_up("Results", "Winner: ConvAE + LOF",
        ("73.3%", "Balanced Accuracy", "held-out file-level test", BLUE),
        ("0.788", "AUC-ROC", "threshold-independent", RED),
        "Best assignment-aligned unsupervised method.")

    # 17 full comparison
    slide_results_table("Results", "All Methods Compared",
        IMG / "all_methods_roc_comparison.png", [
            ("ConvAE Recon Error", "0.595", "0.646", "unsup"),
            ("ConvAE + IF", "0.560", "0.601", "unsup"),
            ("ConvAE + LOF", "0.733", "0.788", "win"),
            ("One-Class SVM", "0.545", "0.615", "unsup"),
            ("PANNs + LOF", "0.702", "0.761", "ext"),
            ("SVM-sup (latent)", "0.827", "0.953", "sup"),
            ("XGBoost (latent)", "0.827", "0.941", "sup"),
            ("SVM-sup (MFCC)", "0.854", "0.934", "sup"),
        ])

    # 18 score distributions
    slide_chart_caption("Results", "Why Reconstruction Error Is Weak",
        IMG / "convae_final_score_distributions.png",
        "Overlap vs separation", [
            "Reconstruction-error and IF scores overlap heavily.",
            "LOF separates normal and anomaly files best.",
            "Average pixelwise MSE misses local, textural faults.",
        ])

    # 19 per-machine diagnosis
    slide_chart_caption("Diagnosis", "The ID 06 Problem",
        IMG / "per_machine_error_rates.png",
        "One unseen machine drives false alarms", [
            "ID 06 LOF false-positive rate: 51.1%.",
            "Other IDs: 14.5% / 10.2% / 16.3%.",
            "The model confuses a new normal machine with an anomaly.",
        ])

    # 20 per-machine score distributions (evidence for the ID 06 diagnosis)
    slide_chart_caption("Diagnosis", "Per-Machine Score Distributions",
        IMG / "per_machine_score_distributions.png",
        "The latent space is the culprit", [
            "ID 06 normal LOF scores straddle the threshold — the 51% FPR, visually.",
            "For recon error, ID 06 normals sit below threshold (lowest FPR).",
            "The ID 06 problem lives in the latent-density model, not reconstruction.",
        ])

    # 21 protocol 2
    slide_two_up("Diagnosis", "Ablation — Add ID 06 Normals (Protocol 2)",
        ("51% → 40%", "ID 06 false-positive rate", "lower is better", BLUE),
        ("0.79 → 0.81", "LOF AUC-ROC", "slight overall gain", GREEN),
        "ID 06 normals taken from the tuning split — test set unchanged. FNR rises 31.9% → 34.8%: "
        "overlap remains, so unseen-machine variation is only a partial cause.")

    # 22 t-SNE (optional — delete to reach 24 slides)
    slide_chart_caption("Diagnosis", "The Latent Space Is Only Partly Separable",
        IMG / "tsne_embedding_visualization.png",
        "Qualitative view", [
            "t-SNE of ConvAE latents on the test set.",
            "Normal and anomaly points partially overlap.",
            "Consistent with the moderate unsupervised scores.",
        ])

    # 23 diagnostics recap — the dataset-quality verdict
    slide_diagnostics_recap("Diagnosis", "A Diagnostics Recap",
        None, [
            ("Normalization",
             "Fixed dB ref = 1.0, clipped to the training 1st / 99th percentiles "
             "(−46 / −10 dB) — absolute loudness preserved, verified on the "
             "training distribution."),
            ("Tuning & aggregation",
             "4 aggregation strategies × 3 tuning metrics swept — mean is the "
             "most stable. Thresholds tuned on a separate split, no leakage."),
            ("MFCC sanity baseline",
             "With anomaly labels, an SVM on hand-crafted MFCC statistics "
             "reaches 0.854 BA (AUC 0.93) — the signal exists; the "
             "unsupervised gap is not missing information."),
            ("Unseen-machine ablation",
             "Adding ID 06 normals to training helps only partially "
             "(FPR 51% → 40%, FNR rises) — genuine overlap remains."),
        ],
        "The signal exists — but without labels, subtle faults overlap "
        "with normal machine variation.")

    # 24 conclusion
    slide_conclusion("Conclusion", "Conclusion & Future Work",
        ["ConvAE latent space + LOF: 0.733 BA / 0.788 AUC, fully unsupervised.",
         "Reconstruction error alone is insufficient for subtle pump faults.",
         "Diagnosed the unseen-machine (ID 06) false-positive effect."],
        ["Richer temporal aggregation beyond a simple mean.",
         "Disentangle machine identity from fault information.",
         "Contrastive / alternative reconstruction objectives."],
        IMG / "sample_log_mel_spectrograms.png")

    # 25 thank you
    slide_closing("Thank you.", "Questions?")

    prs.save(str(OUT))
    print(f"Wrote {OUT}  ({len(prs.slides.__iter__.__self__._sldIdLst)} slides)")


if __name__ == "__main__":
    build()
