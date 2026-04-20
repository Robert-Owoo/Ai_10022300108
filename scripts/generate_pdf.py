"""
Student Name: Robert George Owoo
Index Number: 10022300108

Converts docs/PROJECT_REPORT.md → docs/PROJECT_REPORT.pdf
using PyMuPDF (fitz) — no external dependencies needed beyond what
FormatFactory's Python already has.

Run from project root:
    "C:\Program Files\FormatFactory\FFModules\python\python.exe" scripts/generate_pdf.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    import fitz
except ImportError:
    sys.exit("PyMuPDF (fitz) not found. Run from FormatFactory Python.")

# ── Page geometry ────────────────────────────────────────────────────────────
PAGE_W, PAGE_H = 595, 842          # A4 portrait points
ML, MR, MT, MB = 55, 55, 60, 55   # margins: left, right, top, bottom
TEXT_W = PAGE_W - ML - MR          # usable width

# ── Colours (r,g,b) 0-1 ────────────────────────────────────────────────────
C_BLACK  = (0.05, 0.05, 0.08)
C_H1     = (0.08, 0.08, 0.10)
C_H2     = (0.10, 0.18, 0.38)
C_H3     = (0.14, 0.30, 0.55)
C_CODE_BG= (0.94, 0.95, 0.96)
C_ROW_ODD= (0.96, 0.97, 1.00)
C_HEAD_BG= (0.20, 0.33, 0.60)
C_HEAD_FG= (1.00, 1.00, 1.00)
C_RULE   = (0.70, 0.73, 0.80)
C_MUTED  = (0.40, 0.42, 0.48)
C_ACCENT = (0.12, 0.24, 0.52)

# ── Font helpers ─────────────────────────────────────────────────────────────
FONT_REGULAR  = "helv"
FONT_BOLD     = "hebo"
FONT_ITALIC   = "heit"
FONT_MONO     = "cour"

STYLE = {
    "h1":   {"font": FONT_BOLD,    "size": 21, "color": C_H1,   "space_before": 18, "space_after": 8},
    "h2":   {"font": FONT_BOLD,    "size": 15, "color": C_H2,   "space_before": 14, "space_after": 5},
    "h3":   {"font": FONT_BOLD,    "size": 12, "color": C_H3,   "space_before": 10, "space_after": 4},
    "body": {"font": FONT_REGULAR, "size": 9.5,"color": C_BLACK, "space_before": 0,  "space_after": 5},
    "bold": {"font": FONT_BOLD,    "size": 9.5,"color": C_BLACK},
    "code": {"font": FONT_MONO,    "size": 8.2,"color": C_BLACK, "space_before": 4,  "space_after": 4},
    "meta": {"font": FONT_ITALIC,  "size": 9,  "color": C_MUTED, "space_before": 0,  "space_after": 3},
    "hr":   {},
    "bullet_indent": 14,
    "line_height_factor": 1.40,
}


class PDFWriter:
    def __init__(self):
        self.doc = fitz.open()
        self.page = None
        self.y = MT
        self._new_page()
        self._page_num = 0

    # ── Page management ──────────────────────────────────────────────────────
    def _new_page(self):
        self.page = self.doc.new_page(width=PAGE_W, height=PAGE_H)
        self.y = MT
        self._page_num = self.doc.page_count

    def _ensure_space(self, needed: float):
        if self.y + needed > PAGE_H - MB:
            self._footer()
            self._new_page()

    def _footer(self):
        n = self._page_num
        self.page.insert_text(
            (PAGE_W / 2 - 15, PAGE_H - 30),
            str(n), fontname=FONT_REGULAR, fontsize=8, color=C_MUTED,
        )

    # ── Drawing primitives ───────────────────────────────────────────────────
    def _rect_fill(self, x0, y0, x1, y1, color):
        self.page.draw_rect(fitz.Rect(x0, y0, x1, y1), color=None, fill=color, width=0)

    def _hline(self, y, color=C_RULE, width=0.5):
        self.page.draw_line(fitz.Point(ML, y), fitz.Point(PAGE_W - MR, y), color=color, width=width)

    # ── Text helpers ─────────────────────────────────────────────────────────
    def _text_height(self, fontsize, lines=1):
        return fontsize * STYLE["line_height_factor"] * lines

    def _wrap_text(self, text: str, font: str, size: float) -> list[str]:
        """Approximate word-wrap into lines that fit TEXT_W."""
        words = text.split()
        if not words:
            return [""]
        lines = []
        current = ""
        for word in words:
            test = (current + " " + word).strip()
            # rough character width: ~0.55 * fontsize per char on proportional fonts
            char_w = size * 0.55 if font != FONT_MONO else size * 0.60
            if len(test) * char_w > TEXT_W - 4:
                if current:
                    lines.append(current)
                current = word
            else:
                current = test
        if current:
            lines.append(current)
        return lines if lines else [""]

    def _insert_line(self, x: float, y: float, text: str,
                     font: str, size: float, color, max_w: float | None = None):
        self.page.insert_text(
            (x, y), text,
            fontname=font, fontsize=size, color=color,
        )

    # ── Block renderers ──────────────────────────────────────────────────────
    def heading(self, text: str, level: int):
        key = f"h{level}"
        st = STYLE[key]
        self._ensure_space(st["space_before"] + self._text_height(st["size"]) + st["space_after"] + 4)
        self.y += st["space_before"]

        # strip markdown inline syntax from heading
        clean = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
        clean = re.sub(r"`(.+?)`", r"\1", clean)

        if level == 1:
            # accent bar
            self._rect_fill(ML, self.y, ML + 4, self.y + self._text_height(st["size"]) + 2, C_ACCENT)
            self.page.insert_text((ML + 10, self.y + st["size"]), clean,
                                   fontname=st["font"], fontsize=st["size"], color=st["color"])
            self.y += self._text_height(st["size"])
            self._hline(self.y + 2, color=C_ACCENT, width=0.8)
            self.y += 6
        elif level == 2:
            self.page.insert_text((ML, self.y + st["size"]), clean,
                                   fontname=st["font"], fontsize=st["size"], color=st["color"])
            self.y += self._text_height(st["size"])
            self._hline(self.y + 1, color=C_H2, width=0.6)
            self.y += 4
        else:
            self.page.insert_text((ML, self.y + st["size"]), clean,
                                   fontname=st["font"], fontsize=st["size"], color=st["color"])
            self.y += self._text_height(st["size"])

        self.y += st["space_after"]

    def paragraph(self, text: str, indent: float = 0, style_key: str = "body"):
        if not text.strip():
            self.y += 4
            return
        st = STYLE[style_key]
        # Strip inline bold/code markers for plain text rendering
        clean = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
        clean = re.sub(r"`(.+?)`", r"\1", clean)
        clean = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", clean)  # links → label

        lines = self._wrap_text(clean, st["font"], st["size"])
        needed = self._text_height(st["size"], len(lines)) + st.get("space_after", 5)
        self._ensure_space(needed + 4)

        for line in lines:
            self.y += self._text_height(st["size"])
            self._insert_line(ML + indent, self.y, line, st["font"], st["size"], st["color"])
        self.y += st.get("space_after", 5)

    def bullet(self, text: str, level: int = 0):
        st = STYLE["body"]
        indent = STYLE["bullet_indent"] * (level + 1)
        clean = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
        clean = re.sub(r"`(.+?)`", r"\1", clean)
        clean = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", clean)

        lines = self._wrap_text(clean, st["font"], st["size"])
        needed = self._text_height(st["size"], len(lines)) + 3
        self._ensure_space(needed)

        bullet_char = "•" if level == 0 else "–"
        first = True
        for line in lines:
            self.y += self._text_height(st["size"])
            if first:
                self._insert_line(ML + indent - 10, self.y, bullet_char,
                                   st["font"], st["size"], C_ACCENT)
                first = False
            self._insert_line(ML + indent, self.y, line, st["font"], st["size"], st["color"])
        self.y += 2

    def numbered(self, text: str, n: int):
        st = STYLE["body"]
        indent = STYLE["bullet_indent"]
        clean = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
        clean = re.sub(r"`(.+?)`", r"\1", clean)
        lines = self._wrap_text(clean, st["font"], st["size"])
        needed = self._text_height(st["size"], len(lines)) + 3
        self._ensure_space(needed)
        first = True
        for line in lines:
            self.y += self._text_height(st["size"])
            if first:
                self._insert_line(ML, self.y, f"{n}.", FONT_BOLD, st["size"], C_ACCENT)
                first = False
            self._insert_line(ML + indent + 4, self.y, line, st["font"], st["size"], st["color"])
        self.y += 2

    def code_block(self, lines: list[str]):
        st = STYLE["code"]
        lh = self._text_height(st["size"])
        pad = 5
        block_h = lh * len(lines) + pad * 2
        self._ensure_space(block_h + st["space_before"] + st["space_after"])
        self.y += st["space_before"]
        # background
        self._rect_fill(ML - 4, self.y, PAGE_W - MR + 4, self.y + block_h, C_CODE_BG)
        self._rect_fill(ML - 4, self.y, ML - 1, self.y + block_h, C_ACCENT)
        self.y += pad
        for line in lines:
            self.y += lh
            # truncate very long lines
            max_chars = int((TEXT_W - 8) / (st["size"] * 0.60))
            display = line[:max_chars] + ("…" if len(line) > max_chars else "")
            self._insert_line(ML + 4, self.y, display, st["font"], st["size"], st["color"])
        self.y += pad + st["space_after"]

    def table(self, headers: list[str], rows: list[list[str]]):
        col_count = len(headers)
        if col_count == 0:
            return
        col_w = TEXT_W / col_count
        row_h = 16
        header_h = 18
        total_h = header_h + row_h * len(rows) + 8
        self._ensure_space(total_h + 10)
        self.y += 4

        # Header row
        self._rect_fill(ML, self.y, PAGE_W - MR, self.y + header_h, C_HEAD_BG)
        for ci, hdr in enumerate(headers):
            x = ML + ci * col_w + 4
            self.page.insert_text(
                (x, self.y + header_h - 5),
                hdr[:int(col_w / 5.5)],
                fontname=FONT_BOLD, fontsize=8, color=C_HEAD_FG,
            )
        self.y += header_h

        for ri, row in enumerate(rows):
            bg = C_ROW_ODD if ri % 2 == 0 else (1, 1, 1)
            self._rect_fill(ML, self.y, PAGE_W - MR, self.y + row_h, bg)
            for ci, cell in enumerate(row):
                x = ML + ci * col_w + 4
                max_chars = int(col_w / 5.5)
                display = str(cell)[:max_chars]
                self.page.insert_text(
                    (x, self.y + row_h - 5),
                    display,
                    fontname=FONT_REGULAR, fontsize=8, color=C_BLACK,
                )
            # cell borders
            for ci in range(col_count + 1):
                lx = ML + ci * col_w
                self.page.draw_line(
                    fitz.Point(lx, self.y),
                    fitz.Point(lx, self.y + row_h),
                    color=C_RULE, width=0.3,
                )
            self._hline(self.y, color=C_RULE, width=0.3)
            self.y += row_h

        # bottom border
        self._hline(self.y, color=C_RULE, width=0.4)
        self.y += 6

    def hr(self):
        self._ensure_space(14)
        self.y += 6
        self._hline(self.y, color=C_RULE, width=0.7)
        self.y += 8

    def spacer(self, h: float = 4):
        self.y += h

    def save(self, path: Path):
        self._footer()  # last page footer
        self.doc.save(str(path))
        print(f"Saved: {path}")


# ── Markdown parser ─────────────────────────────────────────────────────────
def parse_and_render(md_path: Path, writer: PDFWriter):
    text = md_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    in_code = False
    code_lines: list[str] = []
    in_table = False
    table_headers: list[str] = []
    table_rows: list[list[str]] = []
    list_counter = 0

    def flush_table():
        nonlocal in_table, table_headers, table_rows
        if table_headers:
            writer.table(table_headers, table_rows)
        in_table = False
        table_headers = []
        table_rows = []

    i = 0
    while i < len(lines):
        line = lines[i]

        # ── Code fence ──────────────────────────────────────────────────────
        if line.strip().startswith("```"):
            if in_table:
                flush_table()
            if in_code:
                writer.code_block(code_lines)
                code_lines = []
                in_code = False
            else:
                in_code = True
            i += 1
            continue

        if in_code:
            code_lines.append(line)
            i += 1
            continue

        # ── Table rows ──────────────────────────────────────────────────────
        if line.strip().startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            # separator row (all dashes)
            if all(re.match(r"^[-: ]+$", c) for c in cells if c):
                i += 1
                continue
            if not in_table:
                in_table = True
                table_headers = cells
            else:
                table_rows.append(cells)
            i += 1
            continue
        else:
            if in_table:
                flush_table()

        stripped = line.strip()

        # ── Horizontal rule ──────────────────────────────────────────────────
        if re.match(r"^---+$", stripped) or re.match(r"^\*\*\*+$", stripped):
            writer.hr()
            list_counter = 0
            i += 1
            continue

        # ── Headings ─────────────────────────────────────────────────────────
        m = re.match(r"^(#{1,3})\s+(.+)", stripped)
        if m:
            level = len(m.group(1))
            writer.heading(m.group(2), level)
            list_counter = 0
            i += 1
            continue

        # ── Numbered list ────────────────────────────────────────────────────
        m = re.match(r"^(\d+)\.\s+(.+)", stripped)
        if m:
            list_counter += 1
            writer.numbered(m.group(2), int(m.group(1)))
            i += 1
            continue

        # ── Bullet list (any indent level) ───────────────────────────────────
        m = re.match(r"^( *)([-*+])\s+(.+)", line)
        if m:
            level = len(m.group(1)) // 2
            writer.bullet(m.group(3), level)
            list_counter = 0
            i += 1
            continue

        # ── Blank line ───────────────────────────────────────────────────────
        if not stripped:
            writer.spacer(4)
            list_counter = 0
            i += 1
            continue

        # ── Regular paragraph ─────────────────────────────────────────────────
        list_counter = 0
        writer.paragraph(stripped)
        i += 1

    if in_table:
        flush_table()
    if in_code and code_lines:
        writer.code_block(code_lines)


# ── Cover page ───────────────────────────────────────────────────────────────
def cover_page(writer: PDFWriter):
    p = writer.page
    # background band
    writer._rect_fill(0, 0, PAGE_W, 220, C_ACCENT)

    # Institution name
    p.insert_text((ML, 90), "ACADEMIC CITY UNIVERSITY",
                  fontname=FONT_BOLD, fontsize=13, color=(1, 1, 1))
    p.insert_text((ML, 110), "Faculty of Computational Sciences and Informatics",
                  fontname=FONT_REGULAR, fontsize=9.5, color=(0.80, 0.87, 1.0))

    # Title
    p.insert_text((ML, 155), "CS4241 – Introduction to Artificial Intelligence",
                  fontname=FONT_BOLD, fontsize=15, color=(1, 1, 1))
    p.insert_text((ML, 178), "Manual RAG Chatbot — Project Report",
                  fontname=FONT_REGULAR, fontsize=11, color=(0.85, 0.90, 1.0))

    # Details block
    detail_y = 240
    details = [
        ("Student Name:", "Robert George Owoo"),
        ("Index Number:", "10022300108"),
        ("Course:",       "CS4241 – Introduction to Artificial Intelligence"),
        ("Lecturer:",     "Godwin N. Danso"),
        ("Date:",         "April 2026"),
    ]
    for label, value in details:
        p.insert_text((ML, detail_y), label,
                      fontname=FONT_BOLD, fontsize=9.5, color=C_H2)
        p.insert_text((ML + 105, detail_y), value,
                      fontname=FONT_REGULAR, fontsize=9.5, color=C_BLACK)
        detail_y += 18

    # Horizontal separator
    p.draw_line(fitz.Point(ML, detail_y + 8), fitz.Point(PAGE_W - MR, detail_y + 8),
                color=C_RULE, width=0.6)

    # Abstract box
    abs_y = detail_y + 24
    writer._rect_fill(ML - 4, abs_y, PAGE_W - MR + 4, abs_y + 115, C_CODE_BG)
    writer._rect_fill(ML - 4, abs_y, ML - 1, abs_y + 115, C_ACCENT)
    p.insert_text((ML + 6, abs_y + 14), "Abstract",
                  fontname=FONT_BOLD, fontsize=10, color=C_ACCENT)
    abstract = (
        "This report documents the design and implementation of a manual "
        "Retrieval-Augmented Generation (RAG) chat assistant for Academic City. "
        "The system answers questions over two datasets — Ghana election results "
        "(CSV) and the 2025 MOFEP Budget Statement (PDF) — without using "
        "LangChain, LlamaIndex, or any pre-built RAG pipeline. All core "
        "components (chunking, embedding, BM25, hybrid retrieval, prompt "
        "construction, logging, and a feedback loop) were built from scratch."
    )
    # word-wrap abstract manually
    words = abstract.split()
    abs_line = ""
    abs_wy = abs_y + 30
    for w in words:
        test = (abs_line + " " + w).strip()
        if len(test) * 4.9 > TEXT_W - 16:
            p.insert_text((ML + 6, abs_wy), abs_line,
                          fontname=FONT_REGULAR, fontsize=8.8, color=C_BLACK)
            abs_wy += 13
            abs_line = w
        else:
            abs_line = test
    if abs_line:
        p.insert_text((ML + 6, abs_wy), abs_line,
                      fontname=FONT_REGULAR, fontsize=8.8, color=C_BLACK)

    # move writer y past cover content so next page starts clean
    writer._footer()
    writer._new_page()


# ── Entry point ──────────────────────────────────────────────────────────────
def main():
    root = Path(__file__).resolve().parents[1]
    md_path = root / "docs" / "PROJECT_REPORT.md"
    out_path = root / "docs" / "PROJECT_REPORT.pdf"

    if not md_path.exists():
        sys.exit(f"Markdown file not found: {md_path}")

    writer = PDFWriter()
    cover_page(writer)
    parse_and_render(md_path, writer)
    writer.save(out_path)
    print(f"Done! Open: {out_path}")


if __name__ == "__main__":
    main()
