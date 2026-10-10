"""Small ReportLab-compatible PDF canvas used when ReportLab is unavailable.

It implements only the vector drawing operations used by
plot_expanded_rl_figures.py and writes portable one-page PDFs with core fonts.
"""
from pathlib import Path


def _pdf_text(value):
    value = str(value).encode("ascii", "replace").decode("ascii")
    return "(" + value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)") + ")"


class Canvas:
    def __init__(self, filename, pagesize=(612, 792)):
        self.filename = Path(filename)
        self.width, self.height = pagesize
        self.ops = []
        self.font = ("Helvetica", 12)
        self.fill = (0, 0, 0)
        self.stroke = (0, 0, 0)
        self.line_width = 1

    def setTitle(self, title):
        self.title = str(title)

    def setFont(self, name, size):
        self.font = (name, size)

    def setFillColorRGB(self, r, g, b):
        self.fill = (r, g, b)
        self.ops.append(f"{r:g} {g:g} {b:g} rg")

    def setStrokeColorRGB(self, r, g, b):
        self.stroke = (r, g, b)
        self.ops.append(f"{r:g} {g:g} {b:g} RG")

    def setLineWidth(self, width):
        self.line_width = width
        self.ops.append(f"{width:g} w")

    def _text_width(self, text):
        # Approximation is sufficient for centering labels in these charts.
        return len(str(text)) * self.font[1] * 0.52

    def _text(self, x, y, text):
        font, size = self.font
        self.ops.append(
            f"BT /{font} {size:g} Tf {x:g} {y:g} Td {_pdf_text(text)} Tj ET"
        )

    def drawString(self, x, y, text):
        self._text(x, y, text)

    def drawCentredString(self, x, y, text):
        self._text(x - self._text_width(text) / 2, y, text)

    def drawRightString(self, x, y, text):
        self._text(x - self._text_width(text), y, text)

    def rect(self, x, y, width, height, fill=0, stroke=1):
        self.ops.append(
            f"{x:g} {y:g} {width:g} {height:g} re "
            + ("B" if fill and stroke else "f" if fill else "S" if stroke else "n")
        )

    def line(self, x1, y1, x2, y2):
        self.ops.append(f"{x1:g} {y1:g} m {x2:g} {y2:g} l S")

    def circle(self, x, y, radius, fill=0, stroke=1):
        k = radius * 0.5522847498
        self.ops.append(
            f"{x+radius:g} {y:g} m {x+radius:g} {y+k:g} {x+k:g} {y+radius:g} {x:g} {y+radius:g} c "
            f"{x-k:g} {y+radius:g} {x-radius:g} {y+k:g} {x-radius:g} {y:g} c "
            f"{x-radius:g} {y-k:g} {x-k:g} {y-radius:g} {x:g} {y-radius:g} c "
            f"{x+k:g} {y-radius:g} {x+radius:g} {y-k:g} {x+radius:g} {y:g} c "
            + ("B" if fill and stroke else "f" if fill else "S" if stroke else "n")
        )

    def saveState(self):
        self.ops.append("q")

    def restoreState(self):
        self.ops.append("Q")

    def translate(self, x, y):
        self.ops.append(f"1 0 0 1 {x:g} {y:g} cm")

    def rotate(self, degrees):
        import math
        angle = math.radians(degrees)
        c, s = math.cos(angle), math.sin(angle)
        self.ops.append(f"{c:g} {s:g} {-s:g} {c:g} 0 0 cm")

    def save(self):
        content = []
        for op in self.ops:
            # Colors and line width are captured at operation time below by
            # prepending their current state where each drawing command occurs.
            content.append(op)
        stream = "\n".join(content).encode("ascii", "replace")
        objects = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {self.width:g} {self.height:g}] /Resources << /Font << /Helvetica 4 0 R /Helvetica-Bold 5 0 R >> >> /Contents 6 0 R >>".encode("ascii"),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
            b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        ]
        pdf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = [0]
        for number, obj in enumerate(objects, 1):
            offsets.append(len(pdf))
            pdf.extend(f"{number} 0 obj\n".encode("ascii")); pdf.extend(obj); pdf.extend(b"\nendobj\n")
        xref = len(pdf)
        pdf.extend(f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode("ascii"))
        for offset in offsets[1:]:
            pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
        pdf.extend(f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode("ascii"))
        self.filename.parent.mkdir(parents=True, exist_ok=True)
        self.filename.write_bytes(pdf)


class _CanvasModule:
    Canvas = Canvas


class pdfgen:
    canvas = _CanvasModule()
