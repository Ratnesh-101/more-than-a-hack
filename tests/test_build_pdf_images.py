"""Tests for figure and image support in docs/build_pdf.py."""
import hashlib
from pathlib import Path
import PIL.Image
import pytest
from reportlab.lib.pagesizes import A4
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate

from docs.build_pdf import CAPTION, MARGIN, WIDTH, footer, parse


@pytest.fixture
def tiny_png(tmp_path):
    """Create a tiny 20x10 PNG image for testing."""
    img_path = tmp_path / "tiny_test.png"
    img = PIL.Image.new("RGB", (20, 10), color=(30, 100, 180))
    img.save(img_path)
    return img_path


def test_parse_markdown_image_syntax(tiny_png, tmp_path):
    """Test that markdown image syntax is parsed into Image and caption in KeepTogether."""
    caption_text = "A tiny generated test figure for verification."
    md = f"![{caption_text}]({tiny_png.name})"
    flow = parse(md, base_dir=tmp_path)

    assert len(flow) == 1
    item = flow[0]
    assert isinstance(item, KeepTogether)
    sub = item._content
    assert len(sub) == 2
    img, cap = sub[0], sub[1]

    assert isinstance(img, Image)
    assert img.drawWidth == pytest.approx(WIDTH)
    expected_height = 10 * (WIDTH / 20)
    assert img.drawHeight == pytest.approx(expected_height)

    assert isinstance(cap, Paragraph)
    assert cap.style.name == "caption"
    assert caption_text in cap.text


def test_image_without_caption(tiny_png, tmp_path):
    """Test image markdown without caption text."""
    md = f"![]({tiny_png.name})"
    flow = parse(md, base_dir=tmp_path)
    assert len(flow) == 1
    assert isinstance(flow[0], KeepTogether)
    img = flow[0]._content[0]
    assert isinstance(img, Image)
    assert img.drawWidth == pytest.approx(WIDTH)


def test_missing_image_raises(tmp_path):
    """Test that referencing a nonexistent image raises FileNotFoundError."""
    md = "![Missing figure](nonexistent_image_12345.png)"
    with pytest.raises(FileNotFoundError):
        parse(md, base_dir=tmp_path)


def test_build_pdf_with_image_determinism(tiny_png, tmp_path):
    """Test building a PDF with an image is fully deterministic across multiple runs."""
    caption_text = "**Figure 1:** Verification of `deterministic` rendering."
    md = f"# Title\n\nParagraph before image.\n\n![{caption_text}]({tiny_png.name})\n\nParagraph after image.\n"

    def render_pdf(dest_path):
        doc = SimpleDocTemplate(
            str(dest_path),
            pagesize=A4,
            leftMargin=MARGIN,
            rightMargin=MARGIN,
            topMargin=14 * 2.83465,  # mm
            bottomMargin=16 * 2.83465,
            invariant=1,
            title="Determinism Test",
            author="nirakshak",
        )
        flowables = parse(md, base_dir=tmp_path)
        doc.build(flowables, onFirstPage=footer("nirakshak"), onLaterPages=footer("nirakshak"))
        with open(dest_path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()

    pdf1 = tmp_path / "out1.pdf"
    pdf2 = tmp_path / "out2.pdf"
    h1 = render_pdf(pdf1)
    h2 = render_pdf(pdf2)

    assert pdf1.exists()
    assert pdf1.stat().st_size > 0
    assert h1 == h2, "PDF output with embedded figures must be byte-for-byte deterministic"
