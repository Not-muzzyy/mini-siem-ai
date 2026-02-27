"""Professional incident response report generator.

This module creates a structured incident response report PDF with the sections:
- Executive summary
- Attack timeline
- Technical breakdown
- Explainable AI reasoning
- Risk assessment
- Containment steps
- Recovery plan
- Prevention strategy

No third-party dependencies are required; PDF bytes are generated directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from textwrap import wrap
from typing import Iterable


@dataclass(frozen=True)
class IncidentResponseReport:
    """Structured incident response report content."""

    title: str
    executive_summary: str
    attack_timeline: str
    technical_breakdown: str
    explainable_ai_reasoning: str
    risk_assessment: str
    containment_steps: str
    recovery_plan: str
    prevention_strategy: str


class IncidentReportGenerationError(RuntimeError):
    """Raised when report generation fails."""


class IncidentResponsePDFGenerator:
    """Generates professional incident response reports as downloadable PDFs."""

    def __init__(
        self,
        page_width: int = 612,
        page_height: int = 792,
        margin: int = 50,
        body_font_size: int = 11,
        heading_font_size: int = 14,
        title_font_size: int = 18,
        line_spacing: int = 16,
        section_spacing: int = 10,
    ) -> None:
        self.page_width = page_width
        self.page_height = page_height
        self.margin = margin
        self.body_font_size = body_font_size
        self.heading_font_size = heading_font_size
        self.title_font_size = title_font_size
        self.line_spacing = line_spacing
        self.section_spacing = section_spacing

        if self.margin * 2 >= self.page_width:
            raise IncidentReportGenerationError("margin is too large for page width.")

        self.text_width = self.page_width - (2 * self.margin)
        self.max_chars_per_line = max(40, int(self.text_width / 5.6))

    def generate_pdf(self, report: IncidentResponseReport, output_path: str | Path) -> Path:
        """Generate the incident response report PDF on disk.

        Args:
            report: Structured report content.
            output_path: Destination file path for the PDF.

        Returns:
            Path to the generated PDF file.
        """
        self._validate_report(report)

        pages = self._build_pages(report)
        pdf_bytes = self._render_pdf_bytes(pages)

        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(pdf_bytes)
        return destination

    def _build_pages(self, report: IncidentResponseReport) -> list[list[tuple[str, int, str]]]:
        sections = [
            ("Executive Summary", report.executive_summary),
            ("Attack Timeline", report.attack_timeline),
            ("Technical Breakdown", report.technical_breakdown),
            ("Explainable AI Reasoning", report.explainable_ai_reasoning),
            ("Risk Assessment", report.risk_assessment),
            ("Containment Steps", report.containment_steps),
            ("Recovery Plan", report.recovery_plan),
            ("Prevention Strategy", report.prevention_strategy),
        ]

        pages: list[list[tuple[str, int, str]]] = [[]]
        y_cursor = self.page_height - self.margin

        def ensure_space(required_height: int) -> None:
            nonlocal y_cursor
            if y_cursor - required_height < self.margin:
                pages.append([])
                y_cursor = self.page_height - self.margin

        title_lines = self._wrap_text(report.title, self.max_chars_per_line)
        for line in title_lines:
            ensure_space(self.line_spacing)
            pages[-1].append((line, self.title_font_size, "B"))
            y_cursor -= self.line_spacing

        ensure_space(self.section_spacing)
        y_cursor -= self.section_spacing

        for heading, body in sections:
            heading_lines = self._wrap_text(heading, self.max_chars_per_line)
            for line in heading_lines:
                ensure_space(self.line_spacing)
                pages[-1].append((line, self.heading_font_size, "B"))
                y_cursor -= self.line_spacing

            body_lines = self._wrap_text(body, self.max_chars_per_line)
            for line in body_lines:
                ensure_space(self.line_spacing)
                pages[-1].append((line, self.body_font_size, ""))
                y_cursor -= self.line_spacing

            ensure_space(self.section_spacing)
            y_cursor -= self.section_spacing

        return pages

    @staticmethod
    def _validate_report(report: IncidentResponseReport) -> None:
        required_values = {
            "title": report.title,
            "executive_summary": report.executive_summary,
            "attack_timeline": report.attack_timeline,
            "technical_breakdown": report.technical_breakdown,
            "explainable_ai_reasoning": report.explainable_ai_reasoning,
            "risk_assessment": report.risk_assessment,
            "containment_steps": report.containment_steps,
            "recovery_plan": report.recovery_plan,
            "prevention_strategy": report.prevention_strategy,
        }

        empty = [field for field, value in required_values.items() if not str(value).strip()]
        if empty:
            raise IncidentReportGenerationError(
                f"Missing required report sections: {', '.join(empty)}"
            )

    @staticmethod
    def _wrap_text(text: str, width: int) -> list[str]:
        normalized = " ".join(str(text).strip().split())
        if not normalized:
            return [""]
        return wrap(normalized, width=width, replace_whitespace=False, drop_whitespace=False)

    def _render_pdf_bytes(self, pages: Iterable[list[tuple[str, int, str]]]) -> bytes:
        objects: list[bytes] = []

        # 1. Catalog
        objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")

        # 2. Pages container placeholder (filled after pages created)
        objects.append(b"")

        page_object_numbers: list[int] = []

        for page_lines in pages:
            content_stream = self._build_content_stream(page_lines)
            content_obj_num = len(objects) + 1
            objects.append(
                b"<< /Length " + str(len(content_stream)).encode("ascii") + b" >>\nstream\n"
                + content_stream
                + b"\nendstream"
            )

            page_obj_num = len(objects) + 1
            page_object_numbers.append(page_obj_num)
            page_obj = (
                b"<< /Type /Page /Parent 2 0 R "
                + f"/MediaBox [0 0 {self.page_width} {self.page_height}] ".encode("ascii")
                + b"/Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >> "
                + f"/Contents {content_obj_num} 0 R >>".encode("ascii")
            )
            objects.append(page_obj)

        kids = " ".join(f"{num} 0 R" for num in page_object_numbers)
        pages_obj = f"<< /Type /Pages /Count {len(page_object_numbers)} /Kids [{kids}] >>".encode("ascii")
        objects[1] = pages_obj

        return self._assemble_pdf(objects)

    def _build_content_stream(self, page_lines: list[tuple[str, int, str]]) -> bytes:
        lines: list[str] = ["BT"]
        y_cursor = self.page_height - self.margin

        for text, font_size, font_style in page_lines:
            escaped = self._escape_pdf_text(text)
            lines.append(f"/F1 {font_size} Tf")
            lines.append(f"1 0 0 1 {self.margin} {y_cursor} Tm")
            if font_style == "B":
                # Synthetic bold via render mode stroke+fill for simple dependency-free output.
                lines.append("2 Tr")
                lines.append("0.2 w")
            else:
                lines.append("0 Tr")
            lines.append(f"({escaped}) Tj")
            y_cursor -= self.line_spacing

        lines.append("ET")
        return "\n".join(lines).encode("latin-1", errors="replace")

    @staticmethod
    def _escape_pdf_text(text: str) -> str:
        return (
            str(text)
            .replace("\\", "\\\\")
            .replace("(", "\\(")
            .replace(")", "\\)")
        )

    @staticmethod
    def _assemble_pdf(objects: list[bytes]) -> bytes:
        header = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
        body = bytearray()
        offsets = [0]

        for index, obj in enumerate(objects, start=1):
            offsets.append(len(header) + len(body))
            body.extend(f"{index} 0 obj\n".encode("ascii"))
            body.extend(obj)
            body.extend(b"\nendobj\n")

        xref_start = len(header) + len(body)
        xref = bytearray()
        xref.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
        xref.extend(b"0000000000 65535 f \n")
        for offset in offsets[1:]:
            xref.extend(f"{offset:010d} 00000 n \n".encode("ascii"))

        trailer = (
            b"trailer\n"
            + f"<< /Size {len(objects) + 1} /Root 1 0 R >>\n".encode("ascii")
            + b"startxref\n"
            + str(xref_start).encode("ascii")
            + b"\n%%EOF\n"
        )

        return header + bytes(body) + bytes(xref) + trailer


def generate_incident_response_report_pdf(
    report: IncidentResponseReport,
    output_path: str | Path,
    generator: IncidentResponsePDFGenerator | None = None,
) -> Path:
    """Convenience API to generate a professional incident response PDF."""
    pdf_generator = generator or IncidentResponsePDFGenerator()
    return pdf_generator.generate_pdf(report=report, output_path=output_path)
