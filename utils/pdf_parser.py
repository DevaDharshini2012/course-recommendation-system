"""
PDF Marksheet Parser Utility
Extracts subject-wise marks from uploaded PDF marksheets.
"""

import re
import os
import pdfplumber
try:
    import pypdf as PyPDF2_mod
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False


SUBJECT_KEYWORDS = {
    "mathematics": ["math", "mathematics", "maths", "calculus", "algebra", "discrete math"],
    "physics": ["physics", "phy", "engineering physics", "applied physics"],
    "chemistry": ["chemistry", "chem", "engineering chemistry"],
    "computer_science": ["computer science", "cs", "programming", "data structures", "dsa",
                         "algorithms", "c programming", "java", "python programming",
                         "object oriented", "oop", "software engineering"],
    "english": ["english", "communication", "technical communication", "professional communication"],
    "statistics": ["statistics", "stat", "probability", "probability and statistics",
                   "applied mathematics", "numerical methods"],
    "biology": ["biology", "bio", "life science", "biotechnology"],
    "economics": ["economics", "eco", "engineering economics", "management", "industrial management"]
}


def extract_marks_from_pdf(pdf_path: str) -> dict:
    """
    Extract subject-wise marks from a PDF marksheet.
    Returns dict with subject -> average_mark mappings.
    """
    extracted_data = {subj: [] for subj in SUBJECT_KEYWORDS}
    raw_text_lines = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    raw_text_lines.extend(text.split("\n"))

                # Try table extraction
                tables = page.extract_tables()
                for table in tables:
                    for row in table:
                        if row:
                            _process_row(row, extracted_data)

    except Exception as e:
        print(f"[PDF Parser] pdfplumber error: {e}. Trying fallback...")
        # Try fallback with PyPDF2
        extracted_data = _fallback_pypdf2(pdf_path, extracted_data)

    # Process text lines for mark extraction
    for line in raw_text_lines:
        _process_line(line, extracted_data)

    # Compute averages and fill missing with estimates
    result = {}
    for subj, marks in extracted_data.items():
        if marks:
            result[subj] = round(sum(marks) / len(marks), 2)
        else:
            # Default estimate for undetected subjects
            result[subj] = 65.0

    result = _compute_derived_strengths(result)
    return result


def _fallback_pypdf2(pdf_path, extracted_data):
    """Fallback PDF reader using pypdf."""
    if not HAS_PYPDF:
        return extracted_data
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2_mod.PdfReader(f)
            for page in reader.pages:
                text = page.extract_text() or ""
                for line in text.split("\n"):
                    _process_line(line, extracted_data)
    except Exception as e:
        print(f"[PDF Parser] pypdf fallback error: {e}")
    return extracted_data


def _process_row(row, extracted_data):
    """Process a table row to find subject-mark pairs."""
    row_text = " ".join([str(cell).lower().strip() for cell in row if cell])
    _process_line(row_text, extracted_data)


def _process_line(line: str, extracted_data: dict):
    """Extract marks from a single text line."""
    line_lower = line.lower().strip()
    if not line_lower:
        return

    # Find numbers in the line (potential marks)
    numbers = re.findall(r'\b(\d{1,3})\b', line)
    marks_in_line = [int(n) for n in numbers if 10 <= int(n) <= 100]

    if not marks_in_line:
        return

    # Check which subject this line belongs to
    for subj, keywords in SUBJECT_KEYWORDS.items():
        for keyword in keywords:
            if keyword in line_lower:
                # Take the first valid mark found
                if marks_in_line:
                    extracted_data[subj].append(marks_in_line[0])
                break


def _compute_derived_strengths(marks: dict) -> dict:
    """Compute derived strength scores from subject marks."""
    cs = marks.get("computer_science", 65)
    math = marks.get("mathematics", 65)
    stat = marks.get("statistics", 65)
    eng = marks.get("english", 65)

    all_vals = list(marks.values())
    avg = sum(all_vals) / len(all_vals) if all_vals else 65

    marks["logical_thinking"] = round(min(100, cs * 0.5 + math * 0.3 + avg * 0.2), 2)
    marks["analytical_ability"] = round(min(100, math * 0.4 + stat * 0.4 + avg * 0.2), 2)
    marks["programming_fundamentals"] = round(min(100, cs * 0.7 + math * 0.3), 2)
    marks["communication_skills"] = round(min(100, eng * 0.6 + avg * 0.4), 2)
    marks["overall_average"] = round(avg, 2)

    return marks


def simulate_marks_from_manual_input(subject_marks: dict) -> dict:
    """
    For cases where user manually enters marks instead of uploading PDF.
    subject_marks: {subject_name: mark_value}
    """
    extracted = {subj: 65.0 for subj in SUBJECT_KEYWORDS}
    for subj, mark in subject_marks.items():
        if subj in extracted:
            extracted[subj] = float(mark)
    return _compute_derived_strengths(extracted)
