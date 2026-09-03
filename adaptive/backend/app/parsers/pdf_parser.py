from pathlib import Path
from typing import Union
import pdfplumber
import fitz  # PyMuPDF

def extract_text_from_pdf(file_path: Union[str, Path]) -> str:
    """
    Extracts text from a PDF file using PyMuPDF (fitz) with column-aware block sorting,
    falling back to pdfplumber if fitz fails.
    """
    path_str = str(file_path)
    text_blocks = []

    try:
        doc = fitz.open(path_str)
        for page in doc:
            page_width = page.rect.width
            blocks = page.get_text("blocks")
            # Filter non-text blocks
            valid_blocks = [b for b in blocks if len(b) >= 5 and b[4].strip()]
            
            # Detect multi-column layout by inspecting x0 coordinates
            x_coords = [b[0] for b in valid_blocks]
            has_two_columns = False
            if len(valid_blocks) > 3 and page_width > 300:
                # Left sidebar is usually 30-35% of page width. 
                # Split at 35% of page width to separate columns.
                split_x = page_width * 0.35
                left_col = [b for b in valid_blocks if b[0] < split_x]
                right_col = [b for b in valid_blocks if b[0] >= split_x]
                if len(left_col) >= 2 and len(right_col) >= 2:
                    has_two_columns = True

            if has_two_columns:
                # Sort left column top-to-bottom, then right column top-to-bottom
                sorted_left = sorted(left_col, key=lambda b: b[1])
                sorted_right = sorted(right_col, key=lambda b: b[1])
                page_text = "\n\n".join([b[4].strip() for b in (sorted_left + sorted_right)])
            else:
                sorted_blocks = sorted(valid_blocks, key=lambda b: (b[1], b[0]))
                page_text = "\n\n".join([b[4].strip() for b in sorted_blocks])

            if page_text.strip():
                text_blocks.append(page_text)

        text = "\n\n".join(text_blocks).strip()
        if len(text) > 50:
            return text
    except Exception:
        text = ""

    # Fallback to pdfplumber if fitz text is empty
    try:
        with pdfplumber.open(path_str) as pdf:
            pages_text = []
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    pages_text.append(extracted)
            text = "\n\n".join(pages_text)
    except Exception:
        pass

    return text

