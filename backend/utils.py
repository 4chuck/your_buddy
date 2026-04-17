import os
import re
import PyPDF2

try:
    import docx
except ImportError:
    docx = None

try:
    from pptx import Presentation
except ImportError:
    Presentation = None


# ----------------------------
# PDF EXTRACTION (SAFE)
# ----------------------------
def extract_text_from_pdf(pdf_path):
    text_content = []

    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)

            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()

                if text and text.strip():
                    text_content.append({
                        "page": page_num + 1,
                        "text": text.strip()
                    })

    except Exception as e:
        print(f"PDF extraction error: {e}")

    return text_content


# ----------------------------
# DOCX EXTRACTION
# ----------------------------
def extract_text_from_docx(docx_path):
    if docx is None:
        return []

    try:
        document = docx.Document(docx_path)
        text = "\n".join([p.text for p in document.paragraphs if p.text.strip()])

        if text.strip():
            return [{"page": 1, "text": text.strip()}]

    except Exception as e:
        print(f"DOCX error: {e}")

    return []


# ----------------------------
# PPTX EXTRACTION
# ----------------------------
def extract_text_from_pptx(pptx_path):
    if Presentation is None:
        return []

    try:
        prs = Presentation(pptx_path)
        slides_text = []

        for slide in prs.slides:
            slide_text = []

            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    slide_text.append(shape.text.strip())

            if slide_text:
                slides_text.append("\n".join(slide_text))

        if slides_text:
            return [{"page": 1, "text": "\n\n".join(slides_text)}]

    except Exception as e:
        print(f"PPTX error: {e}")

    return []


# ----------------------------
# TEXT FILES
# ----------------------------
def extract_text_from_file(file_path):
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext == ".pdf":
        return extract_text_from_pdf(file_path)

    if ext == ".docx":
        return extract_text_from_docx(file_path)

    if ext == ".pptx":
        return extract_text_from_pptx(file_path)

    if ext in [".txt", ".md", ".csv", ".json", ".log", ".html", ".xml"]:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read().strip()

            if text:
                return [{"page": 1, "text": text}]

        except Exception as e:
            print(f"Text file error: {e}")

    return []


# ----------------------------
# CLEAN CHUNKING (IMPORTANT FOR RAG QUALITY)
# ----------------------------
def chunk_text(text_data, chunk_size=900, overlap=150):
    chunks = []

    for item in text_data:
        text = item.get("text", "")
        page = item.get("page", 1)

        if not text:
            continue

        # Clean noisy whitespace
        text = re.sub(r"\s+", " ", text).strip()

        start = 0
        while start < len(text):
            end = start + chunk_size

            chunk = text[start:end].strip()

            if len(chunk) > 50:  # ignore tiny garbage chunks
                chunks.append({
                    "content": chunk,
                    "metadata": {
                        "page": page
                    }
                })

            start += chunk_size - overlap

    return chunks