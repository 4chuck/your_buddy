import PyPDF2
import os
import re

try:
    import docx
except ImportError:
    docx = None

try:
    from pptx import Presentation
except ImportError:
    Presentation = None


def extract_text_from_pdf(pdf_path):
    """Extracts text and page numbers from a PDF file."""
    text_content = []
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text = page.extract_text()
                if text:
                    text_content.append({
                        "page": page_num + 1,
                        "text": text
                    })
    except Exception as e:
        print(f"Error extracting PDF: {e}")
    return text_content


def extract_text_from_docx(docx_path):
    """Extracts text from a DOCX file."""
    if docx is None:
        print("python-docx is not installed.")
        return []
    text_content = []
    try:
        document = docx.Document(docx_path)
        paragraphs = [p.text for p in document.paragraphs if p.text]
        text = "\n".join(paragraphs)
        if text:
            text_content.append({"page": 1, "text": text})
    except Exception as e:
        print(f"Error extracting DOCX: {e}")
    return text_content


def extract_text_from_pptx(pptx_path):
    """Extracts text from a PPTX file."""
    if Presentation is None:
        print("python-pptx is not installed.")
        return []
    text_content = []
    try:
        prs = Presentation(pptx_path)
        slides = []
        for slide in prs.slides:
            slide_text = []
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text:
                    slide_text.append(shape.text)
            if slide_text:
                slides.append("\n".join(slide_text))
        if slides:
            text_content.append({"page": 1, "text": "\n\n".join(slides)})
    except Exception as e:
        print(f"Error extracting PPTX: {e}")
    return text_content


def extract_text_from_binary(file_path):
    """Extract human-readable text from binary files by extracting printable strings."""
    try:
        with open(file_path, 'rb') as f:
            raw = f.read()
        text = raw.decode('latin-1', errors='ignore')
        segments = re.findall(r'[\x09\x0A\x0D\x20-\x7E]{20,}', text)
        if segments:
            content = '\n\n'.join(segments)
            if len(content.strip()) >= 30:
                return [{"page": 1, "text": content}]
    except Exception as e:
        print(f"Error extracting binary text: {e}")
    return []


def extract_text_from_file(file_path):
    """Extracts text from different file types, with PDF and text fallback support."""
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext == '.pdf':
        return extract_text_from_pdf(file_path)

    if ext == '.docx':
        return extract_text_from_docx(file_path)

    if ext == '.pptx':
        return extract_text_from_pptx(file_path)

    if ext in {'.txt', '.md', '.csv', '.json', '.log', '.rst', '.tex', '.html', '.xml'}:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    text = f.read()
            except Exception as e:
                print(f"Error reading text file: {e}")
                return []
        return [{"page": 1, "text": text}]

    # Fallback for other file types: attempt to decode any readable text strings.
    binary_text = extract_text_from_binary(file_path)
    if binary_text:
        return binary_text

    print(f"Unsupported file type or unreadable file extension: {ext}")
    return []


def chunk_text(text_data, chunk_size=1000, overlap=100):
    """Splits extracted text into chunks with metadata."""
    chunks = []
    for item in text_data:
        text = item["text"]
        page = item["page"]
        
        # Simple character-based chunking for demo
        # For more precision, use token-based chunking
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append({
                "content": chunk,
                "metadata": {"page": page}
            })
            start += chunk_size - overlap
    return chunks
