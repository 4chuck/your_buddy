from fpdf import FPDF

pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=15)
pdf.cell(200, 10, txt="Introduction to Agentic AI", ln=1, align='C')
pdf.set_font("Arial", size=12)
pdf.multi_cell(0, 10, txt="Agentic AI systems are designed to interact intelligently with their environments. They perceive state, make decisions using reasoning, and act autonomously based on user alignment goals. \n\nThis is Chapter 1. The key topic is autonomous workflows using tools.")
pdf.output("test_document.pdf")
print("test_document.pdf created.")
