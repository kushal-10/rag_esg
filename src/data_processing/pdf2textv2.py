from pdfminer.high_level import extract_text

pdf_path = "data/reports_final/jungheinrich-ag_2015.pdf"
txt_path = "data/reports_final/jungheinrich-ag_2015.txt"

text = extract_text(pdf_path)
with open(txt_path, 'w', encoding='utf-8') as f:
    f.write(text)
