import os
import json
from langchain.schema import Document
import PyPDF2

def load_json_documents(folder_path: str):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r") as f:
                data = json.load(f)
                content = data.get("content", "")
                if isinstance(content, list):
                    content = " ".join(str(item) for item in content)
                content = str(content).strip()
                if content:
                    # Use the "url" if available; otherwise, fallback to filename
                    source = data.get("url", filename)
                    documents.append(Document(page_content=content, metadata={"source": source}))
    return documents


def load_pdf_documents(pdf_folder: str):
    pdf_docs = []
    for filename in os.listdir(pdf_folder):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(pdf_folder, filename)
            try:
                with open(file_path, "rb") as file:
                    reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    if text.strip():
                        pdf_docs.append(Document(page_content=text.strip(), metadata={"source": filename}))
                print(f"Loaded PDF: {filename}")
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    return pdf_docs

if __name__ == "__main__":
    json_docs = load_json_documents("f13_json")
    pdf_docs = load_pdf_documents("pdf")
    total_docs = json_docs + pdf_docs
    print(f"Loaded {len(total_docs)} documents in total.")
