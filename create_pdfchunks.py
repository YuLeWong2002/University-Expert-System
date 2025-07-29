import json, pathlib, re, pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# ---------------------------------------------------------------------
# 1. helpers -----------------------------------------------------------
# ---------------------------------------------------------------------
HEADER_FOOTER_RE = re.compile(
    r"(?:^|\n)\s*Page\s+\d+\s*(?:of\s+\d+)?\s*(?:\n|$)|"  # Page 1 / Page 1 of 10
    r"(?:^|\n)\s*\d{4}-\d{2}-\d{2}\s*(?:\n|$)|"           # dates
    r"(?:^|\n)\s*Confidential\s*(?:\n|$)", re.I
)

def clean_text(text: str) -> str:
    """strip header/footer, de‑hyphenate, collapse single line‑breaks."""
    text = re.sub(HEADER_FOOTER_RE, "\n", text)
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)       # de‑hyphenate
    text = re.sub(r"\n(?!\n)", " ", text)                # join lines
    text = re.sub(r"\n{2,}", "\n\n", text).strip()
    return text

def load_pdf_documents(pdf_folder: str) -> list[Document]:
    """Extract and clean every PDF page; return as LangChain Documents."""
    docs = []
    for path in pathlib.Path(pdf_folder).glob("*.pdf"):
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                text = clean_text(text)
                if text:
                    docs.append(
                        Document(
                            page_content=text,
                            metadata={"source": path.name, "page": i}
                        )
                    )
    return docs

def main():
    pdf_folder = "pdf"
    out_file   = "pdf_chunks.jsonl" 

    raw_docs = load_pdf_documents(pdf_folder)
    print(f"Loaded {len(raw_docs)} cleaned pages.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size    = 1000,
        chunk_overlap = 100,
        separators    = ["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(raw_docs)
    print(f"Split into {len(chunks)} chunks.")

    with open(out_file, "w", encoding="utf‑8") as f:
        for idx, chunk in enumerate(chunks):
            meta = chunk.metadata
            obj  = {
                "id"         : f"{meta['source']}_p{meta['page']}_c{idx}",
                "source"     : meta["source"],
                "page"       : meta["page"],
                "chunk_index": idx,
                "text"       : chunk.page_content.strip()
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Saved every chunk to {out_file}")

if __name__ == "__main__":
    main()
