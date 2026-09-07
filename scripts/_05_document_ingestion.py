import pymupdf

def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """
    Extracts text from a PDF document page-by-page.
    
    Retaining the page_number as metadata is a fundamental requirement for RAG systems,
    as it enables the LLM to provide accurate citations and ground its answers 
    in specific sections of the source document.
    """
    
    # Using a context manager ensures the file descriptor is safely closed after extraction,
    # preventing memory leaks.
    with pymupdf.open(pdf_path) as doc:
        pages = []

        for page_num, page in enumerate(doc):
            text = page.get_text()

            # PDF parsers often insert hard line breaks mid-sentence due to visual formatting.
            # Replacing '\n' with spaces prevents artificial fragmentation of semantic context,
            # which would otherwise degrade the quality of embeddings.
            cleaned_text = text.replace('\n', ' ').strip()

            cleaned_page = {
                'page_number': page_num + 1,
                'text': cleaned_text
            }

            pages.append(cleaned_page)

        return pages

if __name__ == '__main__':
    pages = extract_text_from_pdf("./data/papers/novel.pdf")
    print(f'Total pages: {len(pages)}')
    print(pages[0]['text'][:500])