import pymupdf

def extract_text_from_pdf(pdf_path: str) -> list[dict]:

    with pymupdf.open(pdf_path) as doc:

        pages = []

        for page_num, page in enumerate(doc):
            text = page.get_text()

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