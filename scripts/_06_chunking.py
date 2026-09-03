from scripts._05_document_ingestion import extract_text_from_pdf

def chunk_text(
    pages: list[dict],
    chunk_size: int = 1000,
    overlap: int = 200
    ) -> list[dict]:

    chunks = []
    chunk_index = 0

    for page in pages:
        text = page['text']
        page_number = page['page_number']

        if len(text) <= chunk_size:
            chunks.append({
                'text': text,
                'page_number': page_number,
                'chunk_index': chunk_index
                })
            
            chunk_index += 1

        else:
            for i in range(0, len(text), chunk_size - overlap):
                chunk = text[i: i + chunk_size]
                chunks.append({
                    'text': chunk,
                    'page_number': page_number,
                    'chunk_index': chunk_index
                    })

                chunk_index += 1

                if i + chunk_size >= len(text):
                    break

    return chunks

if __name__ == '__main__':

    pages = extract_text_from_pdf("./data/papers/novel.pdf")

    chunks = chunk_text(pages)

    print(f"Кількість сторінок: {len(pages)}")
    print(f"Кількість чанків: {len(chunks)}")

    print("\n--- Перший chunk ---")
    print(chunks[0])

    print("\n--- Другий chunk ---")
    print(chunks[1])

    print("\n--- Кінець першого chunk ---")
    print(chunks[0]["text"][-200:])

    print("\n--- Початок другого chunk ---")
    print(chunks[1]["text"][:200])