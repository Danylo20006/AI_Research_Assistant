from scripts._05_document_ingestion import extract_text_from_pdf

def chunk_text(
    pages: list[dict],
    chunk_size: int = 1000,
    overlap: int = 200
    ) -> list[dict]:
    """
    Splits document pages into smaller fragments using a sliding window algorithm.
    
    Chunking solves two primary architectural constraints:
    1. Context Window Limits: Prevents exceeding the token limit of the embedding model.
    2. Semantic Dilution: Ensures the vector represents a specific idea rather than an 
       averaged representation of an entire page.
    """
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
            # Sliding window algorithm: advance the index by (chunk_size - overlap).
            # The overlap mechanism acts as a safety net, ensuring that sentences broken 
            # across chunk boundaries do not lose their semantic context, improving retrieval recall.
            for i in range(0, len(text), chunk_size - overlap):
                chunk = text[i: i + chunk_size]
                chunks.append({
                    'text': chunk,
                    'page_number': page_number,
                    'chunk_index': chunk_index
                    })

                chunk_index += 1

                # Break the loop if the next chunk would extend beyond the end of the text
                if i + chunk_size >= len(text):
                    break

    return chunks

if __name__ == '__main__':
    pages = extract_text_from_pdf("./data/papers/novel.pdf")
    chunks = chunk_text(pages)

    print(f"Amount of pages: {len(pages)}")
    print(f"Amount of chunks: {len(chunks)}")
    print("\n--- First chunk ---")
    print(chunks[0])
    print("\n--- Second chunk ---")
    print(chunks[1])
    print("\n--- The end of first chunk ---")
    print(chunks[0]["text"][-200:])
    print("\n--- THe start of second chunk ---")
    print(chunks[1]["text"][:200])