import chromadb
from scripts._05_document_ingestion import extract_text_from_pdf
from scripts._06_chunking import chunk_text
from scripts._01_llm_basics import generate_response

def setup_database():

    pages = extract_text_from_pdf("./data/papers/novel.pdf")

    chunks = chunk_text(pages)

    client = chromadb.PersistentClient(path='./data/test_chroma_db')

    try:
        client.delete_collection("rag_test")
    except Exception:
        pass

    rag_test = client.create_collection("rag_test")


    documents = [chunk['text'] for chunk in chunks]
    ids = [f"chunk_{chunk['chunk_index']}" for chunk in chunks]
    metadatas = [{'page': chunk['page_number']} for chunk in chunks]

    rag_test.add(
        documents=documents,
        ids=ids,
        metadatas=metadatas
    )

    return rag_test

def ask_question(rag_test, query: str) -> str:
    results = rag_test.query(
        query_texts=[query],
        n_results=3
    )

    results = results['documents'][0]

    print("\nRetrieved chunks:")

    for i, chunk in enumerate(results, 1):
        print(f"\n--- Chunk {i} ---")
        print(chunk)

    context = '\n\n'.join(results)

    system_prompt = """
    You are an AI assistant.
    Use only the context provided to answer the question.
    If the context doesn't answer, say 'I don't know'.
    """

    user_prompt = f'Context: \n {context} \n\n Question: \n {query}'

    response = generate_response(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.0
        )

    return response

if __name__ == '__main__': 

    novel = setup_database()

    question = "How many people are left after the AM apocalypse?"
    # question = "What is the capital of Ukraine?"

    answer = ask_question(rag_test=novel, query=question)

    print()
    print()
    print(question)
    print()
    print(answer)