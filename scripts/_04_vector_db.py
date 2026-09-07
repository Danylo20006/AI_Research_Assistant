import chromadb

# Using PersistentClient ensures that the embeddings are saved to the local disk.
# This is crucial for statefulness, preventing the need to re-embed documents on every restart.
client = chromadb.PersistentClient(path='./data/chroma_db')

# get_or_create_collection is an idempotent operation, safely handling multiple script executions.
# Collections act as logical namespaces for vector indices (similar to tables in SQL).
research_papers = client.get_or_create_collection(name='research_papers')

text1 = 'I love programming in Python and building AI systems.'
text2 = 'Coding in Python for machine learning is my favorite hobby.'
text3 = 'The weather in Lviv is quite rainy today.'

documents = [text1, text2, text3]
ids = ['doc1', 'doc2', 'doc3']

# Metadata is critical for Hybrid Search in production RAG systems.
# It allows the system to apply deterministic pre-filters (e.g., WHERE source="user_input")
# before calculating vector distances, heavily optimizing search speed.
metadatas = [
    {'source':'user_input'},
    {'source':'user_input'},
    {'source':'weather_api'},
]

# Since we don't explicitly pass an embedding function, ChromaDB automatically loads 
# its default sentence-transformer model to convert these texts into vectors behind the scenes.
research_papers.add(
    documents=documents,
    ids=ids,
    metadatas=metadatas,
)

results = research_papers.query(
    query_texts=['Tell me about software engineering'],
    n_results=2
)

for i in range(len(results['documents'][0])):
    print(f'Documents: {results['documents'][0][i]}')
    print(f'Metadata: {results['metadatas'][0][i]}')
    print(f'Distance: {results['distances'][0][i]}')
    print()