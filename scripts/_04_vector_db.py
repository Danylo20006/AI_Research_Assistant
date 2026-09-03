import chromadb


client = chromadb.PersistentClient(path='./data/chroma_db')

research_papers = client.get_or_create_collection(name='research_papers')

text1 = 'I love programming in Python and building AI systems.'
text2 = 'Coding in Python for machine learning is my favorite hobby.'
text3 = 'The weather in Lviv is quite rainy today.'

documents = [text1, text2, text3]
ids = ['doc1', 'doc2', 'doc3']
metadatas = [
    {'source':'user_input'},
    {'source':'user_input'},
    {'source':'weather_api'},
     ]

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