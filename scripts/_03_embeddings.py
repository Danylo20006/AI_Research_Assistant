from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize the embedding model to project text into a dense 384-dimensional vector space.
model = SentenceTransformer('all-MiniLM-L6-v2')

text1 = 'I love programming in Python and building AI systems.'
text2 = 'Coding in Python for machine learning is my favorite hobby.'
text3 = 'The weather in Lviv is quite rainy today.'

# Convert natural language strings into fixed-size float arrays (embeddings)
embeddings1 = model.encode(text1)
embeddings2 = model.encode(text2)
embeddings3 = model.encode(text3)

print('Vector dimension:', len(embeddings1))

# Compute semantic similarity by calculating the cosine angle between vectors.
# Values approach 1.0 for semantically identical texts and drop towards 0.0 for unrelated contexts.
similarity_1_2 = cosine_similarity(
    embeddings1.reshape(1, -1),
    embeddings2.reshape(1, -1)
)[0][0]

similarity_1_3 = cosine_similarity(
    embeddings1.reshape(1, -1),
    embeddings3.reshape(1, -1)
)[0][0]

print('text1 vs text2:', similarity_1_2)
print('text1 vs text3:', similarity_1_3)