from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_PATH = "./data/papers/novel.pdf"

loader = PyMuPDFLoader(PDF_PATH)
documents = loader.load()

# Recursive character splitting is superior to naive slicing. 
# It attempts to split by paragraphs (\n\n), then sentences, then words,
# preserving logical semantic boundaries before relying on hard character limits.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

split_docs = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
)

vectorstore = Chroma.from_documents(
    documents=split_docs,
    embedding=embeddings,
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3},
)

llm = ChatOllama(
    model="qwen3:1.7b",
    temperature=0.0,
    base_url="http://host.docker.internal:11434",
    timeout=120,
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are an AI assistant.
            Answer the question using only the provided context.
            If the context does not contain enough information to answer the question,
            say "I don't know".
            Do not use external knowledge.
            
            Context:
            {context}
            """,
        ),
        ("human", "{question}"),
    ]
)

def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

# LCEL (LangChain Expression Language) Pipeline
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

# Architectural highlight: Decoupling the generation chain from the retrieval chain.
# This allows the API layer to intercept the retrieved 'context' and pass it 
# directly to the evaluator (LLM-as-a-Judge) alongside the generated answer.
answer_chain = (
    prompt
    | llm
    | StrOutputParser()
)