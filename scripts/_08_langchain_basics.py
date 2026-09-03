from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

PDF_PATH = "./data/papers/novel.pdf"

loader = PyMuPDFLoader(PDF_PATH)

documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

split_docs = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(
    model_name='all-MiniLM-L6-v2'
)

vectorstore = Chroma.from_documents(
    documents=split_docs,
    embedding=embeddings
)

retriever = vectorstore.as_retriever(
    search_kwargs={'k': 3}
)

llm = ChatOllama(
    model='qwen3:1.7b',
    temperature=0.0
)

prompt = ChatPromptTemplate.from_messages([
    (
       'system',
        """
        You are an AI assistant.
        Use only the context provided to answer the question.
        If the context doesn't answer, say 'I don't know'.

        Context:
        {context}
        """
    ),
    (
        'human',
        '{question}'
    )
])

def format_docs(docs):
    return '\n\n'.join(doc.page_content for doc in docs)

rag_chain = (
    {
        'context': retriever | format_docs,
        'question': RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)
