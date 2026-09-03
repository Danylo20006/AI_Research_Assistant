from scripts._08_langchain_basics import rag_chain
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title='RAG-System-Assistant',
    description='API for future answering the specific questions with AI',
    version='1.0.0'
)


class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str


@app.post('/ask')
async def generate_response(request: QueryRequest):
    try:
        
        answer = await rag_chain.ainvoke(request.question)

        

        return QueryResponse(answer=answer)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))