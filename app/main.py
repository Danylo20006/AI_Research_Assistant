from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from scripts._08_langchain_basics import retriever, answer_chain, format_docs
from scripts._10_evaluation import evaluate_faithfulness
from app.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="RAG-System-Assistant",
    description="API for answering questions using a RAG system",
    version="1.0.0",
)


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    is_faithful: bool


@app.post("/ask", response_model=QueryResponse)
async def generate_response(request: QueryRequest):
    logger.info("Received question: %s", request.question)

    try:
        docs = await retriever.ainvoke(request.question)

        context = format_docs(docs)

        answer = await answer_chain.ainvoke(
            {
                "question": request.question,
                "context": context,
            }
        )

        is_faithful =  await evaluate_faithfulness(
            context=context,
            answer=answer,
        )
        logger.info("Generation completed. Faithful: %s", is_faithful)

        return QueryResponse(
            answer=answer,
            is_faithful=is_faithful,
        )

    except Exception as exc:
        logger.exception('Error during request processing')

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc
