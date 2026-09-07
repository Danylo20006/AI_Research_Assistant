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
    logger.info("STEP 1: Received question: %s", request.question)

    try:
        logger.info("STEP 2: Starting retriever")
        docs = await retriever.ainvoke(request.question)
        logger.info("STEP 3: Retriever completed, docs=%s", len(docs))

        context = format_docs(docs)
        logger.info("STEP 4: Context formatted, length=%s", len(context))

        logger.info("STEP 5: Starting answer generation")
        answer = await answer_chain.ainvoke(
            {
                "question": request.question,
                "context": context,
            }
        )
        logger.info("STEP 6: Answer generated")

        logger.info("STEP 7: Starting faithfulness evaluation")
        is_faithful = await evaluate_faithfulness(
            context=context,
            answer=answer,
        )
        logger.info("STEP 8: Evaluation completed: %s", is_faithful)

        return QueryResponse(
            answer=answer,
            is_faithful=is_faithful,
        )

    except Exception as exc:
        logger.exception("ERROR in /ask")
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc
