from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from scripts._08_langchain_basics import retriever, answer_chain, format_docs
from scripts._10_evaluation import evaluate_faithfulness
from app.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="RAG-System-Assistant",
    description="API for answering questions using a RAG system with LLM-as-a-Judge evaluation",
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
        # Asynchronous invocation prevents blocking the FastAPI event loop during I/O operations
        docs = await retriever.ainvoke(request.question)
        logger.info("STEP 3: Retriever completed, docs=%s", len(docs))

        context = format_docs(docs)
        logger.info("STEP 4: Context formatted, length=%s", len(context))

        logger.info("STEP 5: Starting answer generation")
        # We decouple retrieval from generation (instead of a single LCEL chain) 
        # to capture the intermediate 'context' string for the evaluator.
        answer = await answer_chain.ainvoke(
            {
                "question": request.question,
                "context": context,
            }
        )
        logger.info("STEP 6: Answer generated")

        logger.info("STEP 7: Starting faithfulness evaluation")
        # LLM-as-a-Judge intercepts the response to ensure no hallucinations occurred 
        # before returning the final payload to the client.
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
        # logger.exception automatically captures and logs the full stack trace
        logger.exception("ERROR in /ask")
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc