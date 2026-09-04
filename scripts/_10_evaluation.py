import time
from openai import AsyncOpenAI
import asyncio
from app.logger import get_logger

logger = get_logger(__name__)

client = AsyncOpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

JUDGE_MODEL = "qwen2.5:1.5b-instruct"


async def evaluate_faithfulness(
    context: str,
    answer: str,
) -> bool:
    
    system_prompt = """
        You are a strict faithfulness classifier.

        Your task is to determine whether the answer is fully supported by the context.

        Rules:

        1. Return 1 if every factual statement in the answer is supported by the context.
        2. Return 0 if at least one factual statement is not supported by the context.
        3. Do not use external knowledge.
        4. Do not infer facts that are not explicitly supported by the context.
        5. Ignore grammar and style.
        6. Return only one character: 1 or 0.
        """

    user_prompt = f"""
        Context:
        {context}
        
        Answer:
        {answer}
        
        Classification:
        """

    start = time.perf_counter()

    response = await client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=0.0,
        max_tokens=1,
    )

    latency = time.perf_counter() - start

    result = response.choices[0].message.content.strip()

    logger.info(f"Evaluation latency: {latency:.3f}s")

    if result == "1":
        return True

    if result == "0":
        return False

    raise ValueError(
        f"Unexpected judge response: {result!r}"
    )

async def main():
    context = """
        After the apocalypse, 5 people were left in the bunker.
        They settled inside the bunker and stocked up on water.
        """

    answer1 = "5 people were left."
    answer2 = "5 people were left, and they live in New York."

    result1 = await evaluate_faithfulness(
        context=context,
        answer=answer1,
    )

    result2 = await evaluate_faithfulness(
        context=context,
        answer=answer2,
    )

    assert result1 is True
    assert result2 is False

    logger.info("Test 1: %s", result1)
    logger.info("Test 2: %s", result2)


if __name__ == "__main__":
    asyncio.run(main())
