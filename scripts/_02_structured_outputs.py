from pydantic import BaseModel, Field, ValidationError
from openai import OpenAI
import json

class PaperMetadata(BaseModel):
    title: str
    key_topics: list[str] = Field(max_length=3)
    is_technical: bool

client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama'
)

def generate_response(
        system_prompt: str,
        user_prompt: str,
        temperature: float
) -> str:

    messages = [
            {
                'role':'system',
                'content':system_prompt
            },
    
            {
                'role':'user',
                'content':user_prompt
            }
        ]

    try:

        response = client.chat.completions.create(
            model='qwen3:1.7b',
            messages=messages,
            temperature=temperature
        )

        answer = response.choices[0].message.content

        return answer
    
    except Exception as e:
        raise RuntimeError(f'Error during request to API: {e}') from e

def clean_response(response_text: str) -> str:
    """
    Removes common Markdown formatting from model response
    """

    response_text = response_text.strip()

    if response_text.startswith("```json"):
        response_text = response_text[7:]

    elif response_text.startswith("```"):
        response_text = response_text[3:]

    if response_text.endswith("```"):
        response_text = response_text[:-3]

    return response_text.strip()

def extract_metadata(abstract: str) -> PaperMetadata:

    system_prompt = """
    You are a data extractor for scientific papers.

    Extract metadata from the provided abstract.

    Return ONLY valid JSON.
    Do not add explanations or markdown.

    The JSON must contain exactly these fields:
    - title: string
    - key_topics: list of strings with maximum 3 topics
    - is_technical: boolean"""

    user_prompt = abstract

    max_retries = 2

    for attempt in range(max_retries + 1):

        response_text = generate_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.0
        )

        try:

            response_text = clean_response(response_text)

            metadata_dict = json.loads(response_text)

            metadata = PaperMetadata(**metadata_dict)

            return metadata

        except json.JSONDecodeError as e:

            user_prompt = f"""
            Your previous response was not valid JSON.

            JSON parsing error:
            {e}

            Please fix your response.

            Return ONLY valid JSON.
            Do not add explanations or markdown.

            The JSON must contain exactly these fields:
            - title: string
            - key_topics: list of strings with maximum 3 topics
            - is_technical: boolean"""

        except ValidationError as e:
            user_prompt = f"""
            Your previous response failed Pydantic validation.

            Validation error:
            {e}

            Please correct the response according to the validation error.

            Return ONLY valid JSON.
            Do not add explanations or markdown.

            The JSON must contain exactly these fields:
            - title: string
            - key_topics: list of strings with maximum 3 topics
            - is_technical: boolean"""

    raise ValueError(
        f'Failed to extract valid metadata after {max_retries + 1} attempts.'
    )

if __name__ == '__main__':

    abstract = """
    This paper presents a deep learning method for solving partial
    differential equations. The proposed neural network improves the
    accuracy of numerical simulations and reduces computational cost.
    """

    metadata = extract_metadata(abstract)

    print(metadata)
    