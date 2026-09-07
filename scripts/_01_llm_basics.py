from openai import OpenAI

# Initialize the client to route traffic from the Docker container 
# to the locally hosted Ollama instance on the host machine.
client = OpenAI(
    base_url='http://host.docker.internal:11434',
    api_key='ollama'
)

def generate_response(
    system_prompt: str,
    user_prompt: str,
    temperature: float
) -> str:
    """
    Generates a response from the local LLM using the OpenAI API spec.
    
    Args:
        system_prompt: Defines the persona and constraints for the model.
        user_prompt: The specific query or task.
        temperature: Controls output determinism (0.0 for strict facts, higher for creativity).
    """
    messages = [
        {
            'role': 'system',
            'content': system_prompt
        },
        {
            'role': 'user',
            'content': user_prompt
        }
    ]

    try:
        response = client.chat.completions.create(
            model='qwen3:1.7b',
            messages=messages,
            temperature=temperature
        )
        return response.choices[0].message.content

    except Exception as e:
        return f'Error during request to API: {e}'

if __name__ == '__main__':
    system_prompt_example = 'You are useful assistant'
    user_prompt_example = 'Explain what does mean Machine Learning in one sentence'

    response_1 = generate_response(system_prompt=system_prompt_example, user_prompt=user_prompt_example, temperature=0.0)
    response_2 = generate_response(system_prompt=system_prompt_example, user_prompt=user_prompt_example, temperature=1.0)

    print("Temperature 0.0:\n", response_1)
    print("\nTemperature 1.0:\n", response_2)