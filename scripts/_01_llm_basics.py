from openai import OpenAI


client = OpenAI(
    base_url='http://host.docker.internal:11434',
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
        return f'Error during request to API: {e}'

if __name__ == '__main__':

    system_prompt_example = 'You are useful assistant'
    user_prompt_example = 'Explain what does mean Machine Learning in one sentence'

    response_1 = generate_response(system_prompt=system_prompt_example, user_prompt=user_prompt_example, temperature=0.0)

    response_2 = generate_response(system_prompt=system_prompt_example, user_prompt=user_prompt_example, temperature=1.0)

    print("Temperature 0.0:")
    print(response_1)

    print("\nTemperature 1.0:")
    print(response_2)
