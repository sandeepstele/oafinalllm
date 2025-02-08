import openai

# Set your API key (hard-coded for testing) and API base.
openai.api_key = ""
openai.api_base = "https://aiproxy.sanand.workers.dev/openai/v1"

def chat_gpt(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

# Example usage:
if __name__ == "__main__":
    prompt = "What is 2 + 2?"
    answer = chat_gpt(prompt)
    print("Assistant:", answer)