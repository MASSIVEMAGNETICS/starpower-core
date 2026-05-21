from zai import ZaiClient

client = ZaiClient(api_key="your-api-key")  # Your API Key

response = client.chat.completions.create(
    model="glm-5",
    messages=[
        {
            "role": "user",
            "content": "As a marketing expert, please create an attractive slogan for my product.",
        },
        {
            "role": "assistant",
            "content": "Sure, to craft a compelling slogan, please tell me more about your product.",
        },
        {"role": "user", "content": "Z.AI Open Platform"},
    ],
    thinking={
        "type": "enabled",  # Optional: "disabled" or "enabled", default is "enabled"
    },
    stream=True,
    max_tokens=4096,
    temperature=0.6,
)

# Stream response
for chunk in response:
    if chunk.choices[0].delta.reasoning_content:
        print(chunk.choices[0].delta.reasoning_content, end="", flush=True)

    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
