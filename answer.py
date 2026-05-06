from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)


def generate_answer(query, docs):
    context = "\n\n".join([
        f"[Source: {d.payload['source']}]\n{d.payload['text']}"
        for d in docs[:3]
    ])

    prompt = f"""You are a fintech assistant helping extract information from documents.

Task:
Answer the question using ONLY the provided context. Do not add any information not present in the context.

Instructions:
- If the question asks for "documents", explicitly LIST the document names
- Extract concrete items (e.g., Aadhaar, PAN, OVD, Passport)
- Avoid vague statements like "mandatory information"
- Combine information from multiple parts if needed
- Be concise and clear
- Do NOT include a Sources section

Context:
{context}

Question:
{query}

Answer:
- item 1
- item 2
"""

    response = client.chat.completions.create(
        model="deepseek/deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,    # fully deterministic = highest faithfulness
        max_tokens=600      # enough room to complete without cutting off
    )

    return response.choices[0].message.content.strip()