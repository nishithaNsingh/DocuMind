from ragas import evaluate
from ragas.metrics import faithfulness, context_precision, context_recall
from ragas.llms import llm_factory
from openai import OpenAI
import os
from eval_data import dataset
# Use OpenRouter via OpenAI-compatible client
from dotenv import load_dotenv
load_dotenv()
load_dotenv(dotenv_path="../.env")

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

llm = llm_factory(
    "deepseek/deepseek-chat",
    client=client
)

result = evaluate(
    dataset,
    metrics=[
        faithfulness,
        context_precision,
        context_recall
    ],
    llm=llm
)

print("\n📊 RAGAS Evaluation Results:\n")
print(result)