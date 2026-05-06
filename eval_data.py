from datasets import Dataset
from retrieval import hybrid_search, rerank
from answer import generate_answer


def build_eval_dataset():
    questions = [
        "What documents are required for KYC?",
        "What is CKYCR?",
        "What is an OVD document?",
    ]

    ground_truths = [
        "KYC requires identity and address proof such as Aadhaar, PAN, passport or other officially valid documents.",
        "CKYCR is a centralized KYC records registry.",
        "OVD refers to officially valid documents used as proof of identity and address.",
    ]

    answers = []
    contexts_list = []

    for q in questions:
        docs = hybrid_search(q)
        docs = rerank(q, docs)

        ans = generate_answer(q, docs)
        ctx = [d.payload["text"] for d in docs]

        answers.append(ans)
        contexts_list.append(ctx)

    return Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths
    })


dataset = build_eval_dataset()