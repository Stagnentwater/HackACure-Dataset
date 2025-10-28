from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    answer_correctness,
    context_relevancy,
    faithfulness,
)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from app import config
from app.rag_pipeline import answer_query


def main():
    test_file = Path(__file__).resolve().parent / "test_queries.json"
    data = json.loads(test_file.read_text(encoding="utf-8"))

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for item in data:
        q = item["question"]
        gt = item.get("ground_truth", "")
        ans, ctx = answer_query(q, top_k=item.get("top_k", config.DEFAULT_TOP_K))
        questions.append(q)
        answers.append(ans)
        contexts.append(ctx)
        ground_truths.append(gt)

    df = pd.DataFrame({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })
    dataset = Dataset.from_pandas(df)

    llm = ChatGoogleGenerativeAI(model=config.LLM_MODEL, google_api_key=config.GOOGLE_API_KEY)
    emb = GoogleGenerativeAIEmbeddings(model=config.EMBEDDING_MODEL, google_api_key=config.GOOGLE_API_KEY)

    result = evaluate(
        dataset=dataset,
        metrics=[answer_relevancy, answer_correctness, context_relevancy, faithfulness],
        llm=llm,
        embeddings=emb,
    )
    print(result)


if __name__ == "__main__":
    main()
