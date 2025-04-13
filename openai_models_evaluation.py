import time
import pandas as pd
# Update your import to the new Ollama integration
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from ragas.metrics import faithfulness, answer_relevancy
from ragas import evaluate
from datasets import Dataset
import os
os.environ["OPENAI_API_KEY"] = "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"


## 1) More complex question sets
evaluation_set = [
    {
        "query": "What is the penalty for a late final year project submission? Are there any exceptions?",
        "context": [
            "Students are required to submit their final year projects by April 30th, 2025.",
            "Late submissions will not be accepted unless there is a valid reason."
        ],
        "ground_truth": "Late submissions are not accepted unless there's a valid reason."
    },
    {
        "query": "If a student has a CGPA of 3.8, do they qualify for the Dean’s list award? Also, when are the awards announced?",
        "context": [
            "Students with a CGPA above 3.75 in the academic year are eligible for the Dean’s list.",
            "Awards are announced at the end of each semester."
        ],
        "ground_truth": "Yes, they qualify. Awards are announced at the end of each semester."
    },
    {
        "query": "If the final year project is due on April 30th, 2025, but a student has a valid medical reason, is it possible to submit after the deadline?",
        "context": [
            "Students are required to submit their final year projects by April 30th, 2025.",
            "Late submissions will not be accepted unless there is a valid reason."
        ],
        "ground_truth": "Yes, a valid reason allows submission after the deadline."
    },
    {
        "query": "If my CGPA is 3.7, am I eligible for the Dean’s list award? Why or why not?",
        "context": [
            "Students with a CGPA above 3.75 in the academic year are eligible for the Dean’s list.",
            "Awards are announced at the end of each semester."
        ],
        "ground_truth": "No, because 3.7 is below the 3.75 threshold."
    }
]


# Prompt Template
prompt_template = PromptTemplate.from_template(
    """Answer the question using only the following context:\n{context}\n\nQuestion: {question}\nAnswer:"""
)

# Models to evaluate
models = ["llama3.2", "mistral", "deepseek-r1:7b"]

# Store all results
all_results = []

for model_name in models:
    print(f"\n🚀 Running model: {model_name}")
    llm = OllamaLLM(model=model_name)
    model_outputs = []

    for item in evaluation_set:
        context_text = "\n".join(item["context"])
        prompt = prompt_template.format(context=context_text, question=item["query"])

        start_time = time.time()
        response = llm.invoke(prompt)
        elapsed_time = time.time() - start_time

        model_outputs.append({
            "model": model_name,
            "question": item["query"],
            "ground_truth": item["ground_truth"],
            "answer": response,
            # Instead of saving Document objects, save just the strings
            "contexts": item["context"],
            "time": round(elapsed_time, 2)
        })

    # Convert to a Hugging Face Dataset
    # IMPORTANT: 'contexts' must be strings or lists of strings, not Document objects
    ragas_dataset = Dataset.from_list([
        {
            "question": r["question"],
            "answer": r["answer"],
            "contexts": r["contexts"],       # list of strings
            "ground_truth": r["ground_truth"]
        }
        for r in model_outputs
    ])

    # Evaluate with RAGAS
    print(f"📊 Evaluating with RAGAS for {model_name}...")
    ragas_scores = evaluate(ragas_dataset, metrics=[faithfulness, answer_relevancy])

    for i, r in enumerate(model_outputs):
        r["faithfulness"] = ragas_scores["faithfulness"][i]
        r["answer_relevancy"] = ragas_scores["answer_relevancy"][i]

    all_results.extend(model_outputs)

# Save to CSV
df = pd.DataFrame(all_results)
df.to_csv("rag_evaluation_results.csv", index=False)
print("\n✅ Evaluation complete. Results saved to 'rag_evaluation_results.csv'.")
