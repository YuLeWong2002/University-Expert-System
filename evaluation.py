import os
import time
import subprocess
import pandas as pd

from pydantic import Field
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate

from ragas.metrics import faithfulness, answer_relevancy
from ragas import evaluate
from datasets import Dataset

class OllamaCLI:
    """
    A lightweight class that invokes the Ollama CLI with a specified model.
    """
    def __init__(self, model: str):
        self.model = model

    def invoke(self, prompt: str) -> str:
        command = ["ollama", "run", self.model, prompt]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.stderr:
            print("stderr:", result.stderr)
        return result.stdout.strip()


def main():
    ########################################
    # 1) Load your existing FAISS index
    ########################################
    # Make sure OPENAI_API_KEY is set for embeddings usage, if needed
    os.environ["OPENAI_API_KEY"] = "sk-proj-eDLFgajgvm7k_9E4..."

    faiss_dir = "faiss_index"  # The folder containing your FAISS index files
    print(f"Loading FAISS index from: {faiss_dir}")

    # Use the same embedding model that was used to create the original index
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # IMPORTANT: add allow_dangerous_deserialization=True if necessary
    vector_store = FAISS.load_local(faiss_dir, embeddings, allow_dangerous_deserialization=True)

    ########################################
    # 2) Define your question set
    ########################################
    evaluation_set = [
        {
            "query": "What is the Quality Manual, and who is responsible for it across all campuses?",
            "ground_truth": "It's a framework for teaching and learning, owned by the Quality and Standards Committee."
        },
        {
            "query": "How can someone propose a change or correction to the Quality Manual?",
            "ground_truth": "They must submit a paper to the Quality and Standards Committee (QSC), which must approve all changes."
        },
        {
            "query": "How is an external examiner for a taught programme appointed, and how long can they serve?",
            "ground_truth": "They are nominated annually by each School and typically serve up to four years."
        },
        {
            "query": "What responsibilities does an External Assessor (EA) have for Apprenticeship Integrated End-Point Assessments?",
            "ground_truth": "They verify the quality of the End-Point Assessment, ensure procedures are fair, and endorse the final outcomes."
        },
        {
            "query": "What is the difference between an External Assessor (EA) and an Independent Assessor (IA) for Apprenticeship Integrated End-Point Assessments?",
            "ground_truth": "Both must be independent, but an EA oversees quality assurance, while an IA carries out the actual End-Point Assessment."
        }
    ]

    prompt_template = PromptTemplate.from_template(
        """Answer the question using only the following context:
{context}

Question: {question}
Answer:"""
    )

    models = ["deepseek-r1:7b"]  # or multiple models if desired
    all_results = []

    for model_name in models:
        print(f"\n🚀 Running model: {model_name}")
        llm = OllamaCLI(model=model_name)
        model_outputs = []

        # Evaluate each question
        for item in evaluation_set:
            query = item["query"]
            ground_truth = item["ground_truth"]

            # 4a) Retrieve top docs from the loaded FAISS store
            top_k = 3
            docs_and_scores = vector_store.similarity_search_with_score(query, k=top_k)

            # Build a context string from those top docs
            final_context = ""
            if docs_and_scores:
                final_context = "\n\n".join(doc.page_content for doc, _ in docs_and_scores)
            else:
                final_context = "No relevant context found."

            # 4b) Format prompt
            prompt = prompt_template.format(context=final_context, question=query)

            # 4c) Invoke the model
            start_time = time.time()
            response = llm.invoke(prompt)
            elapsed_time = time.time() - start_time

            # For debugging or reference, let's keep short snippet from each doc
            context_snippets = [doc.page_content[:200] for doc, _ in docs_and_scores]

            # Save Q&A
            model_outputs.append({
                "model": model_name,
                "question": query,
                "ground_truth": ground_truth,
                "answer": response,
                "contexts": context_snippets,
                "time": round(elapsed_time, 2)
            })

        print(f"📊 Evaluating with RAGAS for {model_name}...")
        ragas_dataset = Dataset.from_list([
            {
                "question": r["question"],
                "answer": r["answer"],
                "contexts": r["contexts"],  # must be list of strings
                "ground_truth": r["ground_truth"]
            }
            for r in model_outputs
        ])

        ragas_scores = evaluate(ragas_dataset, metrics=[faithfulness, answer_relevancy])

        # Attach the faithfulness and answer relevancy scores
        for i, r in enumerate(model_outputs):
            r["faithfulness"] = ragas_scores["faithfulness"][i]
            r["answer_relevancy"] = ragas_scores["answer_relevancy"][i]

        all_results.extend(model_outputs)

    df = pd.DataFrame(all_results)
    df.to_csv("rag_evaluation_results.csv", index=False)
    print("\n✅ Evaluation complete. Results saved to 'rag_evaluation_results.csv'.")

if __name__ == "__main__":
    main()
