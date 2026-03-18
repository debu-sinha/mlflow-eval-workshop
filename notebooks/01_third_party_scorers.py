# Databricks notebook source
# MAGIC %md
# MAGIC # Module 1: Connecting External Evaluation Libraries to MLflow
# MAGIC
# MAGIC **Objective:** Set up and run third-party scorers inside `mlflow.genai.evaluate()`
# MAGIC using Databricks notebooks with MLflow tracking.
# MAGIC
# MAGIC **Tools:** MLflow, Phoenix (Arize), TruLens, Guardrails AI
# MAGIC
# MAGIC **Time:** 15 minutes

# COMMAND ----------

# MAGIC %pip install mlflow[genai] arize-phoenix trulens guardrails-ai -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.1 Sample evaluation dataset
# MAGIC
# MAGIC We start with a simple dataset of LLM responses to evaluate.

# COMMAND ----------

eval_dataset = [
    {
        "inputs": {"question": "What is MLflow?"},
        "outputs": "MLflow is an open-source platform for managing the ML lifecycle, including experiment tracking, model packaging, and deployment.",
        "expectations": {"expected_response": "MLflow is an open-source platform for the complete machine learning lifecycle."},
    },
    {
        "inputs": {"question": "What causes hallucinations in LLMs?"},
        "outputs": "LLM hallucinations occur when the model generates text that sounds plausible but is factually incorrect. Common causes include training data gaps, insufficient grounding in retrieved context, and overconfident token prediction.",
        "expectations": {"expected_response": "Hallucinations happen when LLMs generate plausible but incorrect information due to training limitations."},
    },
    {
        "inputs": {"question": "How does RAG work?"},
        "outputs": "RAG (Retrieval-Augmented Generation) combines a retriever that fetches relevant documents from a knowledge base with a generator (LLM) that produces answers grounded in the retrieved context. This reduces hallucinations by anchoring responses in real data.",
        "expectations": {"expected_response": "RAG retrieves relevant documents and uses them as context for LLM generation."},
    },
    {
        "inputs": {"question": "What is the capital of France?"},
        "outputs": "The capital of France is Berlin.",
        "expectations": {"expected_response": "The capital of France is Paris."},
    },
    {
        "inputs": {"question": "Explain fine-tuning an LLM."},
        "outputs": "Fine-tuning involves training a pre-trained LLM on a smaller, domain-specific dataset to adapt its behavior for a particular task. Techniques include full fine-tuning, LoRA, and QLoRA, each trading off compute cost against adaptation quality.",
        "expectations": {"expected_response": "Fine-tuning adapts a pre-trained model to a specific task using additional training on domain data."},
    },
]

print(f"Evaluation dataset: {len(eval_dataset)} samples")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.2 Phoenix scorer: Hallucination detection
# MAGIC
# MAGIC Phoenix (by Arize) provides LLM-based evaluation for hallucination and relevance.

# COMMAND ----------

from mlflow.genai.scorers.phoenix import Hallucination

hallucination_scorer = Hallucination(model="databricks-claude-sonnet-4")

# Test on a single sample
feedback = hallucination_scorer(
    inputs=eval_dataset[0]["inputs"],
    outputs=eval_dataset[0]["outputs"],
    expectations=eval_dataset[0]["expectations"],
)
print(f"Hallucination check: {feedback.value}")
print(f"Rationale: {feedback.rationale}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.3 TruLens scorer: Groundedness
# MAGIC
# MAGIC TruLens evaluates whether the output is grounded in the provided context.

# COMMAND ----------

from mlflow.genai.scorers.trulens import Groundedness

groundedness_scorer = Groundedness(model="databricks-claude-sonnet-4")

feedback = groundedness_scorer(
    inputs=eval_dataset[2]["inputs"],
    outputs=eval_dataset[2]["outputs"],
    expectations=eval_dataset[2]["expectations"],
)
print(f"Groundedness: {feedback.value}")
print(f"Score: {feedback.metadata.get('score', 'N/A')}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.4 Guardrails AI scorer: Toxicity detection (no LLM needed)
# MAGIC
# MAGIC Guardrails AI provides deterministic validators. No LLM call required.

# COMMAND ----------

from mlflow.genai.scorers.guardrails import ToxicLanguage

toxicity_scorer = ToxicLanguage()

feedback = toxicity_scorer(
    outputs="This is a helpful and informative response.",
)
print(f"Toxicity check: {feedback.value}")

# Test with potentially problematic text
feedback_bad = toxicity_scorer(
    outputs="You are an idiot and should be fired immediately.",
)
print(f"Toxicity check (bad): {feedback_bad.value}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.5 Run all scorers together with `mlflow.genai.evaluate()`
# MAGIC
# MAGIC This is the key integration point. All three scorers run from a single call.

# COMMAND ----------

import mlflow

mlflow.set_experiment("/odsc-eval-workshop/module-1-scorers")

results = mlflow.genai.evaluate(
    data=eval_dataset,
    scorers=[
        Hallucination(model="databricks-claude-sonnet-4"),
        Groundedness(model="databricks-claude-sonnet-4"),
        ToxicLanguage(),
    ],
)

print(f"\nEvaluation complete. Run ID: {results.run_id}")
print(f"\nMetrics:")
for metric_name, metric_value in results.metrics.items():
    print(f"  {metric_name}: {metric_value}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.6 View results in MLflow UI
# MAGIC
# MAGIC Navigate to the MLflow Experiment page to see:
# MAGIC - Per-sample scores for each scorer
# MAGIC - Aggregate metrics
# MAGIC - The evaluation trace showing which samples passed or failed
# MAGIC
# MAGIC The sample that claims "The capital of France is Berlin" should be flagged
# MAGIC by the hallucination scorer.

# COMMAND ----------

# Display the results DataFrame
display(results.result_df)
