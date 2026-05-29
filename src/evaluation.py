import json
import os
import pandas as pd

from dotenv import load_dotenv

from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    HallucinationMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
)

from deepeval.test_case import LLMTestCase
from deepeval.models import DeepEvalBaseLLM

from langchain_groq import ChatGroq

from openpyxl import load_workbook
from openpyxl.styles import Alignment

load_dotenv()


# =========================================================
# Custom Groq Evaluator
# =========================================================

class GroqEvaluator(DeepEvalBaseLLM):

    def __init__(self):

        self.model = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.3-70b-versatile",
            temperature=0
        )

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        return self.model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        return self.model.invoke(prompt).content

    def get_model_name(self):
        return "llama-3.3-70b-versatile"


# =========================================================
# Load Output JSON
# =========================================================

with open("data/results/output.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Convert single object to list
if isinstance(data, dict):
    data = [data]


# =========================================================
# Initialize Evaluator LLM
# =========================================================

evaluator_llm = GroqEvaluator()


# =========================================================
# Metrics
# =========================================================

answer_metric = AnswerRelevancyMetric(
    threshold=0.7,
    model=evaluator_llm
)

faithfulness_metric = FaithfulnessMetric(
    threshold=0.7,
    model=evaluator_llm
)

hallucination_metric = HallucinationMetric(
    threshold=0.5,
    model=evaluator_llm
)

context_precision_metric = ContextualPrecisionMetric(
    threshold=0.7,
    model=evaluator_llm
)

context_recall_metric = ContextualRecallMetric(
    threshold=0.7,
    model=evaluator_llm
)

context_relevancy_metric = ContextualRelevancyMetric(
    threshold=0.7,
    model=evaluator_llm
)


# =========================================================
# Store Results
# =========================================================

results = []


# =========================================================
# Evaluate Each QA Pair
# =========================================================

for item in data:

    question = item["question"]
    answer = item["answer"]

    context_list = [
        chunk["content"]
        for chunk in item["retrieved_chunks"]
    ]

    print("\n===================================")
    print("QUESTION:", question)
    print("===================================")

    # -----------------------------------------------------
    # Build Test Case
    # -----------------------------------------------------

    test_case = LLMTestCase(

        input=question,

        actual_output=answer,

        # Needed for Context Precision / Recall
        expected_output=answer,

        # Needed for Hallucination
        context=context_list,

        # Needed for RAG metrics
        retrieval_context=context_list,
    )

    # -----------------------------------------------------
    # Run Metrics
    # -----------------------------------------------------

    answer_metric.measure(test_case)

    faithfulness_metric.measure(test_case)

    hallucination_metric.measure(test_case)

    context_precision_metric.measure(test_case)

    context_recall_metric.measure(test_case)

    context_relevancy_metric.measure(test_case)

    # -----------------------------------------------------
    # Save Result Row
    # -----------------------------------------------------

    results.append({

        # Basic
        "Question": question,

        "Answer": answer,

        "Context": "\n\n-------------------------\n\n".join(context_list),

        # =========================================
        # Answer Relevancy
        # =========================================

        "Answer Relevancy Score":
            round(answer_metric.score, 3),

        "Answer Relevancy Passed":
            answer_metric.is_successful(),

        "Answer Relevancy Reason":
            answer_metric.reason,

        # =========================================
        # Faithfulness
        # =========================================

        "Faithfulness Score":
            round(faithfulness_metric.score, 3),

        "Faithfulness Passed":
            faithfulness_metric.is_successful(),

        "Faithfulness Reason":
            faithfulness_metric.reason,

        # =========================================
        # Hallucination
        # =========================================

        "Hallucination Score":
            round(hallucination_metric.score, 3),

        "Hallucination Passed":
            hallucination_metric.is_successful(),

        "Hallucination Reason":
            hallucination_metric.reason,

        # =========================================
        # Context Precision
        # =========================================

        "Context Precision Score":
            round(context_precision_metric.score, 3),

        "Context Precision Passed":
            context_precision_metric.is_successful(),

        "Context Precision Reason":
            context_precision_metric.reason,

        # =========================================
        # Context Recall
        # =========================================

        "Context Recall Score":
            round(context_recall_metric.score, 3),

        "Context Recall Passed":
            context_recall_metric.is_successful(),

        "Context Recall Reason":
            context_recall_metric.reason,

        # =========================================
        # Context Relevancy
        # =========================================

        "Context Relevancy Score":
            round(context_relevancy_metric.score, 3),

        "Context Relevancy Passed":
            context_relevancy_metric.is_successful(),

        "Context Relevancy Reason":
            context_relevancy_metric.reason,
    })


# =========================================================
# Create DataFrame
# =========================================================

df = pd.DataFrame(results)


# =========================================================
# Save Excel
# =========================================================

output_excel = "data/results/evaluation_results.xlsx"

df.to_excel(output_excel, index=False)


# =========================================================
# Format Excel
# =========================================================

wb = load_workbook(output_excel)

ws = wb.active


# Wrap Text
for row in ws.iter_rows():

    for cell in row:

        cell.alignment = Alignment(
            wrap_text=True,
            vertical="top"
        )


# Auto Column Width
for column_cells in ws.columns:

    max_length = 0

    column = column_cells[0].column_letter

    for cell in column_cells:

        try:
            if cell.value:

                max_length = max(
                    max_length,
                    len(str(cell.value))
                )

        except:
            pass

    adjusted_width = min(max_length + 5, 50)

    ws.column_dimensions[column].width = adjusted_width


# Increase Row Height
for row in ws.iter_rows():

    ws.row_dimensions[row[0].row].height = 80


# Save Workbook
wb.save(output_excel)


# =========================================================
# Final Message
# =========================================================

print("\n===================================")
print("Evaluation Completed Successfully")
print("Excel Saved At:")
print(output_excel)
print("===================================")

