from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from rouge_score import rouge_scorer
import nltk

# Only needed for sentence splitting (optional)
#nltk.download('punkt')

# 1. Load model and tokenizer
model_id = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

# 2. Set up inference pipeline
llm = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=256)

# 3. Few-shot prompt
few_shot_prompt = """Answer the questions based on the given examples.

Q: What is the capital of France?
A: Paris

Q: Who wrote 'Pride and Prejudice'?
A: Jane Austen

Q: What are SVB’s risk management failures?
A:"""

# 4. Generate response
response = llm(few_shot_prompt)[0]["generated_text"]
print("LLM Response:", response)

# 5. Ground truth
ground_truth = "Lack of interest rate hedging, overconcentration in VC deposits, poor liquidity risk management."

# 6. Evaluate with ROUGE
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
scores = scorer.score(ground_truth, response)

# 7. Print scores
print("\nROUGE Evaluation Scores:")
for metric, score in scores.items():
    print(f"{metric.upper()} - Precision: {score.precision:.2f}, Recall: {score.recall:.2f}, F1: {score.fmeasure:.2f}")
