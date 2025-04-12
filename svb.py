from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import warnings


warnings.filterwarnings("ignore")

# 1. Load and process PDF
loader = PyPDFLoader("svb-review-20230428.pdf")
pages = loader.load_and_split()

# 2. Chunk documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False
)
texts = text_splitter.split_documents(pages)

# 3. Initialize embeddings and vector store
from langchain_community.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.from_documents(texts, embeddings)

# 4. Configure LLM
model_id = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512,
    temperature=0.3
)
llm = HuggingFacePipeline(pipeline=pipe)

# 5. Create RAG prompt template
rag_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""Use the SVB case study context to answer:
    Context: {context}
    Question: {question}
    Answer in bullet points with financial specifics:"""
)

# 6. Build retrieval chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    chain_type_kwargs={"prompt": rag_prompt},
    return_source_documents=True
)

# Example usage
query = "What were SVB's key risk management failures?"
result = qa_chain.invoke({"query": query})
print(result["result"])

from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    answer_correctness,
    faithfulness,
    context_recall,
    context_precision
)
from ragas.llms import HuggingFaceLLM  # Local model integration

from datasets import Dataset
import pandas as pd
import matplotlib.pyplot as plt

# 1. Define test questions and ground truths
test_questions = {
    "questions": [
        "What were SVB's key risk management failures?",
        "What was the total deposit outflow SVB experienced before collapse?",
        "How did SVB's investment in HTM securities contribute to its downfall?"
    ],
    "ground_truths": [
        ["Lack of interest rate hedging", "Overconcentration in VC deposits", 
         "Poor liquidity risk management", "Failure to diversify client base"],
        ["$42 billion in single day", 
         "$100 billion total projected outflow"],
        ["$91 billion in HTM securities with 1.56% yield", 
         "Unrealized losses of $15 billion from rate hikes"]
    ]
}

# 2. Generate answers and collect contexts
def generate_test_results(qa_chain, questions):
    results = []
    for q in questions:
        result = qa_chain.invoke({"query": q})
        results.append({
            "question": q,
            "answer": result["result"],
            "contexts": [doc.page_content for doc in result["source_documents"]]
        })
    return results

test_results = generate_test_results(qa_chain, test_questions["questions"])

# 3. Prepare evaluation dataset
eval_dataset = {
    "question": [r["question"] for r in test_results],
    "answer": [r["answer"] for r in test_results],
    "contexts": [r["contexts"] for r in test_results],
    "ground_truths": test_questions["ground_truths"]
}

dataset = Dataset.from_dict(eval_dataset)

# 4. Configure and run evaluation

metrics = [
    answer_relevancy,
    answer_correctness,
    faithfulness,
    context_recall,
    context_precision
]

local_llm = HuggingFaceLLM(
    model_name="google/flan-t5-large",  # Example alternative
    tokenizer_name="google/flan-t5-large",
    device="cuda"  # Use "cpu" if no GPU
)

# Attach local LLM to metrics
answer_relevancy.llm = local_llm
answer_correctness.llm = local_llm
faithfulness.llm = local_llm

evaluation_result = evaluate(dataset, metrics=metrics)

# 5. Visualize results
df = evaluation_result.to_pandas()
df.set_index('question', inplace=True)

# Calculate averages
avg_scores = df.mean().to_frame().T
avg_scores.index = ['Average']

# Combine and display
combined_df = pd.concat([df, avg_scores])
print(combined_df)

# Visualization
plt.figure(figsize=(10,6))
combined_df.drop('Average').plot(kind='bar', stacked=True)
plt.title('RAG Evaluation Metrics per Question')
plt.ylabel('Score (0-1)')
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()


