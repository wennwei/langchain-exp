from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import warnings

warnings.filterwarnings("ignore")

# 3. Initialize the text2text model
model_id = "google/flan-t5-base"  # Lightweight text2text model
task = "text2text-generation"

text2text_llm = HuggingFacePipeline.from_model_id(
    model_id=model_id,
    task=task,
    pipeline_kwargs={
        "max_length": 150,
        "temperature": 0.7,
        "repetition_penalty": 1.2
    }
)
# 8. Question Answering Example
qa_template = """Answer the question based on the context and following examples:

Example 1:
Context: LangChain supports integration with multiple LLM providers
Question: What LLM providers does LangChain work with?
Answer: LangChain works with various providers like OpenAI, HuggingFace, and Anthropic.

Example 2:
Context: Chains in LangChain can sequence models or use different prompting techniques
Question: What are LangChain chains used for?
Answer: Chains help sequence models and implement complex prompting strategies.

Now answer this question:
Context: {context}
Question: {question}
Answer:"""

qa_prompt = PromptTemplate(
    template=qa_template,
    input_variables=["context", "question"]
)

qa_chain = LLMChain(
    llm=text2text_llm,
    prompt=qa_prompt
)

qa_response = qa_chain.invoke({
    "context": "LangChain is a framework for developing applications powered by language models",
    "question": "What is LangChain used for?"
})

print("\nQA Response:", qa_response["text"])

prompt = "what is the capital of france?"
response = text2text_llm.invoke(prompt)
print(response)

