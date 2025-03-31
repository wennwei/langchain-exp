from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

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

# 4. Basic text generation example
simple_prompt = "Translate English to French: Hello, how are you?"
response = text2text_llm.invoke(simple_prompt)
print("Basic Translation:", response)

# 5. Advanced usage with LangChain templates
template = """Convert the following {input_language} text to {output_language}: 
{input_text}"""
prompt = PromptTemplate(
    input_variables=["input_language", "output_language", "input_text"],
    template=template
)

# 6. Create a LangChain pipeline
translation_chain = LLMChain(
    llm=text2text_llm,
    prompt=prompt
)

# 7. Execute the chain
result = translation_chain.invoke({
    "input_language": "English",
    "output_language": "Spanish",
    "input_text": "The weather is beautiful today"
})

print("\nStructured Translation:", result["text"])

# 8. Question Answering Example
qa_template = """Answer the question based on the context:
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