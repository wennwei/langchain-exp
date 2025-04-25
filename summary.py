from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain.chains.summarize import load_summarize_chain
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate

def summarize_pdf(file_path):
    # Load and split PDF
    #loader = PyPDFLoader(pdf_path)
    if file_path.lower().endswith('.pdf'):
        loader = PyPDFLoader(file_path)
        documents = loader.load_and_split()
    elif file_path.lower().endswith(('.docx', '.doc')):
        # Load Word document with element-wise parsing
        loader = UnstructuredWordDocumentLoader(file_path, mode="elements")
        documents = loader.load()
    else:
        raise ValueError("Unsupported file format. Only PDF and Word documents are supported.")
    documents = loader.load_and_split()

    # Load summarization model (compatible with text2text-generation)
    model_name = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Build HuggingFace pipeline
    summarizer = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        temperature=0.3
    )

    # Wrap in LangChain
    llm = HuggingFacePipeline(pipeline=summarizer)
    custom_prompt = PromptTemplate(
    input_variables=["text"],
    template="Write a concise executive summary of the following:\n\n{text}\n\nSummary:"
    )

    chain = load_summarize_chain(llm, chain_type="map_reduce", map_prompt=custom_prompt)

    # Run the chain
    return chain.run(documents)


if __name__ == "__main__":
    pdf_summary = summarize_pdf("TD37 somthing.pdf")
    print("\nSummary:\n", pdf_summary)
