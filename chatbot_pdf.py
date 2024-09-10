from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import pdfplumber

# Load a smaller conversational model and tokenizer from Hugging Face
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Create a text generation pipeline using the model and tokenizer
hf_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=50)

# Use HuggingFacePipeline from LangChain Community to integrate the pipeline
local_llm = HuggingFacePipeline(pipeline=hf_pipeline)

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

# Read the PDF content
pdf_path = "bitcoin.pdf"  # Replace with your PDF file path
pdf_content = extract_text_from_pdf(pdf_path)

# Split PDF content into chunks for processing
texts = pdf_content.split('. ')  # Split content into sentences

# Initialize the embeddings model with a smaller model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")

# Create a FAISS vector store using the extracted texts and embeddings
vector_store = FAISS.from_texts(texts, embedding_model)

# Set up a RetrievalQA chain with the vector store and LLM
qa_chain = RetrievalQA.from_chain_type(llm=local_llm, retriever=vector_store.as_retriever())

# Chat loop
print("Chatbot is ready! Type 'exit' to end the conversation.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    # Get the chatbot's response based on the PDF content
    response = qa_chain.run(user_input)
    print(f"Chatbot: {response}")
