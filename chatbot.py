from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# Load the PDF document
loader = PyPDFLoader("bitcoin.pdf")
documents = loader.load()

# Create embeddings for the documents
embeddings = HuggingFaceEmbeddings()

# Create a vector store for retrieval
vectorstore = FAISS.from_documents(documents, embeddings)

# Create a retrieval-based QA system
retriever = vectorstore.as_retriever()

# Load a conversational model (replace with a suitable one)
model_name = "deepset/roberta-base-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
hf_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

# Use HuggingFacePipeline from LangChain to integrate the pipeline
local_llm = HuggingFacePipeline(pipeline=hf_pipeline)

# Set up the RetrievalQA chain
qa_chain = RetrievalQA(llm=local_llm, retriever=retriever)

# Chat loop
print("QA system is ready! Type 'exit' to end the conversation.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    # Get the QA system's response
    response = qa_chain.run({"query": user_input})
    print(f"Chatbot: {response}")
