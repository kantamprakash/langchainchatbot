import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA, StuffDocumentsChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering

# Function to load the PDF document and setup the QA chain
@st.cache_resource
def setup_qa_system(pdf_file):
    # Step 1: Load the PDF document
    loader = PyPDFLoader(pdf_file)
    documents = loader.load()

    # Step 2: Split the text into larger chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Increase chunk size for more context
        chunk_overlap=200
    )
    split_docs = text_splitter.split_documents(documents)

    # Step 3: Create embeddings for the chunks
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Step 4: Use FAISS to index the chunks with embeddings
    vectorstore = FAISS.from_documents(split_docs, embeddings)

    # Step 5: Set up the retriever to fetch relevant chunks
    retriever = vectorstore.as_retriever()

    # Step 6: Load a question-answering model (specific to QA tasks)
    model_name = "deepset/roberta-base-squad2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    hf_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer, max_length=512, truncation=True)

    # Step 7: Use HuggingFacePipeline from LangChain to wrap the pipeline
    local_llm = HuggingFacePipeline(pipeline=hf_pipeline)

    # Step 8: Define a prompt template for the LLMChain
    prompt_template = PromptTemplate(
        input_variables=["context", "question"], 
        template="Context: {context}\n\nQuestion: {question}\nAnswer:"
    )

    # Step 9: Create an LLMChain with the local LLM and prompt template
    llm_chain = LLMChain(
        llm=local_llm,
        prompt=prompt_template
    )

    # Step 10: Define how to combine documents using StuffDocumentsChain
    stuff_chain = StuffDocumentsChain(
        llm_chain=llm_chain,  # Pass the LLMChain correctly
        document_variable_name="context"  # Specify the document variable name
    )

    # Step 11: Set up the RetrievalQA chain with the StuffDocumentsChain
    qa_chain = RetrievalQA(combine_documents_chain=stuff_chain, retriever=retriever)
    
    return qa_chain

# Streamlit app
st.title("PDF QA Chatbot")

# Upload PDF
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Set up QA system with the uploaded PDF
    qa_chain = setup_qa_system(uploaded_file)

    # Chat interface
    st.header("Chat with your PDF")
    user_input = st.text_input("You:", "")

    if user_input:
        # Retrieve relevant context from the document
        retrieved_docs = qa_chain.retriever.get_relevant_documents(user_input)
        if retrieved_docs:
            # Combine contexts from the top retrieved documents
            context = " ".join([doc.page_content for doc in retrieved_docs[:5]])  # Combine up to 5 chunks for more context
            response = qa_chain.run({"context": context, "question": user_input})
        else:
            response = "Sorry, I couldn't find relevant information in the document."

        st.write(f"Chatbot: {response}")

