from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA, StuffDocumentsChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering

# Step 1: Load the PDF document
loader = PyPDFLoader("bitcoin.pdf")  # Use the correct path to your PDF
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

# Step 8: Define a custom function to handle the question-answering task
def answer_question(context, question):
    try:
        # Ensure the context is provided correctly
        result = hf_pipeline({'context': context, 'question': question})
        return result['answer']
    except Exception as e:
        return f"An error occurred while processing the question: {str(e)}"

# Step 9: Define a prompt template for the LLMChain
prompt_template = PromptTemplate(
    input_variables=["context", "question"], 
    template="Context: {context}\n\nQuestion: {question}\nAnswer:"
)

# Step 10: Create an LLMChain with the local LLM and prompt template
llm_chain = LLMChain(
    llm=local_llm,
    prompt=prompt_template
)

# Step 11: Define how to combine documents using StuffDocumentsChain
stuff_chain = StuffDocumentsChain(
    llm_chain=llm_chain,  # Pass the LLMChain correctly
    document_variable_name="context"  # Specify the document variable name
)

# Step 12: Set up the RetrievalQA chain with the StuffDocumentsChain
qa_chain = RetrievalQA(combine_documents_chain=stuff_chain, retriever=retriever)

# Chat loop to interact with the QA system
print("QA system is ready! Type 'exit' to end the conversation.")
while True:
    try:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Exiting the chat. Goodbye!")
            break

        # Retrieve relevant context from the document
        retrieved_docs = retriever.get_relevant_documents(user_input)
        if retrieved_docs:
            # Combine contexts from the top retrieved documents
            context = " ".join([doc.page_content for doc in retrieved_docs[:5]])  # Combine up to 2 chunks for more context
            response = answer_question(context, user_input)
        else:
            response = "Sorry, I couldn't find relevant information in the document."

        print(f"Chatbot: {response}")

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Let's continue chatting. Please try again.")
