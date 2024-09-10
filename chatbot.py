from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load a conversational model and tokenizer from Hugging Face
model_name = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Create a text generation pipeline using the model and tokenizer
pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=100)

# Use HuggingFacePipeline from LangChain to integrate the pipeline
local_llm = HuggingFacePipeline(pipeline=pipeline)

# Define a prompt template
prompt_template = PromptTemplate(input_variables=["input"], template="{input}")

# Set up the chain with the prompt template
chain = LLMChain(llm=local_llm, prompt=prompt_template)

# Chat loop
print("Chatbot is ready! Type 'exit' to end the conversation.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    # Get the chatbot's response
    response = chain.run(input=user_input)
    print(f"Chatbot: {response}")
