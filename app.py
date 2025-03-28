import streamlit as st
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

# Load AI Model
llm = OllamaLLM(model="mistral")
# Initialize Memory

if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory() # Stores user AI conversation
    
# Define AI chat prompt
prompt = PromptTemplate(
    input_variable=["chat_history", "question"],
    template="Previous Conversation: {chat_history}\nUser: {question}\nAI:"
)

# Function to run AI chat with memory
def run_chain(question):
    
    # Retrieve chat history
    chat_history_text = "\n".join([f"{msg.type.capitalize()}: {msg.content}" for msg in st.session_state.chat_history.messages])
    # Run the AI response generation
    response = llm.invoke(prompt.format(chat_history=chat_history_text, question=question))
    #store new user input and AI response in memoey
    st.session_state.chat_history.add_user_message(question)
    st.session_state.chat_history.add_ai_message(response)
    
    return response
# streamLit UI

st.title("AI Chat Agent")
st.write("Ask me anything")

user_input=st.text_input("User:")
if user_input:
    response = run_chain(user_input)
    st.write(f"User: {user_input}")
    st.write(f"AI: {response}")
    
# show full chat history
st.subheader("Chat History")







