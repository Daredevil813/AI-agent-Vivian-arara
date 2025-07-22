import speech_recognition as sr
import pyttsx3
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
import streamlit as st

#load model 
llm=OllamaLLM(model="mistral")

if "chat_history" not in st.session_state:
    st.session_state.chat_history=ChatMessageHistory()

#text to s peech 
engine=pyttsx3.init()
engine.setProperty("rate",160)

#speech recognition
recognizer=sr.Recognizer()

#function to speak

def speak(text):
    engine.say(text)
    engine.runAndWait()

#function to listen 

def listen():
    with sr.Microphone() as source:
        print("\n Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio=recognizer.listen(source)
    try:
        query=recognizer.recognize_google(audio)
        print(f"you said {query}")
        return query.lower()
    except sr.UnknownValueError:
        print("sorry I cant understand ")
        query="skip"
        return query
    except sr.RequestError:
        print("speech recognition system unavailable")
        query="skip"
        return query
    
prompt=PromptTemplate(
    input_variables=['chat_history','question'],
    template="Previous conversation: {chat_history}\n User: {question}\n AI:"
)
    
#function to process ai 
def run_chain(question):
    chat_history_text="\n".join([f"{msg.type.capitalize()}: {msg.content}" for msg in st.session_state.chat_history.messages])
    response=llm.invoke(prompt.format(chat_history=chat_history_text,question=question))
    st.session_state.chat_history.add_ai_message(response)
    st.session_state.chat_history.add_user_message(question)
    return response

st.title("Ai voice assistant ")
st.write("Click button below to speak to your Ai assistant ")

if st.button("Start listening"):
    query=listen()
    ai_response=run_chain(query)
    st.write(f"you: {query}")
    st.write(f"Ai: {ai_response}")
    speak(ai_response)

st.subheader("chat history ")
for msg in st.session_state.chat_history.messages:
    st.write(f"** {msg.type.capitalize()}**:{msg.content}")






