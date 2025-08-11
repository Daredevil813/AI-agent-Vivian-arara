from langchain_ollama import OllamaLLM
import streamlit as st
from bs4 import BeautifulSoup 
import requests
import faiss
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS 
from  langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
import PyPDF2


llm=OllamaLLM(model="phi")

embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

index=faiss.IndexFlatL2(384)
vector_store={}
summary_text=""

def extract_text_from_pdf(uploaded_file):
    pdf_reader= PyPDF2.PdfReader(uploaded_file)
    text=""
    for page in pdf_reader.pages:
        text+= page.extract_text()+ "\n"
    return text

def store_in_faiss(text,filename):
    global index,vector_store
    st.write("Storing data in FAISS...")

    splitter=CharacterTextSplitter(chunk_size=500,chunk_overlap=100)
    texts=splitter.split_text(text)

    #convert text into embedding
    vectors=embeddings.embed_documents(texts)
    vectors=np.array(vectors,dtype=np.float32)
    index.add(vectors)
    vector_store[len(vector_store)]=(filename,texts)
    return "Data stored succesfuly"

def generate_summary(text):
    global summary_text
    st.write("📝 Generating AI Summary...")
    summary_text = llm.invoke(f"Summarize the following document:\n\n{text[:3000]}")   
    return summary_text

def retrieve_and_answer(query):
    global index,vector_store
    query_vector=np.array(embeddings.embed_query(query),dtype=np.float32).reshape(1,-1)
    # search 
    D, I=index.search(query_vector,k=2)

    context=""
    for idx in I[0]:
        if idx in vector_store:
            context+=" ".join(vector_store[idx][1]) + "\n\n"
    if not context:
        return "no relevant data found"
    
    return llm.invoke(f"Summarize the following content:\n\n{context[:1000]}")
    
def download_summary():
    if summary_text:
        st.download_button(
            label=("Download Summary"),
            data=summary_text,
            file_name="AI_Summary.txt",
            mime="text/plain"

        )




st.title("Ai Document reader")
st.write("Upload a PDf and ask questions and get summary")

uploaded_file= st.file_uploader("Upload a PDF Documnet",type="pdf")
if uploaded_file:
    text=extract_text_from_pdf(uploaded_file)
    store_message=store_in_faiss(text,uploaded_file.name)
    st.write(store_message)
    summary=generate_summary(text)
    st.subheader("AI-Generated Summary")
    st.write(summary)

    download_summary()

    
query=st.text_input("Ask a question based on stored content")
if query:
    answer=retrieve_and_answer(query)
    st.subheader("AI Answer: ")
    st.write(answer)
