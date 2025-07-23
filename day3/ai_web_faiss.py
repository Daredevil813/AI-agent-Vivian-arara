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


 
llm=OllamaLLM(model="mistral")

embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

index=faiss.IndexFlatL2(384)
vector_store={}

#fcutnion to scrape 

def scrape_website(url):
    try:
        st.write(f"Scraping website{url}")
        headers={"User-Agent": "Mozilla/5.0"}
        response=requests.get(url,headers=headers)

        if response.status_code!=200:
            return f"failed to fetch url"
        soup=BeautifulSoup(response.text,"html.parser")
        paragraph=soup.find_all("p")
        text=" ".join([p.get_text() for p in paragraph])
        return text[:5000]
    except Exception as e:
        return f"error : {str(e)}"


def store_in_faiss(text,url):
    global index,vector_store
    st.write("Storing data in FAISS...")

    splitter=CharacterTextSplitter(chunk_size=500,chunk_overlap=100)
    texts=splitter.split_text(text)

    #convert text into embedding
    vectors=embeddings.embed_documents(texts)
    vectors=np.array(vectors,dtype=np.float32)
    index.add(vectors)
    vector_store[len(vector_store)]=(url,texts)
    return "Data stored succesfuly"



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
    
    return llm.invoke(f"Summarize the following content:\n\n{content[:1000]}")
    



st.title("AI-Powered Web Scraper with fais")
st.write("Enter a website URL below and get a summarized version")

url=st.text_input("Enter Website URl:")
if url:
    content=scrape_website(url)
    if "Failed" in content or "Error" in content:
        st.write(content)
    else:
        store_message=store_in_faiss(content,url)
        st.write(store_message)

query=st.text_input("Ask a question based on stored content")
if query:
    answer=retrieve_and_answer(query)
    st.subheader("AI Answer: ")
    st.write(answer)

# https://en.wikipedia.org/wiki/Artificial_intelligence