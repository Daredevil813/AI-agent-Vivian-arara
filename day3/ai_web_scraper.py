from langchain_ollama import OllamaLLM
import streamlit as st
import requests
from bs4 import BeautifulSoup 
import faiss
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS 
from  langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document


 
llm=OllamaLLM(model="mistral")

embeddiings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

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
        return text[:2000]
    except Exception as e:
        return f"error : {str(e)}"

def summarize_contnet(content):
    st.write("Summarizing content....")
    return llm.invoke(f"Summarize the following content:\n\n{content[:1000]}")

st.title("AI-Powered Web Scraper")
st.write("Enter a website URL below and get a summarized version")

url=st.text_input("Enter Website URl:")
if url:
    content=scrape_website(url)
    if "Failed" in content or "Error" in content:
        st.write(content)
    else:
        summary=summarize_contnet(content)
        st.subheader("website summary")
        st.write(summary)
