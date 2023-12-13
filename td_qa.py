import time

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import docx2txt
import os
import glob
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template

#directory_path = "C:/Users/MyPC/Desktop/ai/ask-multiple-pdfs-main/txts"
directory_path = "C:/Users/MyPC/Desktop/ai/ask-multiple-pdfs-main/dergi_soru_cevap_düzenlenmiş/"
file_list = os.listdir(directory_path)

@st.cache_data
def read_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    return content

def get_pdf_text(dirr):
    pdf_docs = glob.glob(f"{dirr}/*.pdf")
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for pdf in pdf_reader.pages:
            text += pdf.extract_text()
    return text


def read_docx(file_path):
    text = docx2txt.process(file_path)
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="---",
        chunk_size=3000,
        chunk_overlap=500,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(temperature= 0.0,
                     model_name="gpt-3.5-turbo-1106")
    #llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.5, "max_length": 512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,

    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()

    st.set_page_config(page_title="Tevhid Dergisi\n\nSoru-Cevap",
                       page_icon=":books:",layout="wide")

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Tevhid Dergisi")
    st.subheader("Yapay Zeka Soru Cevap Asistanı  :question:", "middle")
    user_question = st.text_input("Lütfen sorunuzu buraya yazınız:")
    if user_question:
        with st.spinner("Tüm veriler taranıyor..."):
            handle_userinput(user_question)

    with st.spinner("Veriler yükleniyor, lütfen bekleyiniz..."):

        for file_name in file_list:
            file_path = os.path.join(directory_path, file_name)
            # get the raw data from documents file:
            raw_text = read_file(file_path)

            # get the text chunks
            text_chunks = get_text_chunks(raw_text)

            # create vector store
            vectorstore = get_vectorstore(text_chunks)

            # create conversation chain
            st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == "__main__":
    main()
