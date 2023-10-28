import streamlit as st
import os
from dotenv import load_dotenv
import time

import pickle
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks import get_openai_callback
from langchain.memory import ConversationBufferMemory



import pytesseract
from pdf2image import convert_from_path, convert_from_bytes

#pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'



@st.cache_data
def get_text_chunks(pdf):
    pdf = convert_from_bytes(pdf.read())
    text = ""

    for page in pdf:
        text += pytesseract.image_to_string(page,lang='eng')
    #st.write(text)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    chunks = text_splitter.split_text(text=text)
    return chunks



@st.cache_data
def get_vector_store(store_name, chunks):
    if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
            
    else:
        embeddings = OpenAIEmbeddings()
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        with open(f"{store_name}.pkl", "wb") as f:
            pickle.dump(VectorStore, f)

    return VectorStore





def main():

   
    with st.sidebar:
        
        st.title('🤗💬 LLM Chat App')
        
        tab1, tab2, tab3 = st.tabs(["Try it!!", "How to??", "About"])

        with tab1:
            # upload a PDF file
            pdf = st.file_uploader("Upload your PDF", type='pdf')   

        with tab2:
            st.markdown('''
            ## How to Use ChatPDF App
    
            1. Upload a PDF.
            2. Ask questions in the "Ask a query" field.
            3. Enjoy the conversation.
    
            The app provides answers based on the PDF content.
            ''')

        with tab3:
            st.markdown('''
                ## About
                This app is an LLM-powered chatbot built using:
                - [Streamlit](https://streamlit.io/)
                - [LangChain](https://python.langchain.com/)
                - [OpenAI](https://platform.openai.com/docs/models) LLM model
                ''')

            st.markdown('''
                ## Developed by [Nishant Singh Parihar](https://nishantparihar.github.io/)
                ''')



    st.title("Chat with PDF 💬")
    
    with st.chat_message('assistant'):
                st.markdown("Hi. . . Upload File Lets Talk!!!")

    load_dotenv()

    if pdf is None:
        st.session_state.clear()

    
    if pdf is not None:

        store_name = pdf.name[:-4]

        chunks = get_text_chunks(pdf)
        VectorStore = get_vector_store(store_name, chunks)
        
        
        # Initialize Streamlit chat UI
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
            
        
        if prompt := st.chat_input("Ask a query about your PDF:"):
            
            with st.chat_message("Human"):
                st.markdown(prompt)
            
            llm = OpenAI(model_name='gpt-3.5-turbo')
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            chain = ConversationalRetrievalChain.from_llm(llm, VectorStore.as_retriever(), memory=memory)

            with get_openai_callback() as cb:
                #response = chain({"input_documents": docs, "human_input": query}, return_only_outputs=True)
                response = chain({'question':prompt})
                print(cb)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""

                for chunk in response["answer"].split():
                    full_response += chunk + " "
                    time.sleep(0.05)
                    # Add a blinking cursor to simulate typing
                    message_placeholder.markdown(full_response + "▌")
                
            message_placeholder.markdown(full_response)
            for msg in response["chat_history"]:
                st.session_state.messages.append({"role": msg.type, "content": msg.content})
            
            
         

if __name__ == '__main__':
    main()
