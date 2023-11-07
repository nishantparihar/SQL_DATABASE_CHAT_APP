import streamlit as st
import os
from dotenv import load_dotenv
import time


from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI

from langchain.callbacks import get_openai_callback
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import AgentType, create_sql_agent
from langchain.prompts.chat import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

#DataBase connection
from langchain import OpenAI, SQLDatabase
import pyodbc



load_dotenv()

ms_uri = f"mssql+pyodbc://{os.getenv('usernamee')}:{os.getenv('password')}@{os.getenv('host')}:{os.getenv('port')}/{os.getenv('mydatabase')}?driver=ODBC+Driver+17+for+SQL+Server"
#ms_uri = f"mssql+pyodbc://@{os.getenv('host')}:{os.getenv('port')}/{os.getenv('mydatabase')}?driver=ODBC+Driver+17+for+SQL+Server"


db = SQLDatabase.from_uri(ms_uri)



def main():

    with st.sidebar:
        
        st.title('🤗💬 LLM Chat App')
        
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



    st.title("Ask DataBase 💬")
    
    with st.chat_message('assistant'):
                st.markdown("Hi. . .Ask Query About DataBase!!!")

       
    # Initialize Streamlit chat UI
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
        
    
    if prompt := st.chat_input("Ask SQL query about your Database:"):
        
        with st.chat_message("Human"):
            st.markdown(prompt)
        
        st.session_state.chat_history.append({"role":"Human",  "content": prompt})
        llm = OpenAI(model_name='gpt-3.5-turbo', temperature=0)

        toolkit = SQLDatabaseToolkit(db=db,llm=llm)
        toolkit.get_tools()
        agent_executor = create_sql_agent(
                            #llm=llm,
                            llm=llm,
                            toolkit=toolkit,
                            handle_parsing_errors=True,
                            verbose=True,
                            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                            )

        with get_openai_callback() as cb:
            response = agent_executor.run(prompt)
            print(cb)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            for chunk in response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")
        
        message_placeholder.markdown(full_response)
        st.session_state.chat_history.append({"role":"AI",  "content": response})
            
            
            

if __name__ == '__main__':
    main()
