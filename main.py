import os

import streamlit as st
from dotenv import load_dotenv
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains.llm import LLMChain
from langchain.chains.chat_vector_db.prompts import CONDENSE_QUESTION_PROMPT
from langchain.prompts import PromptTemplate
# from langchain.chat_models import ChatOpenAI
from langchain.chat_models import AzureChatOpenAI

from langchain.document_loaders import WebBaseLoader, ConfluenceLoader
# from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import AzureOpenAIEmbeddings

from langchain.prompts.chat import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from streamlit_chat import message

import requests
from bs4 import BeautifulSoup

from llm_helper import LLMHelper
from vector_storage import add_website_to_vector_store, add_confluence_to_vector_store, init_vector_store
from customprompt import PROMPT


# Load environment variables from .env file (Optional)
load_dotenv()

CONFLUENCE_URL = os.getenv("CONFLUENCE_URL", None)

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

system_template = """Use the following pieces of context to answer the users question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
"""

messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
prompt = ChatPromptTemplate.from_messages(messages)
chain_type_kwargs = {"prompt": prompt}


def setup_ui():
    # Set the title and subtitle of the app
    st.title('🦜🔗 Chat with Senacor Website')
    st.subheader('Input your website URL, ask questions, and receive answers directly from the website.')


def get_text():
    input_text = st.text_input("You: ", "Hello, how are you?", key="input")
    return input_text


def main():
    if "messages" not in st.session_state.keys():  # Initialize the chat message history
        st.session_state.messages = [
            {"role": "assistant",
             "content": "Ask me a question about all the websites you have added to the vector store!"}
        ]
    setup_ui()

    prompt = st.text_input("Ask a question (query/prompt) about all the websites you have added to the vector store.")
    if st.button("Submit Query", type="primary"):
        llm_helper = LLMHelper()
        response = llm_helper.standard_query(prompt)
        # question, response, contextDict, sources = get_semantic_answer_lang_chain(question=prompt, chat_history=[])
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": response["result"]})

    if st.session_state.messages:
        for i in range(0, len(st.session_state['messages']), 1, ):
            if st.session_state['messages'][i]['role'] == 'user':
                message(st.session_state.messages[i]["content"], is_user=True, key=str(i) + '_user',
                        avatar_style="big-smile")
            else:
                message(st.session_state.messages[i]["content"], key=str(i))

    url = st.text_input("Insert The website URL")
    if st.button("Add Website", type="secondary"):
        add_website_to_vector_store(url)

    if CONFLUENCE_URL and st.button("Add Confluence", type="secondary"):
        add_confluence_to_vector_store()


if __name__ == '__main__':
    main()
