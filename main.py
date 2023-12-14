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
from unused_new_ui import new_main


def main():
    # Set page layout to wide screen and menu item
    menu_items = {
        'Get help': None,
        'Report a bug': None,
        'About': '''
         ## Embeddings App
         Embedding testing application.
        '''
    }
    st.set_page_config(layout="wide", menu_items=menu_items)


if __name__ == '__main__':
    main()
