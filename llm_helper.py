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

from vector_storage import add_website_to_vector_store, add_confluence_to_vector_store, init_vector_store
from customprompt import PROMPT


class LLMHelper:
    def __init__(self, custom_prompt="", temperature=0.7):
        self.llm = AzureChatOpenAI(
            azure_deployment="AskSenacor-gpt35turbo-v1",
            openai_api_version="2023-05-15",
        )
        self.prompt = PROMPT if custom_prompt == '' else PromptTemplate(template=custom_prompt,
                                                                        input_variables=["summaries", "question"])
        self.vector_store = init_vector_store(
            embeddings=AzureOpenAIEmbeddings(
                azure_deployment="AskSenacor-ada002-v1",
                openai_api_version="2023-05-15",
            ),
            index_name="langchain-vector-demo")

    def get_semantic_answer_lang_chain(self, question, chat_history):
        question_generator = LLMChain(llm=self.llm, prompt=CONDENSE_QUESTION_PROMPT, verbose=False)
        doc_chain = load_qa_with_sources_chain(self.llm, chain_type="stuff", verbose=False, prompt=self.prompt)
        chain = ConversationalRetrievalChain(
            retriever=self.vector_store.as_retriever(),
            question_generator=question_generator,
            combine_docs_chain=doc_chain,
            return_source_documents=True,
            # top_k_docs_for_context= self.k
        )
        result = chain({"question": question, "chat_history": chat_history})
        sources = "\n".join(set(map(lambda x: x.metadata["source"], result['source_documents'])))

        # container_sas = self.blob_client.get_container_sas()

        contextDict = {}
        # for res in result['source_documents']:
        #     source_key = self.filter_sourcesLinks(
        #         res.metadata['source'].replace('_SAS_TOKEN_PLACEHOLDER_', container_sas)).replace('\n', '').replace(' ',
        #                                                                                                             '')
        #     if source_key not in contextDict:
        #         contextDict[source_key] = []
        #     myPageContent = self.clean_encoding(res.page_content)
        #     contextDict[source_key].append(myPageContent)
        #
        # result['answer'] = \
        # result['answer'].split('SOURCES:')[0].split('Sources:')[0].split('SOURCE:')[0].split('Source:')[0]
        # result['answer'] = self.clean_encoding(result['answer'])
        # sources = sources.replace('_SAS_TOKEN_PLACEHOLDER_', container_sas)
        # sources = self.filter_sourcesLinks(sources)

        return question, result['answer'], contextDict, sources

    # Simple QA
    def standard_query(self, question, k=3, model_name="gpt-3.5-turbo"):
        # Create a retriever from the Chroma vector database
        retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
        # Create a RetrievalQA from the model and retriever
        qa = RetrievalQA.from_chain_type(llm=self.llm, chain_type="stuff", retriever=retriever)
        return qa(question)