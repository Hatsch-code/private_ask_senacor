import os

import streamlit as st
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
#from langchain.chat_models import ChatOpenAI
from langchain.chat_models import AzureChatOpenAI

from langchain.document_loaders import WebBaseLoader
#from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import AzureOpenAIEmbeddings

from langchain.prompts.chat import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.vectorstores.azuresearch import AzureSearch
from streamlit_chat import message

# Load environment variables from .env file (Optional)
load_dotenv()

#OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")



vector_store_address: str = os.getenv("YOUR_AZURE_SEARCH_ENDPOINT")
vector_store_password: str = os.getenv("YOUR_AZURE_SEARCH_ADMIN_KEY")

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


def init_vector_store(embeddings, index_name="langchain-vector-demo"):
    index_name: str = index_name
    vector_store: AzureSearch = AzureSearch(
        azure_search_endpoint=vector_store_address,
        azure_search_key=vector_store_password,
        index_name=index_name,
        embedding_function=embeddings.embed_query,
    )
    return vector_store


def add_website_to_vector_store(url, vector_store):
    # Load data from the specified URL
    loader = WebBaseLoader(url)
    data = loader.load()
    # Split the loaded data
    text_splitter = CharacterTextSplitter(separator='\n',
                                          chunk_size=500,
                                          chunk_overlap=40)

    docs = text_splitter.split_documents(data)
    vector_store.add_documents(documents=docs)


def print_semantic_similarity(question, vector_store, k=3, search_type="similarity"):
    docs = vector_store.similarity_search(
        query=question,
        k=k,
        search_type=search_type,
    )
    print(docs[0].page_content)


# Simple QA
def standard_query(question, k=3, model_name="gpt-3.5-turbo"):
    # Create OpenAI embeddings
    #openai_embeddings = OpenAIEmbeddings()
    openai_embeddings = AzureOpenAIEmbeddings(
        azure_deployment="AskSenacor-ada002-v1",
        openai_api_version="2023-05-15",
    )
    # Init Vector Store
    vector_store = init_vector_store(openai_embeddings)

    # Create a retriever from the Chroma vector database
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    # Use a ChatOpenAI model
    #llm = ChatOpenAI(model_name=model_name)
    llm = AzureChatOpenAI(
        azure_deployment="AskSenacor-gpt35turbo-v1",
        openai_api_version="2023-05-15",
    )

    # Create a RetrievalQA from the model and retriever
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    return qa(question)


def get_text():
    input_text = st.text_input("You: ","Hello, how are you?", key="input")
    return input_text


def main():
    if "messages" not in st.session_state.keys():  # Initialize the chat message history
        st.session_state.messages = [
            {"role": "assistant", "content": "Ask me a question about all the websites you have added to the vector store!"}
        ]
    setup_ui()

    prompt = st.text_input("Ask a question (query/prompt) about all the websites you have added to the vector store.")
    if st.button("Submit Query", type="primary"):
        response = standard_query(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": response["result"]})

    if st.session_state.messages:
        for i in range(0, len(st.session_state['messages']), 1,):
            if st.session_state['messages'][i]['role'] == 'user':
                message(st.session_state.messages[i]["content"], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
            else:
                message(st.session_state.messages[i]["content"], key=str(i))

    url = st.text_input("Insert The website URL")
    if st.button("Add Website", type="secondary"):
        vector_store = init_vector_store()
        add_website_to_vector_store(url, vector_store)


if __name__ == '__main__':
    main()
