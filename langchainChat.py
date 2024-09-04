__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import base64
import os
from dotenv import load_dotenv

load_dotenv()

from langchain_community.embeddings import HuggingFaceBgeEmbeddings

from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

#llms
from langchain_groq import ChatGroq

@st.cache_resource
def load_and_return_query_engine():
    gllm = ChatGroq(model="llama3-8b-8192", api_key=os.getenv("GROQ_API_KEY"), temperature=0.0000000000001, seed=3242)

    embed_model = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-large-en-v1.5", model_kwargs={"device":"cpu"}, encode_kwargs={'normalize_embeddings': True})

    vec_store = Chroma(embedding_function=embed_model,
                       persist_directory="esaf_llamaparsed_chroma_db",
                       collection_name="rag")
    retriever=vec_store.as_retriever(search_kwargs={'k': 10})

    custom_prompt_template = """Use the following pieces of information to answer the user's question. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    

    Context: {context}
    Question: {question}
    Also translate the response to Hindi & Kannada. If any word can not be translated use it as is.
    Helpful answer:
    """

    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])

    qa = RetrievalQA.from_chain_type(llm=gllm,
                               retriever=retriever,
                               return_source_documents=True,
                               verbose=True,
                               chain_type_kwargs={"prompt": prompt,"verbose":True})
    return qa

if "messages" not in st.session_state:
    st.chat_message("assistant").markdown("Hi! How may I help you today?")
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("Please type your query related to Bank Policies?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            rqe = load_and_return_query_engine()
            response = rqe.invoke({"query":prompt})
            st.markdown(response['result'])
    print(response)
    st.session_state.messages.append({"role": "assistant", "content": response['result']})