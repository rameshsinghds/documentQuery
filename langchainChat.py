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
    # gllm = Groq(model="mixtral-8x7b-32768", api_key='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', temperature=0.0)
    # gllm = Ollama(model="mistral")
    gllm = ChatGroq(model="llama3-8b-8192", temperature=0.0000000000001, seed=3242)

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

    # qa = RetrievalQAWithSourcesChain.from_chain_type(llm=gllm,
    #                         retriever=retriever,
    #                         return_source_documents=True,
    #                         verbose=True,
    #                         chain_type_kwargs={"prompt": prompt,"verbose":True})
    return qa

    # What is the role of the Company Secretarial Department with respect to the disclosure policy in Canara Bank

# The code snippet you provided is using Streamlit to create a web application interface.
# st.markdown(
#     f"""
#     <div class="container" style="display: flex">
#     <span style="float: left;"><h3>Welcome to Small Finance Bank</h3></span>
#  <!--   <span style="float: right;"><img style=”float: right;" size="70%"; src="data:image/png;base64,{base64.b64encode(open("./esaf/esaf_logo.png", "rb").read()).decode()}"></span> -->
#     </div>
#     """,
#     unsafe_allow_html=True
# )

# st.title("Welcome to ESAF Small Finance Bank")

# Initialize chat history
if "messages" not in st.session_state:
    st.chat_message("assistant").markdown("Hi! How may I help you today?")
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input

if prompt := st.chat_input("Please type your query related to Bank Policies?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            rqe = load_and_return_query_engine()
            response = rqe.invoke({"query":prompt})
            # with st.stdout("info"):
            #     print(response)
            st.markdown(response['result'])
    # Add assistant response to chat history
    print(response)
    st.session_state.messages.append({"role": "assistant", "content": response['result']})