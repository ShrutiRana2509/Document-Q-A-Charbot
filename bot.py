import os
import time
import streamlit as st

from langchain_groq import ChatGroq
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.vectorstores.faiss import FAISS
from langchain.document_loaders import PyPDFDirectoryLoader


os.environ["GROQ_API_KEY"] = "YOUR_API_KEY"

llm = ChatGroq(
    groq_api_key=os.environ["GROQ_API_KEY"],
    model="llama-3.3-70b-versatile"
)


prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the context provided below.
Be concise and accurate.

<context>
{context}

Question: {input}
""")



st.markdown("""
    <style>
        .main {padding-top: 2rem;}
        h1 {text-align: center;}
        .stTextInput>div>div>input {font-size: 16px;}
        .response-box {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid #dfe3e8;
        }
    </style>
""", unsafe_allow_html=True)

st.title("AI PDF Chatbot")
st.caption("Ask questions based on the content of your local PDF documents.")

with st.sidebar:
    st.header("Document Setup")
    st.write("Place your PDF files in the `data/` directory.")
    if st.button("Embed Documents"):
        with st.spinner("Embedding documents..."):
            try:
                st.session_state.embeddings = GPT4AllEmbeddings()
                loader = PyPDFDirectoryLoader("data")
                docs = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                split_docs = splitter.split_documents(docs)
                st.session_state.vectors = FAISS.from_documents(split_docs, st.session_state.embeddings)
                st.success("Vector database ready.")
            except Exception as e:
                st.error(f"Failed to embed documents: {e}")

st.subheader("Ask a Question")
user_prompt = st.text_input("Enter your question:", placeholder="e.g., What are the main conclusions of the report?")

if user_prompt:
    if "vectors" not in st.session_state:
        st.warning("Please embed documents using the sidebar first.")
    else:
        with st.spinner("Generating response..."):
            try:
                doc_chain = create_stuff_documents_chain(llm, prompt)
                retriever = st.session_state.vectors.as_retriever()
                chain = create_retrieval_chain(retriever, doc_chain)

                start = time.process_time()
                response = chain.invoke({'input': user_prompt})
                end = time.process_time()

                answer = response.get("answer", "No answer found.")

                st.markdown(" Answer:")
                st.markdown(f"<div class='response-box'>{answer}</div>", unsafe_allow_html=True)
                st.caption(f"Response time: {end - start:.2f} seconds")

    

            except Exception as e:
                st.error(f"Error generating response: {e}")


st.caption("@copyright by Shruti rana")
