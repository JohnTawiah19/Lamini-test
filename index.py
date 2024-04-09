from langchain_community.document_loaders import HuggingFaceDatasetLoader
from langchain_community.vectorstores import LanceDB
from langchain.chains import RetrievalQA
import lancedb
import streamlit as st
import base64

from helpers import get_embeddings, transformer, file_preprocessing
from ui import displayPDF

config={
    'temperature': 0.7,
    'max_new_tokens': 300,
    'min_length': 150,
    'max_length': 300,
    'device': 'cpu'
}

def run(model, question):
     # Initialise our vector database 
    db = lancedb.connect("/tmp/lancedb")

    docs =   file_preprocessing('')
    embeddings = get_embeddings()
    # question = "Explain Bayesian regret"

    db.create_table("my_table", data=[{
        "vector": embeddings.embed_query("".join(text.page_content for text in docs)),
        "text": "".join(doc.page_content for doc in docs),
        "id": "1", }],
        mode="overwrite", )

    db = LanceDB.from_documents(docs, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 5})
    docs = retriever.get_relevant_documents(question)
    
    llm = transformer(model)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="refine", retriever=retriever, return_source_documents=False)

    result = qa.invoke({"query": question})
    print(result["result"])
    return result["result"]


def output(uploaded_file, option, input_text):
    col1,col2 = st.columns(2)

    filepath = f"data/{uploaded_file.name}"
    with open(filepath, 'wb') as temp_file:
        temp_file.write(uploaded_file.read())

    with col1:
        
        st.info('Uploaded PDF file')
        displayPDF(filepath)
    with col2:   
        output = run(option,input_text )
        st.success(output)

# Streamlit 
st.set_page_config(layout='wide', page_title="Lamini Test")

def main():
    st.title('Lamini Test')

    uploaded_file = st.file_uploader("Upload your PDF file", type= ['pdf']) 

    with st.sidebar:
        option = st.selectbox(
            'Pick a model from the options',
            ('google/flan-t5-base'))
        
    if uploaded_file is not None:
        btn_col1,  btn_col3 = st.columns([8,2])

        with btn_col1:
            input_text = st.text_input('Make request', '')

            if(input_text):
                output(uploaded_file, option, input_text)

if __name__ == '__main__':
    main()