from langchain_community.vectorstores import LanceDB
from langchain.chains import RetrievalQA
import lancedb
import streamlit as st

from helpers import get_embeddings, transformer, file_preprocessing
from ui import displayPDF

config={
    'temperature': 0.6,
    'max_new_tokens': 300,
    'min_length': 150,
    # 'max_length': 300,
    'device': 'cpu'
}

def run(model, question, filepath):
    # Initialise our vector database 
    db = lancedb.connect("/tmp/lancedb")

    docs =   file_preprocessing(filepath)
    embeddings = get_embeddings()
    # question = "Explain Bayesian regret"

    print('### Creating vector store ###')
    db.create_table("my_table", data=[{
        "vector": embeddings.embed_query("".join(text.page_content for text in docs)),
        "text": "".join(doc.page_content for doc in docs),
        "id": "1", }],
        mode="overwrite", )

    print('### Creating vector embedding in vector store ###')
    db = LanceDB.from_documents(docs, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 5})
    
    print('### Performing a similarity search on query ###')
    docs = retriever.get_relevant_documents(question)    
    
    llm = transformer(model, config)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="refine", retriever=retriever, return_source_documents=False)
    print('### Retrieve output from LLM ###')
    result = qa.invoke({"query": question})
    # print(result["result"])
    return result["result"]


def output(uploaded_file, option, input_text):
    # col1,col2 = st.columns(2)

    filepath = f"data/{uploaded_file.name}"
    with open(filepath, 'wb') as temp_file:
        temp_file.write(uploaded_file.read())
    # st.info('Uploaded PDF file')
    # displayPDF(filepath)
    # with col2: 
    st.info("Response")  
    output = run(option,input_text, filepath )
    st.success(output)

# Streamlit 
st.set_page_config(layout='wide', page_title="Lamini Test")

def main():
    st.title('Custom Transformer For Textbook Extraction')

    uploaded_file = st.file_uploader("Upload your PDF file", type= ['pdf']) 

    with st.sidebar:
        option = st.selectbox(
            'Pick a model from the options',
            ('google/flan-t5-base', 'MBZUAI/LaMini-Flan-T5-248M  ' ))

        config['temperature'] = st.slider('Select Temperature', 0.0, 1.0, (config['temperature']))
        config['min_length']= st.number_input('Select min token length',value=config['min_length'], placeholder="Type a number...")
        # config['max_length']= st.number_input('Select max token length',value=config['max_length'], placeholder="Type a number...")
        config['max_new_tokens']= st.number_input('Select max new token length',value= config['max_new_tokens'], placeholder="Type a number...")

    if uploaded_file is not None:
        if input_text := st.text_input('Make request', ''):
            output(uploaded_file, option, input_text)

if __name__ == '__main__':
    main()