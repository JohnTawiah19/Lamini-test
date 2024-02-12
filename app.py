import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
# from langchain.chains.summarize import load_summarize_chain
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline
import torch
import base64

# Model and Tokenizer
checkpoint = 'LaMini-Flan-T5-248M'
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, device_map = 'auto', torch_dtype=torch.float32 )

# Load and preprocess file
def file_preprocessing(file):
    loader = PyPDFLoader(file)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    texts = text_splitter.split_documents(pages)
    final_texts = ""
    for text in texts:
        final_texts = final_texts + text.page_content
    return final_texts

def summarise_pipeline(filepath):
    pipe_sum = pipeline(
        'summarization',
        model = base_model,
        tokenizer = tokenizer,
        max_length = 4000,
        min_length = 1000
    )
    
    input_text = file_preprocessing(filepath)
    result = pipe_sum(input_text)
    result = result[0]['summary_text']
    return result


def generation_pipeline(filepath, prompt):
    pipe_gen = pipeline(
        'text2text-generation',
        model = base_model, 
        tokenizer = tokenizer,
        do_sample=True,
        max_length = 4000,
        min_length = 1000
    )
    
    input_text = file_preprocessing(filepath)
    
    input_prompt = f"From this data:{input_text}. Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{prompt}\n\n### Response:"
    result = pipe_gen(input_prompt)
    result = result[0]['generated_text']
    return result

@st.cache_data
def displayPDF(file):
    with open(file, 'rb') as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        
        # Embed PDF in HTML
        pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
        
        # Display File
        st.markdown(pdf_display, unsafe_allow_html=True)
        
# Streamlit 
st.set_page_config(layout='wide', page_title="Lamini Test")


def main():
    
    st.title('Lamini Test')   
    
    uploaded_file = st.file_uploader("Upload your PDF file", type= ['pdf']) 
    
    if uploaded_file is not None:
        btn_col1,  btn_col3 = st.columns([1,9])
        
        with btn_col1:
            summarize_btn = st.button('Summarize')
            

        with btn_col3:
            input_text = st.text_input('Make request', '')
        
        if input_text:
            col1,col2 = st.columns(2)
            filepath = "data/" + uploaded_file.name
            with open(filepath, 'wb') as temp_file:
                temp_file.write(uploaded_file.read())
                
            with col1:
                st.info('Upload PDF file')
                displayPDF(filepath)
                
            with col2:
                st.info('Response below')
                summary = generation_pipeline(filepath, input_text)
                st.success(summary)
                
        if summarize_btn:
                col1,col2 = st.columns(2)
                filepath = "data/" + uploaded_file.name
                with open(filepath, 'wb') as temp_file:
                    temp_file.write(uploaded_file.read())
                    
                with col1:
                    st.info('Upload PDF file')
                    displayPDF(filepath)
                    
                with col2:
                    st.info('Summary below')
                    summary = summarise_pipeline(filepath)
                    st.success(summary)
                    
                
    
if __name__ == '__main__':
    main()