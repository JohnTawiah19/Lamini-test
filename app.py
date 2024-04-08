import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, GPT2Tokenizer 
from transformers import pipeline
import torch
import base64
from store import Store
# Model and Tokenizer
checkpoint = 'google/flan-t5-base'
sentences = ''

tokenizers = {
    'LaMini-Flan-T5-248M': T5Tokenizer,
    'LaMini-GPT-1.5B': GPT2Tokenizer,
    'LaMini-T5-738M': T5Tokenizer,
    'LaMini-Neo-1.3B': T5Tokenizer
}


store = Store(checkpoint)
store.create()



# Load and preprocess file
def file_preprocessing(file):
    loader = PyPDFLoader(file)
    # The `pages` variable is being used to store the result of loading and splitting the content of
    # the PDF file. It is obtained by loading the PDF file using the PyPDFLoader and then splitting
    # the content into individual pages using the text splitter. The `pages` variable will contain the
    # text content of each page of the PDF document after the splitting process.
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(pages)
    return [text.page_content for text in texts]
    
def summarise_pipeline(checkpoint, filepath):
    # sourcery skip: inline-immediately-returned-variable
    # tokenizer = tokenizers[checkpoint].from_pretrained(checkpoint)
    # base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, device_map = 'auto', torch_dtype=torch.float32 )
    # pipe_sum = pipeline(  # noqa: F841
    #     'summarization',
    #     model = base_model,
    #     tokenizer = tokenizer,

    # )
    
    sentences = file_preprocessing(filepath)
    query = 'Summarize the text'
    output = store.load(sentences, query, filepath)
    # result = pipe_sum(sentences)
    # result = result[0]['summary_text']
    return output[0]['vector']



def generation_pipeline(checkpoint, filepath, prompt):
    tokenizer = tokenizers[checkpoint].from_pretrained(checkpoint)
    base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, device_map = 'auto', torch_dtype=torch.float32 )
    pipe_gen = pipeline(
        'text2text-generation',
        model = 'google/flan-t5-base', 
        # tokenizer = tokenizer,
        # do_sample=True,
        # max_length = 4000,
        # min_length = 1000
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

    with st.sidebar:
        option = st.selectbox(
            'Pick a model from the options',
            ('LaMini-Flan-T5-248M', 'LaMini-T5-738M', 'LaMini-GPT-1.5B','LaMini-Neo-1.3B'))

    if uploaded_file is not None:
        btn_col1,  btn_col3 = st.columns([1,9])

        with btn_col1:
            summarize_btn = st.button('Summarize')


        with btn_col3:
            input_text = st.text_input('Make request', '')

        if input_text:
            llm_view(uploaded_file, option, input_text)
        if summarize_btn:
            col1,col2 = st.columns(2)
            filepath = f"data/{uploaded_file.name}"
            with open(filepath, 'wb') as temp_file:
                temp_file.write(uploaded_file.read())

            with col1:
                st.info('Uploaded PDF file')
                displayPDF(filepath)

            with col2:
                st.info('Summary below')
                summary = summarise_pipeline(option, filepath)
                st.success(summary)


def llm_view(uploaded_file, option, input_text):
    col1,col2 = st.columns(2)
    filepath = f"data/{uploaded_file.name}"
    print(filepath)
    with open(filepath, 'wb') as temp_file:
        temp_file.write(uploaded_file.read())

    with col1:
        st.info('Uploaded PDF file')
        displayPDF(filepath)

    with col2:
        st.info('Response below')
        summary = generation_pipeline(option, filepath, input_text)
        st.success(summary)
                    
if __name__ == '__main__':
    main()