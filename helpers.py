from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from transformers import AutoTokenizer, T5Tokenizer, T5ForConditionalGeneration


from transformers import pipeline


def get_embeddings():
        # Define the path to the pre-trained model you want to use
    modelPath = "sentence-transformers/all-mpnet-base-v2"

    # Create a dictionary with model configuration options, specifying to use the CPU for computations
    model_kwargs = {'device':'cpu'}

    # Create a dictionary with encoding options, specifically setting 'normalize_embeddings' to False
    encode_kwargs = {'normalize_embeddings': False}

    return HuggingFaceEmbeddings(
        model_name=modelPath,  # Provide the pre-trained model's path
        model_kwargs=model_kwargs,  # Pass the model configuration options
        encode_kwargs=encode_kwargs,  # Pass the encoding options
    )


def transformer(checkpoint, config):

    # checkpoint = "google/flan-t5-base", 

    # # Load the tokenizer associated with the specified model
    # model =  T5ForConditionalGeneration.from_pretrained(checkpoint)
    # tokenizer = T5Tokenizer.from_pretrained(checkpoint, padding=True, truncation=True, max_length=512)

    # Define a question-answering pipeline using the model and tokenizer
    chain = pipeline(
        'text2text-generation',
        model = checkpoint, 
        min_length = config['min_length'], 
        # max_length = config['max_length'],
        max_new_tokens=config['max_new_tokens']
    )

    return HuggingFacePipeline(
        pipeline=chain,
        model_kwargs={"temperature": config['temperature'], "max_length": 512},
    )

# Load and preprocess file
def file_preprocessing(filepath):
    print('###  Extracting words from pdf file')
    loader = PyPDFLoader(filepath)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    return text_splitter.split_documents(pages)
