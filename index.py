from langchain.document_loaders import HuggingFaceDatasetLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import LanceDB
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import AutoTokenizer, pipeline
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA

# Load and preprocess file
def file_preprocessing(file):
    loader = PyPDFLoader("data/Agile-PM-101-Beginners-Guide-Non-PM-Ebook-download-open.pdf")
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(pages)
    return texts[0].page_content
    # return [text.page_content for text in texts]

def main():
    file_preprocessing()


if __name__ == '__main__':
    main()