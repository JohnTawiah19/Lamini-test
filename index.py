from langchain_community.document_loaders import HuggingFaceDatasetLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import LanceDB
from transformers import AutoTokenizer, T5Tokenizer, T5ForConditionalGeneration
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
import lancedb
from transformers import pipeline
from langchain_core.prompts import PromptTemplate


# Load and preprocess file
def file_preprocessing(file):
    loader = PyPDFLoader("data/0130091154.pdf")
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    texts = text_splitter.split_documents(pages)
    # return "".join(text.page_content for text in texts)
    return texts


def transformer():

    checkpoint = "google/flan-t5-base", 

    # # Load the tokenizer associated with the specified model
    # model =  T5ForConditionalGeneration.from_pretrained(checkpoint)
    # tokenizer = T5Tokenizer.from_pretrained(checkpoint, padding=True, truncation=True, max_length=512)

    # Define a question-answering pipeline using the model and tokenizer
    chain = pipeline(
        'summarization',
        model = "google/flan-t5-base", 
        max_new_tokens=50
    )

    # Create an instance of the HuggingFacePipeline, which wraps the question-answering pipeline
    # with additional model-specific arguments (temperature and max_length)
    llm = HuggingFacePipeline(
    pipeline=chain,
    model_kwargs={"temperature": 0.9, "max_length": 512},)
    return llm






def main():

    # Define the path to the pre-trained model you want to use
    modelPath = "sentence-transformers/all-MiniLM-l6-v2"

    # Create a dictionary with model configuration options, specifying to use the CPU for computations
    model_kwargs = {'device':'cpu'}

    # Create a dictionary with encoding options, specifically setting 'normalize_embeddings' to False
    encode_kwargs = {'normalize_embeddings': False}

    # Initialize an instance of HuggingFaceEmbeddings with the specified parameters
    embeddings = HuggingFaceEmbeddings(
        model_name=modelPath,     # Provide the pre-trained model's path
        model_kwargs=model_kwargs, # Pass the model configuration options
        encode_kwargs=encode_kwargs # Pass the encoding options
    )

    # # Initialise our vector database 
    db = lancedb.connect("/tmp/lancedb")

    docs =   file_preprocessing('')
    question = "In 100 words what is software development"

    table = db.create_table("my_table", data=[{
        "vector": embeddings.embed_query("".join(text.page_content for text in docs)),
        "text": "".join(doc.page_content for doc in docs),
        "id": "1", }],
        mode="overwrite", )

    db = LanceDB.from_documents(docs, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 5})
    docs = retriever.get_relevant_documents(question)
    print(''.join(text.page_content for text in docs))
    
    llm = transformer()

    # Create a question-answering instance (qa) using the RetrievalQA class.
    # It's configured with a language model (llm), a chain type "refine," the retriever we created, and an option to not return source documents.
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="refine", retriever=retriever, return_source_documents=False)

    result = qa.invoke({"query": question})
    print(result["result"])
    # return result["result"]

    # query = "What is Testing"
    # results = retriever.get_relevant_documents(query)
    # print("".join(result.page_content for result in results))


if __name__ == '__main__':
    main()