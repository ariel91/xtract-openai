from PIL import Image as PILImage
from utils.utils import image_export_to_html
from tempfile import TemporaryDirectory
from pathlib import Path
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import os

from langchain.llms.openai import OpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

# Define constants
PDF_FILE = "../data/amwins/pdf/contract_demo.pdf"
TEXT_FILE = "../data/amwins/txt/contract_demo.txt"
IMAGE_OUTPUT_DIR = "../data/amwins/img/"
API_KEY = "API-KEY"
QUERY = "What is the contract value?"

def convert_pdf_to_images(pdf_file, output_dir, dpi=500):
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Convert PDF to images
    pdf_pages = convert_from_path(pdf_file, dpi)
    
    image_file_list = []
    for i, page in enumerate(pdf_pages):
        # Create a file name to store the image
        filename = f"{output_dir}/page_{i}.jpg"
        
        # Save the image of the page
        page.save(filename, "JPEG")
        
        image_file_list.append(filename)
    
    return image_file_list

def extract_text_from_images(image_files, text_file):
    with open(text_file, "w") as output_file:
        for image_file in image_files:
            # Recognize the text as a string in the image using pytesseract
            text = pytesseract.image_to_string(Image.open(image_file))
            
            # Clean up the text
            text = text.replace("-\n", "")
            
            # Write the processed text to the file
            output_file.write(text)

def run_qa_query(query, text_file):
    # Load the text documents
    loader = TextLoader(text_file, encoding='utf8')
    documents = loader.load()
    
    # Split the documents into smaller chunks
    text_splitter = CharacterTextSplitter(chunk_size=1_000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    
    # Use GPT3.5 Embeddings
    embeddings = OpenAIEmbeddings()
    
    # Load vector embeddings into Vector DB
    db = Chroma.from_documents(texts, embeddings)
    
    # Set up Chroma as a retriever
    retriever = db.as_retriever()
    
    # Set up RetrievalQA
    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever)
    
    # Perform the query
    result = qa.run(query).lstrip()
    
    return result

def main():
    # Convert PDF to images
    image_files = convert_pdf_to_images(PDF_FILE, IMAGE_OUTPUT_DIR)
    
    # Extract text from images
    extract_text_from_images(image_files, TEXT_FILE)
    
    # Perform the QA query
    result = run_qa_query(QUERY, TEXT_FILE)
    
    # Print the result
    print(result)
    
    print("Finalizing Process")

if __name__ == "__main__":
    os.environ['OPENAI_API_KEY'] = API_KEY  # Reemplaza con tu clave de API real
    main()
