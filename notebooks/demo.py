from PIL import Image as PILImage
from utils.utils import image_export_to_html

# Import libraries
import platform
from tempfile import TemporaryDirectory
from pathlib import Path
 
import pytesseract
from pdf2image import convert_from_path
from PIL import Image

PDF_file = "../data/amwins/pdf/contract_demo.pdf"
# read pdf to image
pdf_pages = convert_from_path(PDF_file, 500)

#Convert pdf to jpg
image_file_list = []
for page_enumeration, page in enumerate(pdf_pages, start=0):
    # enumerate() "counts" the pages for us.
    # Create a file name to store the image
    filename = f"../data/amwins/img/page_{page_enumeration}.jpg"
    # Save the image of the page in system
    page.save(filename, "JPEG")
    image_file_list.append(filename)

#Jpeg to text via OCR
text_file = "../data/amwins/txt/contract_demo.txt"
with open(text_file, "a") as output_file:
    # Open the file in append mode
    for image_file in image_file_list:
        # Recognize the text as string in image using pytesserct
        text = str(((pytesseract.image_to_string(Image.open(image_file)))))
 
        # The recognized text is stored in variable text
        # Any string processing may be applied on text
        text = text.replace("-\n", "")
 
                # Finally, write the processed text to the file.
        output_file.write(text)

# Sample of final result

# GPT Embedding Search
import os
from langchain.llms.openai import OpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

os.environ['OPENAI_API_KEY']= "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"

loader = TextLoader('../data/amwins/txt/contract_demo.txt', encoding='utf8')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1_000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# use GPT3.5 Embeddings
embeddings = OpenAIEmbeddings()
# load vector embeddings into Vector DB
db = Chroma.from_documents(texts, embeddings)

# set up Chroma as a retriever
retriever = db.as_retriever()
qa = RetrievalQA.from_chain_type(llm=OpenAI(), 
                                 chain_type="stuff", 
                                 retriever=retriever)

# now that the retrieval chain is setup we can query it

query = "What is the name of the owner of the project?"
print(qa.run(query).lstrip())

query = "What is the deposit required in this contract?"
print(qa.run(query).lstrip())

query = "What is the contract value?"
print(qa.run(query).lstrip())

print("Finalizando Proceso")