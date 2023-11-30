#imports
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os
import time
import fitz
import re
#creating the environment
import openai  # Import openai library
os.environ["OPENAI_API_KEY"] = "API_KEY"

# Function to extract text from PDF using PyMuPDF
def extract_text_from_pdf(pdf_path_1):
    with fitz.open(pdf_path_1) as doc_lec
   ture:
        for page_number in range(doc_lecture.page_count):
            page = doc_lecture[page_number]
            page_text = page.get_text()
            yield page_number + 1, page_text
          
            
def etracting_text_from_pdf(pdf_path_2):
    with fitz.open(pdf_paths_2) as docs_textbook:
        for page_numbers in range (docs_textbook.page_counts):
            pages = docs_textbook[page_numbers]
            page_texts = pages.get_text()
            yield page_numbers + 1, page_texts


# Function to extract metadata from document text
def extract_metadata(text, pattern):
    match = re.search(pattern,text)
    return f"{pattern} {match.group(1)}" if match else f"Unknown {pattern}"
    # Example regular expressions to extract chapter and lecture information
    lecture_match = re.search(r'Lecture_slides (\d+)', text)
    chapter_match = re.search(r'chapter_textbook (\d+)', text)
    lecture_slides = f"Lecture {lecture_match.group(1)}" if lecture_match else "Unknown Lecture"
    chapter_textbook = f"chapter_textbook {chapter_match.group(1)}" if chapter_match else "Unknown Chapter"
   
    return lecture_slides,  chapter_textbook

def generate_questions(text):
    sentences = text.split('.')
    questions = [f"   {sentence}?" for sentence in sentences if sentence.strip()]
    return questions
#Function to generate an answer from ChatGPT
def generate_chatgpt_answer(question):
    openai.api_key = "API_KEY"
    prompt = f"Question: {question}\nAnswer:"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=100,
        n=1,
    )
    chatgpt_answer = response.choices[0].text.strip()
    return chatgpt_answer

pdf_path_1 = '.pdf'
pdf_path_2 = '.pdf'


try:
    start_time = time.time()

    # Extract text from the first PDF using PyMuPDF
    extracted_text = extract_text_from_pdf(pdf_path_1)
    # Combine all text from different pages into a single string
    combined_text = ' '.join(page_text for _, page_text in extracted_text)

    # Split text into smaller chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=3000,
        chunk_overlap=100,
        length_function=len,
    )
    texts = text_splitter.split_text(combined_text)

    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.from_texts(texts, embeddings)

    openai_api_key = os.environ["OPENAI_API_KEY"]
    os.environ["OPENAI_API_KEY"] = "API_KEY"
    llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)
    
    chain = load_qa_chain(llm, chain_type="stuff")

    user_question = input("Enter your Question: ")

    doc_lecture = docsearch.similarity_search(user_question)

    answer_from_document = chain.run(input_documents=doc_lecture, question=user_question)

    if answer_from_document.strip():
        metadata_1 = extract_metadata(combined_text, r'Lecture_slides (\d+)')
        print(f"Metadata from NS_All_Slides : {metadata_1}")
        print(f"Question: {user_question}")
        print(f"Document Answer from Slides  : {answer_from_document}")
        print(f"PDF: {doc_lecture[0]}" )
        print(f"PDF: {pdf_path_1}")
        answer_source = "Lecture1-slide, pgno:6"
        print("-" * 50)

    # Extract text from the second PDF using PyMuPDF
    extracted_texts = extract_text_from_pdf(pdf_path_2)
    # Combine all text from different pages into a single string
    combined_texts = ' '.join(page_texts for _, page_texts in extracted_texts)
    # Split text into smaller chunks
    texts_textbook = text_splitter.split_text(combined_texts)

    docs_textbook = docsearch.similarity_search(user_question)

    answer_from_textbook = chain.run(input_documents=docs_textbook, question=user_question)

    if answer_from_textbook.strip():
        metadata_2 = extract_metadata(combined_texts, r'chapter_documents (\d+)')
        print(f"Metadata from : {metadata_2}")
        print(f"Question: {user_question}")
        print(f"Document Answer from Networksecuritytextbook Metadata : {answer_from_textbook}")
        print("-" * 50)
        print(f"PDF: {docs_textbook[0]}" )
        print(f"PDF: {pdf_path_2}")
        answer_source = "Network Security Textbook"
        print("-" * 50)
        
    openai_api_key = os.environ["OPENAI_API_KEY"]
    os.environ["OPENAI_API_KEY"] = "API_KEY"
    llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)
    
    # Display ChatGPT answer even if a document answer is found
    chatgpt_answer = generate_chatgpt_answer(user_question)
    print(f"ChatGPT Answer: {chatgpt_answer}")
    answer_source = "ChatGPT Answer"
    print(f"Answer source: {answer_source}")
    print("-" * 50)

except Exception as e:
    print(f"Error processing {pdf_path_1} or {pdf_path_2}: {str(e)}")




   
