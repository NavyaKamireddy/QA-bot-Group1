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


import logging

import random
import openai
import json


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Graphsignal: configure
import graphsignal
graphsignal.configure(api_key='GRAPHSIGNAL_API_KEY', deployment='YOUR_DEPLOYEMENT_NAME')

#creating the environment
import openai  # Import openai library
os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"

#creating the environment
import openai  # Import openai library
os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"

# Function to extract text from PDF using PyMuPDF
def extract_text_from_pdf_1(pdf_path_1):
    with fitz.open(pdf_path_1) as doc_lecture:
        for page_number in range(doc_lecture.page_count):
            page = doc_lecture[page_number]
            page_text = page.get_text()
            yield page_number + 1, page_text
          
            
def extract_text_from_pdf_2(pdf_path_2):
    with fitz.open(pdf_path_2) as docs_textbook:
        for page_numbers in range(docs_textbook.page_count):
            pages = docs_textbook[page_numbers]
            page_texts = pages.get_text()
            yield page_numbers + 1, page_texts

#extracting metadata from the path_pdf_1 which is NS_All_slides
def extract_metadata_1(text, patterns_1):
    with graphsignal.start_trace('extract_metadata_1') as span:
        metadata = {}
        for key, pattern in patterns_1.items():
            match = re.search(pattern, text)
            metadata[key] = match.group(1) if match else "Unknown"
            span.set_data(key, metadata[key])
            span.set_data(f'{key}_pattern', pattern)

        return metadata
#extracting metadata from the path_pdf_1 which is NS_Textbook
def extract_metadata_2(text, patterns_2):
    with graphsignal.start_trace('extract_metadata_2') as span:
        metadata = {}
        for key, pattern in patterns_2.items():
            match = re.search(pattern, text)
            metadata[key] = match.group(1) if match else "Unknown"
            span.set_data(key, metadata[key])
            span.set_data(f'{key}_pattern', pattern)

        return metadata



#given input generating question
def generate_questions(text):
    sentences = text.split('.')
    questions = [f"   {sentence}?" for sentence in sentences if sentence.strip()]
    return questions
#Function to generate an answer from ChatGPT
def generate_chatgpt_answer(question):
    openai.api_key = "OPENAI_API_KEY"
    prompt = f"Question: {question}\nAnswer:"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=100,
        n=1,
    )
    chatgpt_answer = response.choices[0].text.strip()
    return chatgpt_answer

pdf_path_1 = 'LOCAL_PATH.pdf'
pdf_path_2 = "LOCAL_PATH.pdf"
try:
    start_time = time.time()

    # Extract text from the first PDF using PyMuPDF
    extracted_text_1 = extract_text_from_pdf_1(pdf_path_1)
    # Combine all text from different pages into a single string
    combined_text_1 = ' '.join(page_text for _, page_text in extracted_text_1)

    # Split text into smaller chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=3000,
        chunk_overlap=100,
        length_function=len,
    )
    texts_1 = text_splitter.split_text(combined_text_1)

    embeddings = OpenAIEmbeddings()
    docsearch_1 = FAISS.from_texts(texts_1, embeddings)

    openai_api_key = os.environ["OPENAI_API_KEY"]
    os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"
    llm_1 = OpenAI(temperature=0.7, openai_api_key=openai_api_key)
    
    chain_1 = load_qa_chain(llm_1, chain_type="stuff")

    user_question = input("Enter your Question: ")

    doc_lecture = docsearch_1.similarity_search(user_question)
    with graphsignal.start_trace('run_conversation') as span:
        span.set_data('user_question', user_question)
        span.set_data('doc_lecture', doc_lecture)


    answer_from_document = chain_1.run(input_documents=doc_lecture, question=user_question)
    
    if answer_from_document.strip():
        # Example: Extracting Lecture, Page Number, 
        patterns_1 = {
            'lecture': r'Lecture (\d+)',
            'page_number': r'Page (\d+)',
           
        }
        with graphsignal.start_trace('extract_metadata_1') as span:
            metadata_1 = extract_metadata_1(combined_text_1, patterns_1)
            span.set_data('metadata_1', metadata_1)

        print(f"Metadata from NS_All_Slides : {metadata_1}")
        print(f"Question: {user_question}")
        print(f"Document Answer from Slides  : {answer_from_document}")
        print(f"PDF: {pdf_path_1}")
        answer_source = "Lecture1-slide, pgno:6"
        print("-" * 50)

    # Extract text from the second PDF using PyMuPDF
    extracted_text_2 = extract_text_from_pdf_2(pdf_path_2)
    # Combine all text from different pages into a single string
    combined_text_2 = ' '.join(page_texts for _, page_texts in extracted_text_2)
    # Split text into smaller chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=3000,
        chunk_overlap=100,
        length_function=len,
    )
    texts_2 = text_splitter.split_text(combined_text_2)
    embeddings = OpenAIEmbeddings()
    docsearch_2 = FAISS.from_texts(texts_2, embeddings)
    openai_api_key = os.environ["OPENAI_API_KEY"]
    os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"
    llm_2 = OpenAI(temperature=0.7, openai_api_key=openai_api_key)
    
    chain_2 = load_qa_chain(llm_2, chain_type="stuff")



    docs_textbook = docsearch_2.similarity_search(user_question)

    answer_from_textbook = chain_2.run(input_documents=docs_textbook, question=user_question)
    with graphsignal.start_trace('run_conversation') as span:
        span.set_data('user_question', user_question)
        span.set_data('docs_textbook', docs_textbook)


    if answer_from_textbook.strip():
        # Example: Page Number, and Chapter Number
        patterns_2 = {
            
            'page_number': r'Page (\d+)',
            'chapter_number': r'Chapter (\d+)'
        }
        with graphsignal.start_trace('extract_metadata_2') as span:
            metadata_2 = extract_metadata_2(combined_text_2, patterns_2)
            span.set_data('metadata_2', metadata_2)

        print(f"Metadata from : {metadata_2}")
        print(f"Question: {user_question}")
        print(f"Document Answer from Networksecuritytextbook Metadata : {answer_from_textbook}")
        print(f"PDF: {pdf_path_2}")
        answer_source = "Network Security Textbook"
        print("-" * 50)
        
    openai_api_key = os.environ["OPENAI_API_KEY"]
    os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"
    llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)
    
    # Display ChatGPT answer even if a document answer is found
    chatgpt_answer = generate_chatgpt_answer(user_question)
    print(f"ChatGPT Answer: {chatgpt_answer}")
    answer_source = "ChatGPT Answer"
    print(f"Answer source: {answer_source}")
    print("-" * 50)

except Exception as e:
    print(f"Error processing {pdf_path_1} or {pdf_path_2}: {str(e)}")
