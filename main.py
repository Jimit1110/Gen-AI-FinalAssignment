from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import fitz
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import psycopg2
from langchain import Tokenizer
from typing import List

app = FastAPI()

class PDFInfo(BaseModel):
    pdf_name: str

class Segment(BaseModel):
    text: str

class Embedding(BaseModel):
    embedding: list[float]

class Question(BaseModel):
    question: str
    pdf_name: str

class SearchResult(BaseModel):
    segments: list[str]

class Answer(BaseModel):
    answers: list[str]

class ChromaDB:
    def __init__(self, dbname, user, password, host, port):
        self.connection = psycopg2.connect(
            dbname=dbname,
            user=user,
            password=password,
            host=host,
            port=port
        )
        self.cursor = self.connection.cursor()
        self.create_table()

    def create_table(self):
        create_table_query = """
            CREATE TABLE IF NOT EXISTS embeddings (
                id SERIAL PRIMARY KEY,
                name TEXT,
                embedding TEXT
            )
        """
        self.cursor.execute(create_table_query)
        self.connection.commit()

    def save_embedding(self, name, embedding):
        insert_query = """
            INSERT INTO embeddings (name, embedding) VALUES (%s, %s)
        """
        self.cursor.execute(insert_query, (name, ','.join(str(x) for x in embedding)))
        self.connection.commit()

    def retrieve_embedding(self, name):
        select_query = """
            SELECT embedding FROM embeddings WHERE name = %s
        """
        self.cursor.execute(select_query, (name,))
        result = self.cursor.fetchone()
        if result:
            return [float(x) for x in result[0].split(',')]
        else:
            return None

    def search(self, question):
        search_query = """
            SELECT name FROM embeddings WHERE name LIKE %s
        """
        self.cursor.execute(search_query, ('%' + question + '%',))
        results = self.cursor.fetchall()
        return [result[0] for result in results]

    def close(self):
        self.connection.close()

def segment_text(text: str) -> List[str]:
    #tokenize the text using LangChain
    tokenized_text = tokenizer.tokenize(text)
    #segment the tokenized text
    segmented_text = [segment.text for segment in tokenized_text.segments]
    return segmented_text

def create_embeddings(text: str) -> List[float]:
    embedding = np.random.rand(768)  #placeholder random embedding
    return embedding.tolist()

def search_documents(question: str, chroma_db: ChromaDB) -> List[str]:
    relevant_segments = chroma_db.search(question)
    return relevant_segments

def answer_question(question: Question, chroma_db: ChromaDB, tokenizer, model) -> List[str]:
    tokenized_question = tokenizer(question.question, return_tensors="pt")
    relevant_segments = chroma_db.retrieve_embedding(question.pdf_name)
    answers = []
    for segment in relevant_segments:
        input_ids = tokenizer.encode_plus(tokenized_question["input_ids"], segment, return_tensors="pt")["input_ids"]
        answer_start_scores, answer_end_scores = model(input_ids)
        start_index = torch.argmax(answer_start_scores)
        end_index = torch.argmax(answer_end_scores) + 1
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[0][start_index:end_index]))
        answers.append(answer)
    return answers

tokenizer = Tokenizer()
tokenizer_palm = AutoTokenizer.from_pretrained("google/palm-2-large-uncased")
model_palm = AutoModelForQuestionAnswering.from_pretrained("google/palm-2-large-uncased")

chroma_db = ChromaDB(
    dbname="qadb",
    user="postgres",
    password="admin123",
    host="localhost",
    port="5432"
)

@app.post("/upload/")
async def upload_pdf(pdf_name: str, pdf_file: UploadFile = File(...)):
    contents = await pdf_file.read()
    #process the PDF file
    text = extract_text_from_pdf(contents)
    #segment the text
    segmented_text = segment_text(text)
    #create embeddings for the segmented text
    for i, segment in enumerate(segmented_text):
        embedding = create_embeddings(segment)
        #save embeddings in Chroma Vector Database
        chroma_db.save_embedding(f"{pdf_name}_{i}", embedding)
    return {"message": "PDF uploaded and embeddings created and stored successfully"}

@app.post("/question/")
async def ask_question(question: Question):
    #answer the question
    answers = answer_question(question, chroma_db, tokenizer_palm, model_palm)
    return {"answers": answers}

@app.post("/search/")
async def search(question: str):
    #search for relevant segments in the database
    relevant_segments = search_documents(question, chroma_db)
    return {"segments": relevant_segments}
