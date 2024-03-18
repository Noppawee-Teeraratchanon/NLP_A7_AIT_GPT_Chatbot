from flask import Flask, render_template, request
from langchain.chains import LLMChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationalRetrievalChain
from transformers import AutoTokenizer, pipeline, AutoModelForSeq2SeqLM
from langchain import HuggingFacePipeline
import torch
from langchain import PromptTemplate
from langchain_community.vectorstores import FAISS
import os
from langchain_community.embeddings import HuggingFaceInstructEmbeddings

app = Flask(__name__)

model_name = 'hkunlp/instructor-base'

embedding_model = HuggingFaceInstructEmbeddings(
    model_name = model_name,
    model_kwargs = {"device" : 'cpu'}
)


#calling vector from local
vector_path = './vector-store'
db_file_name = 'AIT'

from langchain.vectorstores import FAISS

vectordb = FAISS.load_local(
    folder_path = os.path.join(vector_path, db_file_name),
    embeddings = embedding_model,
    index_name = 'ait', #default index
)   

retriever = vectordb.as_retriever()


prompt_template = """
    I'm your friendly chatbot named AITBot created by Noppawee Teeraratchanon, here to assist AIT information member to answer the question people may have about AIT.
    Just let me know what you're wondering about, and I'll do my best to guide you through it!
    {context}
    Question: {question}
    Answer:
    """.strip()

PROMPT = PromptTemplate.from_template(
    template = prompt_template
)

model_id = './models/fastchat-t5-3b-v1.0/'

tokenizer = AutoTokenizer.from_pretrained(
    model_id)

tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

pipe = pipeline(
    task="text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens = 256,
    model_kwargs = {
        "temperature" : 0,
        "repetition_penalty": 1.5
    }
)

llm = HuggingFacePipeline(pipeline = pipe)


question_generator = LLMChain(
    llm = llm,
    prompt = CONDENSE_QUESTION_PROMPT,
    verbose = True
)

doc_chain = load_qa_chain(
    llm = llm,
    chain_type = 'stuff',
    prompt = PROMPT,
    verbose = True
)

memory = ConversationBufferWindowMemory(
    k=3, 
    memory_key = "chat_history",
    return_messages = True,
    output_key = 'answer'
)

chain = ConversationalRetrievalChain(
    retriever=retriever,
    question_generator=question_generator,
    combine_docs_chain=doc_chain,
    return_source_documents=True,
    memory=memory,
    verbose=True,
    get_chat_history=lambda h : h
)

@app.route('/', methods=['GET', 'POST'])
def index():
    search_query = None
    answer = None
    source = None

    if request.method == 'POST':
        # Clear the cache
        search_query = None
        answer = None
        source = None

        search_query = request.form['search_query']
        answer = chain({"question":search_query})['answer']
        source = chain({"question":search_query})['source_documents']
        
    return render_template('index.html', search_query=search_query,answer=answer, source=source)

if __name__ == '__main__':
    app.run(debug=True)
