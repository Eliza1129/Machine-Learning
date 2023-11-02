from langchain.embeddings.openai import OpenAIEmbeddings
import openai
import os
import sys
import requests
import uuid
from langchain.chains import RetrievalQA
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']

sentence1 = "How can I appy for admission?"
sentence2 = "What is the application deadline?"
sentence3 = "Is there an application fee?"
sentence4 = "Are there specific majors or concentrations available?"
sentence5 = "What scholarships are available?"
sentence6 = "What recreational facilities are available?"
sentence7 = "What are the housing options and costs?"
sentence8 = "When are classes in session and when are breaks?"
sentence9 = "What is the procedure for withdrawing from a course?"
sentence10 = "What are the office hours for different departments?"

embedding = OpenAIEmbeddings(openai_api_key=openai.api_key)
embedding1 = embedding.embed_query(sentence1)
embedding2 = embedding.embed_query(sentence2)
embedding3 = embedding.embed_query(sentence3)
embedding4 = embedding.embed_query(sentence4)
embedding5 = embedding.embed_query(sentence5)
embedding6 = embedding.embed_query(sentence6)
embedding7 = embedding.embed_query(sentence7)
embedding8 = embedding.embed_query(sentence8)
embedding9 = embedding.embed_query(sentence9)
embedding10 = embedding.embed_query(sentence10)


import numpy as np



# numpy.dot(vector_a, vector_b, out = None) 
# returns the dot product of vectors a and b.
np.dot(embedding1, embedding2)
np.dot(embedding1, embedding3)
np.dot(embedding2, embedding3)



#############################################################
# 4. Vectorstores
#############################################################

from langchain.vectorstores import Chroma




persist_directory = 'docs/chroma/'


vectordb = Chroma(
    embedding_function=embedding,
    persist_directory=persist_directory
)


from langchain.memory import ConversationBufferMemory



llm_name = "gpt-3.5-turbo"

from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(model_name=llm_name, temperature=0, openai_api_key=openai.api_key)
llm.predict("Hello world!")

#############################################################
# Step 5.1: Craete a prompt template
#
# - Define a prompt template which has
#   + Some instructions about how to use the following pieces 
#     of context
#   + A placeholder for a context variable.
#     - This is where the documents will go 
#   + A placeholder for the questions variable. 
#############################################################
from langchain.prompts import PromptTemplate

template = """Use the following pieces of \
   context to answer \
   the question at the end. If you don't know \
   the answer, \
   just say that you don't know, don't try \
   to make up an \
   answer. Use three sentences maximum. \
   Keep the answer as \
   concise as possible. Always say \
   "thanks for asking!" \
   at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""

#############################################################
# Step 5.2: Create QA Chain Prompt from prompt template
#############################################################
QA_CHAIN_PROMPT = PromptTemplate(
     input_variables=["context", "question"],
     template=template,)
# Set up the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

# Define your questions
questions = [
    "Is probability a class topic?",
    "What is the application deadline?",
    "Are there specific majors or concentrations available?",
    # Add more questions here
]

# Ask questions and get answers
for question in questions:
    result = qa_chain({"query": question})
    answer = result["result"]
    print(f"Question: {question}")
    print(f"Answer: {answer}")

from langchain.chains import ConversationalRetrievalChain

retriever=vectordb.as_retriever()

def load_db():
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        # Set return messages equal true
        # - Return the chat history as a  list of messages 
        #   as opposed to a single string. 
        # - This is  the simplest type of memory. 
        #   + For a more in-depth look at memory, go back to  
        #     the first class that I taught with Andrew.  
        return_messages=True
    )
    qa = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        memory=memory,
    )
    return qa

import panel as pn
import param

#############################################################
# Step 7.1.2: cbfs class
#############################################################
class cbfs(param.Parameterized):
    chat_history = param.List([])
    answer = param.String("")
    db_query  = param.String("")
    db_response = param.List([])
    
    #########################################################
    # Step 7.1.2.1: init function
    #########################################################
    def __init__(self,  **params):
        super(cbfs, self).__init__( **params)
        self.panels = []
        self.qa = load_db()
    
    #########################################################
    # Step 7.1.2.2: call_load_db function
    #########################################################
    def call_load_db(self, count):
        self.clr_history()
        self.qa = load_db()

    #########################################################
    # Step 7.1.2.3: convchain(self, query) function
    #########################################################
    def convchain(self, query):
        if not query:
            return pn.WidgetBox(pn.Row('User:', 
               pn.pane.Markdown("", width=600)), scroll=True)
        result = self.qa({"question": query, 
                          "chat_history": self.chat_history})
        self.chat_history.extend([(query, result["answer"])])
        return result["answer"]


    #########################################################
    # Step 7.1.2.7: clr_history function
    #########################################################
    def clr_history(self):
        self.chat_history = []
        self.qa = load_db()
        return