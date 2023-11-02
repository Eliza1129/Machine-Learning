#!/usr/bin/env python
# coding: utf-8

# # Vectorstores and Embeddings
# 
# Recall the overall workflow for retrieval augmented generation (RAG):

# ![overview.jpeg](attachment:overview.jpeg)



import os
import openai
import sys
import subprocess 
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']


# We just discussed `Document Loading` and `Splitting`.



from langchain.document_loaders import PyPDFLoader

# Load PDF
loaders = [
    # Duplicate documents on purpose - messy data
    PyPDFLoader("docs/SFBU_Catalog/2023Catalog.pdf"),
    # PyPDFLoader("docs/SFBU_Catalog/2023Catalog.pdf"),
    # PyPDFLoader("docs/SFBU_Catalog/2023Catalog.pdf"),
    # PyPDFLoader("docs/SFBU_Catalog/2023Catalog.pdf"),
]
docs = []
for loader in loaders:
    docs.extend(loader.load())



# Split
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 150
)




splits = text_splitter.split_documents(docs)




len(splits)


# ## Embeddings
# 
# Let's take our splits and embed them.

from langchain.embeddings.openai import OpenAIEmbeddings
embedding = OpenAIEmbeddings()



sentence1 = "What academic programs are offered?"
sentence2 = "Are there specific majors or concentrations available?"
sentence3 = "the weather is ugly outside"



embedding1 = embedding.embed_query(sentence1)
embedding2 = embedding.embed_query(sentence2)
embedding3 = embedding.embed_query(sentence3)



import numpy as np



np.dot(embedding1, embedding2)
np.dot(embedding1, embedding3)
np.dot(embedding2, embedding3)


print("Dot Product 1-2:", np.dot(embedding1, embedding2))
print("Dot Product 1-3:", np.dot(embedding1, embedding3))
print("Dot Product 2-3:", np.dot(embedding2, embedding3))
# ## Vectorstores


# ! pip install chromadb



from langchain.vectorstores import Chroma

persist_directory = 'docs/chroma/'

subprocess.run('rm -rf ./docs/chroma', shell=True)


vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
)



print(vectordb._collection.count())


# # ### Similarity Search



question = "is there an email i can ask for help"
docs = vectordb.similarity_search(question,k=3)
len(docs)


docs[0].page_content


# Let's save this so we can use it later!



vectordb.persist()
for doc in docs:
    print("Similar Document Page Content:")
    print(doc.page_content)

# ## Failure modes
# 
# This seems great, and basic similarity search will get you 80% of the way there very easily. 
# 
# But there are some failure modes that can creep up. 
# 
# Here are some edge cases that can arise - we'll fix them in the next class.


question = "what did they say about scholarship?"
docs = vectordb.similarity_search(question,k=5)


# Notice that we're getting duplicate chunks (because of the duplicate `MachineLearning-Lecture01.pdf` in the index).
# 
# Semantic search fetches all similar documents, but does not enforce diversity.
# 
# `docs[0]` and `docs[1]` are indentical.

docs[0]
docs[1]
for doc in docs:
    print("Similar Document Page Content:")
    print(doc.page_content)


# We can see a new failure mode.
# 
# The question below asks a question about the third lecture, but includes results from other lectures as well.



question = "how many majors in SFBU?"



docs = vectordb.similarity_search(question,k=5)



for doc in docs:
    print(doc.metadata)


print(docs[4].page_content)


# Approaches discussed in the next lecture can be used to address both!



