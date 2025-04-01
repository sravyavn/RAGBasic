from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq

load_dotenv()

#Step 1: Read Text from PDF File

text = ""
pdf_reader = PdfReader("incorrect_facts.pdf")
for page in pdf_reader.pages:
    text += page.extract_text() + "\n"

# Step 2: Split text into chunks

text_splitter = CharacterTextSplitter(separator="\n", chunk_size=200, chunk_overlap=20, length_function=len)
chunks = text_splitter.split_text(text)
# for i in range(len(chunks)):
#     print(f"Chunk {i}: {chunks[i]}")

# Step 3: Initialize Pinecone and create index

pc = Pinecone()

index_name = "rag-demo"

# pc.create_index(
#         name=index_name,
#         dimension=384,  # My model embedding dimension
#         metric="cosine",
#         spec=ServerlessSpec(cloud="aws", region="us-east-1")
#     )
index = pc.Index(index_name)

#Select embedding model

embedding_model = SentenceTransformer("BAAI/bge-small-en")

#Embed and Store in Vector DB and comment once the database is created in pinecone
for i, chunk in enumerate(chunks):
    chunk_embedding=embedding_model.encode(chunk, normalize_embeddings=True)
    index.upsert([(str(i+1),chunk_embedding.tolist(),{"chunk":chunk})])
    
query = "How do Birds Migrate"
question_embedding = embedding_model.encode(query, normalize_embeddings=True)  # Fix applied

#Retrive top K results
result=index.query(vector=question_embedding.tolist(),top_k=3,include_metadata=True)


augmented_text="\n\n".join([match.metadata["chunk"] for match in result.matches])
# print(augmented_text)

prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="You are a helpful assistant. Use the context provided to answer the question accurately. Only use this context, not your knowledge.\n\n"
                "Context:{context}"
                "Question:{question}"
                
    )
#User query
llm=ChatGroq(temperature=0,model="llama3-70b-8192")

chain = prompt | llm 

response = chain.invoke({"context":augmented_text, "question": query})


print(response.content)