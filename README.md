# Advance-RAG


# Advanced Retrieval-Augmented Generation (RAG) Model

This repository contains an advanced implementation of Retrieval-Augmented Generation (RAG) using a combination of state-of-the-art libraries like LangChain, Qdrant, Flashrank, and others. The model is designed to efficiently retrieve, process, and rank information to generate high-quality responses, making it suitable for a wide range of applications, including question-answering systems, chatbots, and more.

## Features

- **RetrievalQA Chain**: Utilizes the `RetrievalQA` chain from LangChain to create a robust question-answering system. This chain retrieves relevant documents from a dataset to provide accurate answers.
  
- **Document Loaders & Text Splitters**: Efficiently processes large text documents using the `UnstructuredMarkdownLoader` and `RecursiveCharacterTextSplitter` to break down content into manageable chunks.
  
- **Qdrant Vector Store**: Employs Qdrant, a high-performance vector search engine, to store and retrieve document embeddings, enabling fast and accurate similarity searches.

- **Embeddings with FastEmbed**: Uses `FastEmbedEmbeddings` for generating high-quality embeddings that capture the semantic meaning of text, improving the relevance of the retrieved information.

- **Reranking with Flashrank**: Enhances the quality of search results by applying `FlashrankRerank`, a reranking model that orders the retrieved documents based on their relevance to the query.
  
- **Custom Prompting & Chatbot Integration**: Incorporates `ChatGroq`, allowing the creation of a conversational AI model with custom prompts tailored to specific use cases.

## Usage

1. **Environment Setup**: The notebook installs all necessary libraries, including `langchain-groq`, `qdrant-client`, `flashrank`, and others. Make sure your environment meets these requirements before running the notebook.

2. **Data Preparation**: Load your markdown or other text-based documents using the provided document loaders. The text splitter will then process these documents into smaller segments, which are easier to manage and retrieve.

3. **Vectorization and Storage**: The processed text is converted into embeddings using `FastEmbedEmbeddings` and stored in Qdrant, where they can be efficiently searched and retrieved.

4. **Querying and Reranking**: When a query is made, the system retrieves relevant documents, which are then reranked using Flashrank to ensure the most relevant documents are surfaced.

5. **Generating Responses**: The `RetrievalQA` chain combines the retrieved and reranked information to generate a coherent and contextually appropriate response.

6. **Chatbot Interaction**: Use `ChatGroq` to create an interactive chatbot interface that can handle complex queries using the RAG model.

## Dependencies

- langchain-groq==0.1.3
- langchain==0.1.17
- qdrant-client==1.9.1
- unstructured[md]==0.13.6
- fastembed==0.2.7
- flashrank==0.2.4

## Example

python
# Example of loading documents and querying the model
loader = UnstructuredMarkdownLoader("your_markdown_file.md")
documents = loader.load()

# Split the documents
splitter = RecursiveCharacterTextSplitter(chunk_size=500)
chunks = splitter.split_documents(documents)

# Create embeddings and store them in Qdrant
embedder = FastEmbedEmbeddings()
vector_store = Qdrant()
vector_store.add_texts(chunks, embedder)

# Query the vector store
query = "why prompt engineering."
results = vector_store.search(query)
reranked_results = FlashrankRerank().rerank(results)

# Generate the final answer
qa_chain = RetrievalQA(...)
answer = qa_chain.run(query, reranked_results)
print(answer)
