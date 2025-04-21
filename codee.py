import os
import fitz  # PyMuPDF
import threading
import shutil
from http.server import SimpleHTTPRequestHandler, HTTPServer
from urllib.parse import quote
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document  # Fix: Import Document
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM  # Using DeepSeek R1

# Define paths
DATA_PATH = os.path.join(os.getcwd(), "data", "pdf_files")
CHROMA_PATH = "chroma_db"
PDF_SERVER_PORT = 8000  # Local server port

# Ensure the folder exists
os.makedirs(DATA_PATH, exist_ok=True)

# Start a simple HTTP server for PDFs
def start_pdf_server():
    os.chdir(DATA_PATH)
    handler = SimpleHTTPRequestHandler
    server = HTTPServer(("localhost", PDF_SERVER_PORT), handler)
    print(f"Starting local PDF server at http://localhost:{PDF_SERVER_PORT}/")
    server.serve_forever()

# Run the PDF server in a separate thread
server_thread = threading.Thread(target=start_pdf_server, daemon=True)
server_thread.start()

# Function to extract text from PDFs with page numbers
import pdfplumber

def extract_pdf_text(file_path):
    chunks = []

    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            all_text_parts = []

            # Extract normal text
            text = page.extract_text()
            if text and text.strip():
                all_text_parts.append(text.strip())

            # Extract tables and convert to readable rows
            tables = page.extract_tables()
            for table_index, table in enumerate(tables):
                table_text = f"Table {table_index + 1} on Page {page_num + 1}:\n"
                for row_index, row in enumerate(table):
                    row_text = " | ".join([f"{col.strip()}" if col else "" for col in row])
                    table_text += f"• Row {row_index + 1}: {row_text}\n"
                all_text_parts.append(table_text)

            # If there's any content to keep
            if all_text_parts:
                full_page_text = "\n".join(all_text_parts)
                chunks.append({"page": page_num + 1, "text": full_page_text})
            else:
                print(f"⚠️ No usable text or tables on page {page_num + 1}")

    if not chunks:
        print(f"❌ No text extracted from {file_path}")
    return chunks


# Load PDF documents and extract text with page numbers
def load_pdf_documents():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data path '{DATA_PATH}' does not exist.")
    pdf_files = [f for f in os.listdir(DATA_PATH) if f.endswith(".pdf")]
    if not pdf_files:
        raise ValueError("No PDF files found in the specified directory.")
    
    documents = []
    for pdf_file in pdf_files:
        pdf_path = os.path.join(DATA_PATH, pdf_file)
        extracted_text = extract_pdf_text(pdf_path)
        
        if not extracted_text:
            print(f"⚠️ Skipping empty PDF: {pdf_file}")
            continue
        
        for chunk in extracted_text:
            documents.append({
                "source": pdf_file,  # Store only filename for URL generation
                "page_number": chunk["page"],
                "content": chunk["text"]
            })
    
    print(f"✅ {len(documents)} pages extracted from {len(pdf_files)} PDFs")
    return documents

# Delete & rebuild ChromaDB
shutil.rmtree(CHROMA_PATH, ignore_errors=True)
os.makedirs(CHROMA_PATH, exist_ok=True)

pdf_documents = load_pdf_documents()

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=500,
    length_function=len,
    add_start_index=True
)

chunks = []
for doc in pdf_documents:
    content = doc["content"].strip()
    
    if not content:
        print(f"⚠️ Skipping empty content for {doc['source']} - Page {doc['page_number']}")
        continue

    split_chunks = text_splitter.create_documents([content])  # ✅ Fix: Using create_documents()

    print(f"🔹 Splitting {doc['source']} (Page {doc['page_number']}) -> {len(split_chunks)} Chunks")
    
    for i, chunk in enumerate(split_chunks):
        chunk.metadata = {
            "source": doc["source"],
            "page_number": doc["page_number"],
            "chunk_id": i + 1  # Add chunk ID
        }
        chunks.append(chunk)  # ✅ Fix: Store as Document object

if not chunks:
    raise ValueError("❌ No text chunks were created. Ensure PDFs contain valid text.")

# Initialize embeddings
embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

# Initialize ChromaDB
db = Chroma(
    collection_name="pdf_collection",
    embedding_function=embedding_function,
    persist_directory=CHROMA_PATH
)

# Store chunks in ChromaDB
db.add_documents(chunks)  # ✅ Fix: Directly passing Document objects

print(f"✅ Chunks stored successfully in ChromaDB! Total Chunks: {db._collection.count()}")

# Function to process a query
def process_query(query_text):
    if not query_text:
        return "Error: Query text cannot be empty."

    results = db.similarity_search(query_text, k=5)

    if not results:
        return "No relevant information found. Try rephrasing your query."

    print("\n📖 Retrieved Top 5 Chunks:\n" + "="*50)
    context_texts = []
    chunk_links = []  # 🆕 Collect links for all chunks

    for idx, doc in enumerate(results, 1):
        source = doc.metadata["source"]
        page_number = doc.metadata.get("page_number", 1)
        highlight = quote(doc.page_content[:50])  # You can tune this

        pdf_link = f"http://localhost:5000/viewer?file={quote(source)}&page={page_number}&highlight={highlight}"

        chunk_text = doc.page_content.strip()
        context_texts.append(chunk_text)
        chunk_links.append((idx, pdf_link, source, page_number))  # 🆕 Collect for output

        print(f"\n🔹 Chunk {idx} (Page {page_number}) from {source}:\n{chunk_text[:300]}...\n🔗 Link: {pdf_link}\n" + "-"*50)

    # Generate context
    context_text = "\n\n---\n\n".join(context_texts)

    # Prompt
    PROMPT_TEMPLATE = """
    Answer the question based only on the following context:
    {context}
    ---
    Answer the question based on the above context: {question}
    """
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = OllamaLLM(model="deepseek-r1")
    response_text = model.invoke(prompt)

    # Print links again (optional)
    print("\n📎 Source Links:")
    for idx, link, source, page in chunk_links:
        print(f"🔗 Chunk {idx} → [{source} - Page {page}]({link})")

    return response_text

# Run the PDF server in a separate thread (Non-daemon so it stays alive)
server_thread = threading.Thread(target=start_pdf_server, daemon=False)
server_thread.start()

# Get user query and process it
query_text = input("🔎 Enter your query: ")
response = process_query(query_text)
print("\n🤖 AI Response:\n" + "="*50 + f"\n{response}\n" + "="*50)

# ✅ Keep the script running so the PDF server doesn't exit
print("\n🚀 Server is running at http://localhost:8000/ Press Ctrl+C to stop.")
while True:
    try:
        pass  # Keep script alive
    except KeyboardInterrupt:
        print("\n🛑 Server stopped.")
        break
