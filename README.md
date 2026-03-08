# Document Q&A with RAG (Retrieval-Augmented Generation)

A web application that allows users to upload PDF documents and images, then ask questions about their content. The application uses Retrieval-Augmented Generation (RAG) with embeddings and FAISS for intelligent document retrieval.

## Features

- **Multi-format Document Upload**: Support for both PDF documents and images (PNG, JPG, etc.)
- **OCR for Images**: Automatic text extraction from images using Tesseract OCR
- **RAG-based Q&A**: Uses semantic search to find relevant document chunks and generate accurate answers
- **Chat Interface**: Interactive UI for asking questions about uploaded documents
- **Document Chunking**: Intelligent text chunking with overlap to preserve context

## Tech Stack

### Backend
- **Framework**: Flask (Python web framework)
- **LLM**: Mistral AI (mistral-medium-latest model)
- **Embeddings**: Sentence Transformers (`all-MiniLM-L6-v2`)
- **Vector Search**: FAISS (Facebook AI Similarity Search)
- **PDF Processing**: PyMuPDF (pymupdf)
- **OCR**: Tesseract OCR (pytesseract)

### Frontend
- **HTML/CSS**: Custom styling for modern UI

## Project Structure
```
ABYSALTO_ASSIGNMENT/
├── backend/
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css          # Application styling
│   │   └── js/
│   │       └── main.js             # Frontend JavaScript
│   ├── templates/
│   │   └── index.html              # Main HTML template
│   └── app.py                       # Flask application (main file)
├── .env                            # Environment variables (Mistral API key)
├── .gitignore                      # Git ignore file
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Docker container configuration
├── docker-compose.yml              # Docker Compose setup
└── README.md                       # This file
```

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- Docker and Docker Compose (optional)
- Mistral AI API key
- Tesseract OCR installed

### Manual Installation

1. **Clone the repository** and navigate to the project directory:
```bash
   git clone <repository-url>
   cd ABYSALTO_ASSIGNMENT
```

2. **Install system dependencies** (for OCR):
   - **Ubuntu/Debian**:
```bash
     sudo apt-get update
     sudo apt-get install tesseract-ocr
```
   - **macOS**:
```bash
     brew install tesseract
```
   - **Windows**: Download from [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)

3. **Create a virtual environment**:
```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
```

4. **Install Python dependencies**:
```bash
   pip install -r requirements.txt
```

5. **Configure environment variables**:
   Create a `.env` file in the project root:
```
   MISTRAL_API_KEY=your_api_key_here
   APP_SECRET_KEY=your_secret_key_for_session
```

6. **Run the application**:
```bash
   python backend/app.py
```
   The app will be available at `http://localhost:5000`

### Docker Installation

1. **Configure environment variables**:
   Create a `.env` file in the project root:
```
   MISTRAL_API_KEY=your_api_key_here
   APP_SECRET_KEY=your_secret_key_for_session
```

2. **Build and run with Docker Compose**:
```bash
   docker-compose up --build
```

3. **Access the application**:
   - Open your browser to `http://localhost:5000`

## API Endpoints

### Home Page
- **Route**: `GET /`
- **Description**: Main page displaying uploaded documents and chat interface
- **Response**: HTML page with session data (questions, answers)

### Upload Documents
- **Route**: `POST /upload`
- **Description**: Upload one or multiple PDF or image files
- **Parameters**: 
  - `files` (form data): Multiple files (PDF or images)
- **Returns**: Redirects to home page with uploaded documents loaded
- **Example Request**:
```bash
  curl -X POST http://localhost:5000/upload \
    -F "files=@document.pdf" \
    -F "files=@image.png"
```

### Ask Question
- **Route**: `POST /ask`
- **Description**: Submit a question about the uploaded documents
- **Parameters**: 
  - `question` (form data): The question to ask
- **Returns**: Redirects to home page with generated answer
- **Example Request**:
```bash
  curl -X POST http://localhost:5000/ask \
    -d "question=What is the main topic of the document?"
```

### Reset Session
- **Route**: `POST /reset`
- **Description**: Clear all uploaded documents and conversation history
- **Returns**: Redirects to home page with empty state

## Usage Example

1. **Open the application** at `http://localhost:5000`

2. **Upload documents**:
   - Click "Upload files" button
   - Select one or multiple PDFs or images
   - Click "Upload"

3. **Ask questions**:
   - Type your question in the text input
   - Click "Ask"
   - The AI will search through the documents and provide an answer

4. **View results**:
   - Your questions and answers are displayed in the chat interface
   - File list shows all uploaded documents

5. **Reset**:
   - Click "Reset" button to clear everything and start over

## How It Works

### RAG Pipeline

1. **Document Upload**: Files are processed to extract text (PDFs via PyMuPDF, images via Tesseract)

2. **Chunking**: Text is split into overlapping chunks (800 characters with 150-character overlap) to maintain context

3. **Embeddings**: Each chunk is converted to embeddings using Sentence Transformers model

4. **Indexing**: Embeddings are stored in a FAISS index for fast similarity search

5. **Question Processing**:
   - User question is converted to embeddings
   - FAISS search finds the 3 most relevant chunks
   - Relevant chunks are combined as context

6. **Answer Generation**: Mistral AI generates an answer based on the context and question

### Configuration Details

- **Embedding Model**: `all-MiniLM-L6-v2` (384-dimensional embeddings)
- **Chunk Size**: 800 characters
- **Chunk Overlap**: 150 characters
- **Top-K Retrieval**: 3 chunks
- **LLM Model**: mistral-medium-latest
- **Session Timeout**: 15 minutes

## Error Handling

- Invalid file types are silently skipped during upload
- If no documents are uploaded, the chat interface is hidden
- Empty files are handled gracefully
- If no relevant chunks are found, the LLM will indicate that information isn't available

## Security Notes

- Session data is stored server-side (in memory)
- Session timeout is set to 15 minutes for security
- All user questions are processed by Mistral AI
- No persistent data storage (documents are cleared on reset)

## Dependencies

See `requirements.txt` for complete list. Key packages:
- `flask` - Web framework
- `mistralai` - LLM API client
- `sentence-transformers` - Embeddings
- `faiss-cpu` - Vector search
- `pymupdf` - PDF processing
- `pytesseract` - OCR
- `python-dotenv` - Environment variable management
- `Pillow` - Image processing

## Limitations

- Session data is stored in memory (resets on server restart)
- Maximum document size depends on available memory
- Tesseract OCR accuracy depends on image quality
- FAISS index is not persistent across restarts
- Only text-based content is supported (tables and complex layouts may not extract perfectly)

### API Key Error
Ensure your `.env` file contains the correct Mistral API key:
```
MISTRAL_API_KEY=your_key_here
```

### Large PDF Processing Issues
Large PDFs may take time to process. Consider splitting them into smaller documents.

### NOTE
- This README.md file was writen using ChatGPT.
