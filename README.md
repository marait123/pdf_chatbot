# PDF Chatbot

A FastAPI-based chatbot that allows users to upload PDFs and ask questions about their content using Google's Generative AI.

## Prerequisites

- Python 3.12 or higher
- pip (Python package installer)
- A Google API key for Generative AI

## Setup

1. Clone the repository:

```bash
git clone <repository-url>
cd pdf_chatbot
```

2. Create a virtual environment:

```bash
python -m venv venv
```

3. Activate the virtual environment:

On Windows:

```bash
.\venv\Scripts\activate
```

On Unix or MacOS:

```bash
source venv/bin/activate
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

5. Set up environment variables:
   - Copy the `.env.example` file to `.env`:

```bash
cp .env.example .env
```

- Edit the `.env` file and set your configuration values:

```plaintext
# Required
APP_GOOGLE_API_KEY=your_google_api_key_here

# Optional (defaults shown)
APP_LOG_LEVEL=INFO
APP_ML_BUCKET_NAME=genai
APP_ML_MODEL_MODEL_NAME=gemini-1.5-flash
APP_ML_MODEL_TEMPERATURE=0.0
APP_ML_MODEL_MAX_RETRIES=2
APP_ML_TRIMMER_MAX_TOKENS=65000
APP_ML_CHUNK_SIZE=1000
APP_ML_CHUNK_OVERLAP=200
```

6. Create required directories:

```bash
mkdir -p uploads static templates faiss_index
```

## Running the Application

1. Development mode with auto-reload:

```bash
uvicorn main:app --reload --port 8000
```

2. Production mode:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Or using Python directly:

```bash
python main.py
```

The application will be available at `http://localhost:8000`

## Docker Support

1. Build the Docker image:

```bash
docker build -t pdf-chatbot .
```

2. Run the container:

```bash
docker run -p 8000:8000 --env-file .env pdf-chatbot
```

## Usage

1. Open your browser and navigate to `http://localhost:8000`
2. Upload a PDF document using the upload form
3. Start chatting with the bot about the content of your PDF

## Project Structure

```
pdf_chatbot/
├── main.py              # FastAPI application
├── config.py            # Configuration management
├── faiss_db.py         # Vector database handler
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables
├── .env.example        # Example environment file
├── static/             # Static files
│   ├── css/
│   └── js/
├── templates/          # Jinja2 templates
├── uploads/           # PDF uploads directory
└── faiss_index/       # Vector database storage
```

## License

mit-license
