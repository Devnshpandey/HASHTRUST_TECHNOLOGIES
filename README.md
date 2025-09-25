Of course\! Here is a comprehensive README file for your GitHub repository, explaining both Python scripts.

-----

# Python AI Assistants: Productivity and RAG

This repository contains two command-line AI assistant applications built in Python, showcasing different AI paradigms.

1.  **Productivity Assistant**: A conversational assistant that helps manage tasks and schedule meetings while maintaining conversation history.
2.  **RAG Policy Reviewer**: A Retrieval-Augmented Generation (RAG) system that answers questions based on a collection of PDF documents.

## Table of Contents

  - [Features](https://www.google.com/search?q=%23features)
  - [Project Structure](https://www.google.com/search?q=%23project-structure)
  - [Setup and Installation](https://www.google.com/search?q=%23setup-and-installation)
  - [How to Use](https://www.google.com/search?q=%23how-to-use)
  - [How It Works](https://www.google.com/search?q=%23how-it-works)
  - [License](https://www.google.com/search?q=%23license)

-----

## Features

### Productivity Assistant (`Productivity_assistant.py`)

  - **Conversational Context**: Remembers previous parts of the conversation within a session.
  - **State Persistence**: Saves the complete conversation state (history, tasks, etc.) to a `conversation_history.json` file, allowing you to resume sessions later.
  - **Multi-Step Operations**: Engages in a detailed, multi-question flow to gather all necessary information for scheduling a meeting.
  - **Task Management**: Allows for quick addition of tasks and can provide a summary of all scheduled meetings and tasks.
  - **Modular Design**: Uses an abstract `LLMInterface` class, making it easy to swap the included `SimulatedLLM` with a real language model API (like OpenAI's GPT or Google's Gemini).

### RAG Policy Reviewer (`RAG_Task.py`)

  - **Document Ingestion**: Automatically scans a designated folder (`./policy_documents/`) for PDF files to use as its knowledge base.
  - **PDF Text Extraction**: Uses `PyPDF2` to parse and extract text from policy documents, including page numbers.
  - **Text Chunking**: Intelligently splits document text into smaller, overlapping chunks to ensure semantic context is preserved for embedding.
  - **Vector Embeddings**: Leverages `sentence-transformers` to convert text chunks into high-dimensional vector embeddings.
  - **Semantic Search**: Takes a user's question, embeds it, and uses cosine similarity to find the most relevant text chunks from the policy documents.
  - **Source-Cited Answers**: Generates answers based on the retrieved information and cites the source document and page number for verification.

-----

## Project Structure

```
.
├── policy_documents/         # Directory for RAG policy PDFs
│   └── (add your .pdf files here)
├── Productivity_assistant.py # Script for the conversational assistant
├── RAG_Task.py               # Script for the RAG policy reviewer
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

-----

## Setup and Installation

### 1\. Clone the Repository

Clone this repository to your local machine:

```bash
git clone <your-repository-url>
cd <repository-folder>
```

### 2\. Create a Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies.

```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3\. Install Dependencies

Install all the required Python libraries from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

The `requirements.txt` file should contain:

```
PyPDF2
sentence-transformers
numpy
scikit-learn
torch
torchvision
torchaudio
```

*(Note: `sentence-transformers` uses PyTorch. The last three lines ensure it's installed correctly.)*

### 4\. Prepare Policy Documents (for RAG Task)

  - Create a folder named `policy_documents` in the root of the project directory.
  - Place any company policy PDF files you want to query inside this folder. The application will not work without this folder and at least one PDF file inside it.

-----

## How to Use

### Running the Productivity Assistant

To start the conversational assistant, run the following command in your terminal:

```bash
python Productivity_assistant.py
```

  - The assistant will greet you. You can interact with it by typing commands like:
      - `hello`
      - `schedule a meeting` (this will start the multi-step scheduling process)
      - `add task Finish the report`
      - `what are my tasks`
  - To end the session, type `quit`. Your conversation and tasks will be saved to `conversation_history.json`.

### Running the RAG Policy Reviewer

Ensure you have added PDF files to the `policy_documents` folder before running.

```bash
python RAG_Task.py
```

  - On the first run, the script will process all PDFs, create text chunks, and generate embeddings. This may take a moment depending on the number and size of your documents.
  - Once initialized, it will prompt you to ask a question.
  - Example questions:
      - `What is the company policy on remote work?`
      - `How many vacation days do new employees get?`
  - The assistant will provide an answer based on the document content and cite its sources.
  - To end the session, type `quit`.

-----

## How It Works

### Productivity Assistant

The assistant is built around the `ProductivityAssistant` class, which manages the application's state, including conversation history and a list of tasks. When a user provides input, the assistant first checks if it's in the middle of a special flow (like scheduling a meeting). If not, it checks for simple commands (`add task`, `summarize`). If no command is found, it passes the conversation history to a simulated LLM to generate a generic, helpful response. The entire state is serialized to JSON upon exit.

### RAG Policy Reviewer

This application follows a classic Retrieval-Augmented Generation (RAG) pipeline:

1.  **Ingestion**: The `PolicyProcessor` scans the `policy_documents` directory for PDFs.
2.  **Parsing & Chunking**: It extracts text from each page and breaks it down into smaller, overlapping chunks to maintain context.
3.  **Embedding**: The `EmbeddingManager` uses a pre-trained `SentenceTransformer` model to convert each text chunk into a numerical vector (an embedding). All embeddings are stored in memory.
4.  **Retrieval**: When a user asks a question, it's also converted into an embedding. The system then calculates the **cosine similarity** between the question's embedding and all the chunk embeddings to find the chunks that are most semantically similar to the question.
5.  **Generation**: The text from the top-matching chunks is compiled and formatted into a user-friendly answer, complete with source document and page number references.

-----
