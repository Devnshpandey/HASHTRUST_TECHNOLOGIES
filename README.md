# Productivity Assistant and RAG Policy Reviewer

Two Python backend applications for productivity assistance and policy document querying.

## 📁 Project Structure

```
.
├── Productivity_assistant.py  # Conversational productivity assistant
├── RAG_Task.py               # Policy document query system
└── policy_documents/         # Directory for PDF policy files
```

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- Required packages: `PyPDF2`, `sentence-transformers`, `scikit-learn`, `numpy`

### Installation

1. Clone the repository and navigate to the project directory
2. Install dependencies:
   ```bash
   pip install PyPDF2 sentence-transformers scikit-learn numpy
   ```
3. Create the policy documents directory:
   ```bash
   mkdir policy_documents
   ```
4. Add your PDF policy files to the `policy_documents` directory

## 📋 Productivity Assistant

A conversational assistant that maintains context across multiple exchanges to help with productivity tasks.

### Features

- Maintains conversation history with configurable length
- Rule-based response system (easily extendable)
- Save/load conversation history to/from JSON files
- Clean architecture with abstract LLM interface

### Usage

```bash
python Productivity_assistant.py
```

Example interaction:
```
Productivity Assistant initialized. Type 'quit' to exit.
How can I help with your productivity today?

You: Hello
Assistant: Hello! How can I assist you with your productivity today?

You: I need help with scheduling
Assistant: I can help with scheduling. Would you like to check your availability or add a new event?
```

### Extending the Assistant

To integrate with a real LLM API, implement the `LLMInterface`:

```python
class OpenAILLM(LLMInterface):
    def generate_response(self, context: str, user_input: str) -> str:
        # Add OpenAI API integration here
        pass
```

## 🔍 RAG Policy Reviewer

A Retrieval-Augmented Generation system that answers questions based on company policy documents using semantic search.

### Features

- Processes PDF documents from the `policy_documents` directory
- Text extraction with page number preservation
- Semantic search using sentence transformers
- Configurable chunk size and similarity thresholds
- Source attribution in responses

### Usage

```bash
python RAG_Task.py
```

Example interaction:
```
Policy Reviewer initialized with 3 policy documents.
Type 'quit' to exit or ask a question about company policies.

Your question: What is the vacation policy?

Regarding your question 'What is the vacation policy?', here's what I found in company policies:

From employee_handbook.pdf:
- Page 15: Employees accrue vacation time at a rate of 1.5 days per month...
- Page 16: Vacation requests must be submitted at least two weeks in advance...

Please consult the full policy documents or HR for complete information.
```

### Configuration

Adjust these parameters in the `PolicyReviewer` class:
- `chunk_size`: Size of text chunks (default: 500 characters)
- `overlap`: Overlap between chunks (default: 50 characters)
- `similarity_threshold`: Minimum similarity score for results (default: 0.5)

## 🛠️ Technical Details

### Code Quality

Both applications follow Python best practices:
- Comprehensive docstrings with parameter and return descriptions
- Complete type hints throughout
- Proper error handling with specific exception types
- Extensive logging configuration
- Abstract interfaces for extensibility
- Separation of concerns with dedicated classes

### Dependencies

- `PyPDF2`: PDF text extraction
- `sentence-transformers`: Text embeddings and semantic search
- `scikit-learn`: Cosine similarity calculations
- `numpy`: Numerical operations

## 📊 Logging

Both applications use Python's logging module with configurable levels:
- **INFO**: General operation information
- **DEBUG**: Detailed processing information
- **WARNING**: Warning conditions
- **ERROR**: Error conditions

## 🐛 Troubleshooting

### Common Issues

1. **No PDF files found**: Ensure PDFs are placed in the `policy_documents` directory
2. **Module not found errors**: Install all required dependencies
3. **PDF extraction errors**: Corrupted PDF files may cause extraction issues

### Debug Mode

Enable debug logging for more detailed information:
```python
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Ensure all code follows the established patterns and guidelines
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the application logs for error details
3. Open an issue in the repository with specific error information
