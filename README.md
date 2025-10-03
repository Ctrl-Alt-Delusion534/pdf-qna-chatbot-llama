# PDF Q&A Chatbot (Streamlit + Groq + LangChain)

This is a simple chatbot app that lets you upload a PDF and then ask questions about it.  
The app breaks the PDF into chunks, creates embeddings, and uses Groq’s Llama-3 model to answer your questions based only on the document.  

Think of it as a way to "chat" with your PDFs.

---

## What it can do
- Upload any PDF and ask questions in plain English  
- Gives answers that come directly from the document  
- Keeps a running chat history while you explore  
- Shows how long each response took  
- Built with Groq’s Llama-3.3-70B model for fast, accurate answers  

---

## Getting started

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/pdf-chatbot-groq.git
cd pdf-chatbot-groq
