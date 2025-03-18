# AI-Projects

📚 Q&A Chatbot (Multiple PDFs)
🔹 Overview
This chatbot extracts and answers questions from multiple PDF files using LLMs and AI-powered retrieval. It enhances multi-document queries by intelligently structuring responses for better accuracy and contextual understanding.

🔹 Features
✅ Processes multiple PDFs for improved answer accuracy
✅ Fast and scalable API for inference
✅ Can integrate with external sources (Tavily API) for enriched answers

🔹 Technologies Used
Llama – for language model inference
Groq API – for AI-powered processing
FastAPI – for backend API
PDFMiner / PyMuPDF – for PDF parsing
VS Code – for development
🔹 Installation & Setup
2️⃣ Set up a Virtual Environment
bash
Copy code
python -m venv venv  
source venv/bin/activate  # Linux/macOS  
venv\Scripts\activate     # Windows  
3️⃣ Install Dependencies
bash
Copy code
pip install -r requirements.txt
4️⃣ Run the Chatbot API
bash
Copy code
uvicorn main:app --reload
🔹 Usage
Upload multiple PDF files.
Ask questions, and the system will retrieve the most relevant answers from the PDFs.
