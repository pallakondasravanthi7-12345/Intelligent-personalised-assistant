# Intelligent Personalised Assistant

## Run on Windows

1. Open this folder in VS Code.
2. Open Terminal.
3. Run:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

The application will open at http://localhost:8501

## How to use

1. Enter your name, course and year.
2. Upload a text-based PDF.
3. Click "Add Materials".
4. Ask a question about the uploaded material.
5. The assistant searches the ChromaDB knowledge base and shows relevant information.

## Optional Gemini

The current version does not require a Gemini API key. If you later want generated natural-language answers, add your key to `.env` and the answer-generation layer can be added.
