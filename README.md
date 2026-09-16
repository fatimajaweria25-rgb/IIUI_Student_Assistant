# 🎓 IIUI Student & Academic Assistant

A beginner-friendly Retrieval-Augmented Generation (RAG) application for students and academics of **Ibadat International University, Islamabad**.

## Technology

- Streamlit — web interface
- FAISS — local vector search
- Sentence Transformers — free local embeddings
- Groq — LLM inference
- `openai/gpt-oss-120b` — current Groq production model used by this project
- PDFs + DOCX links + webpages — knowledge sources
- GitHub + Streamlit Community Cloud — deployment

> The project deliberately separates **index creation** from the live app. The Colab builder creates `index.faiss` and `metadata.json`; the deployed Streamlit app only performs retrieval and generation.

## Folder structure

```text
iiui-student-assistant/
│
├── app.py
├── requirements.txt
├── README.md
├── index_faiss/
│   ├── index.faiss
│   └── metadata.json
└── assets/
    └── iiui_bg.jpg              # optional
```

## 1. Build the knowledge base in Google Colab

Use the one-cell builder supplied with this project.

It will:

1. Download the public Google Drive folder.
2. Read PDF files.
3. Read the Word document.
4. Extract URLs from the Word document.
5. Download readable webpage text from those URLs.
6. Split the material into chunks.
7. Generate local Sentence Transformer embeddings.
8. Build a FAISS cosine-similarity index.
9. Save:
   - `index_faiss/index.faiss`
   - `index_faiss/metadata.json`
10. Zip the index so it can be downloaded and uploaded to GitHub.

The builder does not send your documents to Groq for embedding.

## 2. GitHub

Create a new repository, for example:

`iiui-student-academic-assistant`

Upload:

- `app.py`
- `requirements.txt`
- `README.md`
- `index_faiss/index.faiss`
- `index_faiss/metadata.json`
- optionally `assets/iiui_bg.jpg`

Do **not** upload your Groq API key.

## 3. Create your Groq API key

Create an API key in your Groq account.

The application reads it from:

```text
GROQ_API_KEY
```

## 4. Deploy on Streamlit Community Cloud

1. Sign in to Streamlit Community Cloud with GitHub.
2. Choose **Create app**.
3. Select your GitHub repository.
4. Select branch `main`.
5. Select `app.py`.
6. Open **Advanced settings**.
7. Add this secret:

```toml
GROQ_API_KEY = "paste-your-key-here"
```

8. Deploy.

Never commit `.streamlit/secrets.toml` or your API key to GitHub.

## 5. Admission portal behavior

The app has a deterministic route for admission/application questions and provides the official Ibadat International University online admissions portal:

https://admissions.iiui.edu.pk/index.php

This is intentionally not dependent on the RAG index, so the application link remains available even if a retrieval result is weak.

## 6. Updating the knowledge base

Whenever university documents change:

1. Put the new PDFs/DOCX/links into the Drive folder.
2. Run the Colab builder again.
3. Download the new `index_faiss` folder.
4. Replace the old index files in GitHub.
5. Streamlit Community Cloud will redeploy.

## 7. Important enterprise note

This is an **enterprise-oriented architecture implemented as a low-cost Streamlit MVP**.

For a large university-wide production deployment, later phases should add:

- authentication / role-based access
- document versioning
- admin knowledge-base management
- audit logs
- analytics
- rate limiting
- document permissions
- automated ingestion
- hybrid keyword + vector retrieval
- reranking
- a managed vector database
- monitoring and evaluation
- backup/version rollback

FAISS is excellent for a packaged, read-heavy knowledge base, but a multi-admin, frequently changing, high-concurrency university system will eventually benefit from a managed/vector service architecture.

## 8. Quality and safety behavior

The assistant is instructed to:

- prioritize retrieved university sources
- avoid inventing policies or university facts
- state when information is missing
- show retrieved source information
- distinguish general academic explanations from university-specific information

For high-stakes academic, legal, financial, medical, or administrative decisions, users should verify the information with the relevant university office or official source.

## 9. Local test

Install dependencies:

```bash
pip install -r requirements.txt
```

Then:

```bash
streamlit run app.py
```

For local testing, you can set:

```bash
export GROQ_API_KEY="your-key"
```

On Windows PowerShell:

```powershell
$env:GROQ_API_KEY="your-key"
streamlit run app.py
```
