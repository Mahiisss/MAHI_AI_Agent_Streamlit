# 📄 MAHI AI Agent — Streamlit Version  

## Overview  
This is a **Document Analyzer AI Agent** built using **Streamlit**.  
It allows users to upload PDFs, extract key fields (Name, Semester, GPA/CGPA, Email, Phone, GitHub), and ask natural language questions with answers derived using **regex + semantic search**.  

---

## 🚀 Features  
- Upload PDF files (up to 200MB).  
- Extract structured fields:  
  - Name  
  - Semester  
  - CGPA / GPA  
  - Email  
  - Phone  
  - GitHub  
- Ask natural language questions → exact field if found, or semantic context if not.  
- Summarize the document with **CGPA highlighted first**.  
- Pure **Streamlit-based app** → no separate backend required.  

---

## 🛠️ Tech Stack  
- **Streamlit** — frontend + backend (single app).  
- **pdfplumber** — extract text from PDFs.  
- **sentence-transformers (MiniLM-L6-v2)** — semantic embeddings.  
- **faiss-cpu** — vector search.  
- **Regex + NLP preprocessing** for field extraction.  

---

## 📦 Requirements  
All dependencies are in `requirements.txt`:  

## 🚀 Live Demo
You can try the hosted app here: [Document Analyzer Agent](https://github.com/Mahiisss/MAHI_AI_Agent_Streamlit/edit/main/README.md)



