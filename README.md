# PaperMind 🧠📄

PaperMind is a Flask-based web application for document understanding, exploration, and question answering. Users can upload files, paste text, or link content (including YouTube and webpages) to generate **Summaries**, **FAQs**, **Study Guides**, **Timelines**, and **Mind Maps**, or engage in a **context-aware chatbot**. All tasks are powered by Google Gemini through LangChain with RAG (Retrieval-Augmented Generation) support.

## 📹 Demo

![App Demo](demo.gif)

---

## ✨ Features

- ✅ Upload `.txt`, `.pdf`, and a wide range of **code files** (C, C++, Python, JS, Java, etc.)
- 🌐 Add content via **URL** (supports websites and YouTube videos)
- 📋 Paste raw **text** directly
- 🧠 Select from five AI-powered modes:
  - **Summarise**
  - **FAQ generation**
  - **Study Guide**
  - **Timeline**
  - **Mind Map**
- 💬 **Interactive chatbot** over uploaded documents (RAG-based)
- 🧾 Automatic **vector store indexing** for contextual memory
- 🔐 **Session-aware** user state management
- 🛡️ Robust logging and error handling

---

## 🛠 Tech Stack

- Python 🐍
- Flask + Flask-Session 🌐
- LangChain + Google Gemini (Generative + Embedding APIs) ✨
- Chroma Vector Store for RAG 🔍
- HTML + Jinja2 templates 📄
- dotenv for secure API management 🔑

---

## 🧰 File Support

Supports a wide variety of file formats:

| Type     | Formats |
|----------|---------|
| Code     | cpp, py, js, ts, java, swift, rust, php, etc. |
| Document | txt, pdf |
| Notebook | ipynb |
| Raw Text | Paste directly |
| URL      | Web pages, YouTube links |

All content is automatically chunked, validated, and embedded for efficient retrieval and analysis.

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/DarshilPungalia/PaperMind
cd PaperMind
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up Enviroment Variables

```env
GOOGLE_API_KEY=your_google_api_key_here
```

### 4. Run

```bash
python app.py
```

---

### 🙋‍♂️ Note on Frontend

The frontend template used in this project was not created by me. My primary contributions lie in building the backend infrastructure, AI integration, and document-processing logic.

---

### 📜 License

This project is licensed under the MIT License
