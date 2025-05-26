# PaperMind

This is a Flask-based web application that allows users to process different types of text data — including uploaded files, URLs, and pasted content — and either generate **Summaries** or **FAQs** using Google’s Gemini generative AI model via LangChain.

## 📹 Demo

![App Demo](demo.gif)

## Features

- Upload `.txt`, `.pdf` or code files of common programming languages and extract their content
- Input a website URL or paste raw text
- Choose between **"Summarise"** or **"FAQ"** modes
- Leverages Google Gemini (via LangChain) for AI-powered processing
- Clean web interface for easy interaction

## 🛠 Tech Stack

- Python 🐍
- Flask 🌐
- LangChain + Google Gemini ✨
- HTML (with Jinja2 templates)
- dotenv for managing API keys

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/DarshilPungalia/PaperMind
cd your-repo-name
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
