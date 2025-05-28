from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, WebBaseLoader
from langchain_community.document_loaders.notebook import NotebookLoader
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders.parsers import RapidOCRBlobParser
from langchain_core.runnables import RunnableBranch, RunnableSequence, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import validators

load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

gemini = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.5
)

parser = StrOutputParser()

summarise_prompt = PromptTemplate(
    template='Summarise the following text in detail.\n {text}',
    input_variables=['text'],
)

faq_prompt = PromptTemplate(
    template='Generate some frequently asked questions for the topic.\n {text}',
    input_variables=['text']
)

SUPPORTED_FILE_TYPES = {
    "cpp": "cpp",
    "cc": "cpp",
    "cxx": "cpp",
    "hpp": "cpp",
    "h": "cpp",

    "go": "go",

    "java": "java",

    "kt": "kotlin",
    "kts": "kotlin",

    "js": "js",
    "mjs": "js",
    "cjs": "js",

    "ts": "ts",
    "tsx": "ts",

    "php": "php",
    "phtml": "php",
    "php3": "php",
    "php4": "php",

    "proto": "proto",

    "py": "python",
    "pyw": "python",

    "ipynb": "notebook",

    "rst": "rst",

    "rb": "ruby",
    "erb": "ruby",

    "rs": "rust",

    "scala": "scala",
    "sc": "scala",

    "swift": "swift",

    "md": "markdown",
    "markdown": "markdown",

    "tex": "latex",
    "ltx": "latex",
    "latex": "latex",

    "html": "html",
    "htm": "html",

    "sol": "sol",

    "cs": "csharp",

    "cob": "cobol",
    "cbl": "cobol",
    "cpy": "cobol",

    "c": "c",

    "lua": "lua",

    "pl": "perl",
    "pm": "perl",
    "t": "perl",
    "pod": "perl",

    "hs": "haskell",
    "lhs": "haskell",

    "ex": "elixir",
    "exs": "elixir",

    "ps1": "powershell",
    "psm1": "powershell",
    "psd1": "powershell",

    "txt": "text",

    "pdf": "pdf"
}

def validate_file_type(filename, expected_type):
    """Validate if the uploaded file matches the expected type"""
    if not filename:
        return False, "No filename provided"
    
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    
    if not ext:
        return False, "File has no extension"
    
    if ext not in SUPPORTED_FILE_TYPES:
        return False, f"Unsupported file extension: .{ext}"
    
    if expected_type == 'text' and ext != 'txt':
        return False, "Please upload a .txt file for Text File type"
    elif expected_type == 'pdf' and ext != 'pdf':
        return False, "Please upload a .pdf file for PDF File type"
    elif expected_type == 'code' and ext in ['txt', 'pdf']:
        return False, "Please upload a programming file for Code File type"
    
    return True, ext

def file_save(file, expected_type):
    """Save uploaded file with proper validation"""
    if not file or file.filename == '':
        return None, "No file selected"
    
    is_valid, result = validate_file_type(file.filename, expected_type)
    if not is_valid:
        return None, result
    
    ext = result
    secure_name = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_name)
    
    try:
        file.save(file_path)
        return file_path, ext
    except Exception as e:
        return None, f"Failed to save file: {str(e)}"

def validate_url(url):
    """Validate URL format"""
    if not url or not url.strip():
        return False, "URL cannot be empty"
    
    url = url.strip()
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    if not validators.url(url):
        return False, "Invalid URL format"
    
    return True, url

def file_loader(file_type, request):
    """Load and process different file types"""
    content = ''
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)


    if file_type == 'text':
        file_path, _ = file_save(request.files.get('file'), 'text')
        loader = TextLoader(file_path)

    elif file_type == 'pdf':
        file_path, _ = file_save(request.files.get('file'), 'pdf')
        loader = PyMuPDFLoader(file_path, mode="page", images_inner_format="text", images_parser=RapidOCRBlobParser())
    
    elif file_type == 'code':
        file_path, ext = file_save(request.files.get('file'), 'code')
        if ext != 'ipynb':
            loader = TextLoader(file_path)
            try:
                splitter = splitter.from_language(SUPPORTED_FILE_TYPES[ext])
            except:
                pass
        else:
            loader = NotebookLoader(file_path, include_outputs=True, max_output_length=30)

    elif file_type == 'link':
        url = request.form.get('url', '').strip()
        is_valid, validated_url = validate_url(url)
        if not is_valid:
            raise ValueError(validated_url)
        
        loader = WebBaseLoader(validated_url)

    elif file_type == 'pasted':
        content = request.form.get('pasted', '').strip()
        if not content:
            raise ValueError("No text was pasted")

    try:
        for page in loader.lazy_load():
            content += page.page_content + '\n'
    except Exception as e:
        raise ValueError(f"Could not load content from file: {str(e)}")
        
    if not content.strip():
        raise ValueError("The file appears to be empty or contains no readable content")
    
    chunks = splitter.split_text(content)

    return chunks

@app.route('/', methods=['GET', 'POST'])
def index():
    result = ''

    if request.method == 'POST':
        file_type = request.form.get('file_type', '').lower().strip()
        query = request.form.get('query', '').lower().strip()
        text = file_loader(file_type, request)

        if len(text)==0:
            raise ValueError("No content available for processing")

        input_data = {"mode": query, "text": text}

        summariser_chain = RunnableSequence(summarise_prompt, gemini, parser)
        faq_chain = RunnableSequence(faq_prompt, gemini, parser)

        branch_chain = RunnableBranch(
            (lambda x: x["mode"] == "summarise", summariser_chain),
            (lambda x: x["mode"] == "faq", faq_chain),
            RunnablePassthrough()
        )

        result = branch_chain.invoke(input_data)

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
