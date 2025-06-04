import os
from flask import Flask
from werkzeug.utils import secure_filename
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, WebBaseLoader
from langchain_community.document_loaders.notebook import NotebookLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.parsers import RapidOCRBlobParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
import validators
import logging

logger = logging.getLogger(__name__)


class File():
    def __init__(self, app):
        self.SUPPORTED_FILE_TYPES = {
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
        if isinstance(app, Flask):
            self.app = app
        else:
            raise TypeError(f"app type {type(app)} is invalid. app must be Flask type") 

    def validate_file_type(self, filename, expected_type):
        """Validate if the uploaded file matches the expected type"""
        logger.info('Validating File Type')
        if not filename:
            return False, "No filename provided"
        
        ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
        
        if not ext:
            logger.error("File has no extension")
            return False, "File has no extension"
        
        if ext not in self.SUPPORTED_FILE_TYPES:
            logger.error(f"Unsupported file extension: .{ext}")
            return False, f"Unsupported file extension: .{ext}"
        
        if expected_type == 'text' and ext != 'txt':
            return False, "Please upload a .txt file for Text File type"
        elif expected_type == 'pdf' and ext != 'pdf':
            return False, "Please upload a .pdf file for PDF File type"
        elif expected_type == 'code' and ext in ['txt', 'pdf']:
            return False, "Please upload a programming file for Code File type"
        
        return True, ext

    def file_save(self, file, expected_type):
        """Save uploaded file with proper validation"""
        if not file or file.filename == '':
            return None, "No file selected"
        
        is_valid, result = self.validate_file_type(file.filename, expected_type)
        if not is_valid:
            return None, result
        
        ext = result
        secure_name = secure_filename(file.filename)
        file_path = os.path.join(self.app.config['UPLOAD_FOLDER'], secure_name)
        
        try:
            logger.info(f"Saving file to: {file_path}")
            file.save(file_path)
            return file_path, ext
        except Exception as e:
            logger.exception(f"Failed to save file: {str(e)}")
            return None, f"Failed to save file: {str(e)}"

    @staticmethod
    def validate_url(url):
        logger.debug(f"Validating URL: {url}")
        if not url or not url.strip():
            return False, "URL cannot be empty"
        
        url = url.strip()
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        if not validators.url(url):
            logger.error("Invalid URL format.")
            return False, "Invalid URL format"
        
        logger.debug(f"URL validated: {url}")
        return True, url

    def file_loader(self, file_type, request, group_id):
        logger.info(f"Loading file type: {file_type} for group: {group_id}")
        content = ''
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)

        try:
            if file_type == 'text':
                file_path, _ = self.file_save(request.files.get(f'file_{group_id}'), 'text')
                loader = TextLoader(file_path)

            elif file_type == 'pdf':
                file_path, _ = self.file_save(request.files.get(f'file_{group_id}'), 'pdf')
                loader = PyMuPDFLoader(file_path, mode="page", images_inner_format="text", images_parser=RapidOCRBlobParser())

            elif file_type == 'code':
                file_path, ext = self.file_save(request.files.get(f'file_{group_id}'), 'code')
                if ext != 'ipynb':
                    loader = TextLoader(file_path)
                    try:
                        splitter = splitter.from_language(self.SUPPORTED_FILE_TYPES[ext])
                        logger.debug(f"Splitter set for language: {self.SUPPORTED_FILE_TYPES[ext]}")
                    except Exception as e:
                        logger.warning(f"No specific splitter for language: {ext}, using default")
                else:
                    loader = NotebookLoader(file_path, include_outputs=True, max_output_length=30)

            elif file_type == 'link':
                url = request.form.get(f'url_{group_id}', '').strip()
                is_valid, validated_url = self.validate_url(url)
                if not is_valid:
                    raise ValueError(validated_url)
                loader = WebBaseLoader(validated_url)

            elif file_type == 'pasted':
                content = request.form.get(f'pasted_{group_id}', '').strip()
                if not content:
                    raise ValueError("No text was pasted")
                logger.debug("Pasted content received.")

            if file_type != 'pasted':
                for page in loader.lazy_load():
                    content += page.page_content + '\n'

        except Exception as e:
            logger.exception(f"Could not load content from file: {str(e)}")
            raise ValueError(f"Could not load content from file: {str(e)}")
            
        if not content.strip():
            logger.error("Empty or unreadable content in uploaded file.")
            raise ValueError("The file appears to be empty or contains no readable content")
        
        chunks = splitter.split_text(content)
        logger.info(f"Split content into {len(chunks)} chunks.")
        return chunks
