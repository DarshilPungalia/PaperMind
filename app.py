from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, WebBaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders.parsers import RapidOCRBlobParser
from langchain_core.runnables import RunnableBranch, RunnableSequence, RunnablePassthrough
from dotenv import load_dotenv

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

def file_loader(file_type, request):
    content = ''

    if file_type == 'text':
        file = request.files['file']
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(file_path)
        loader = TextLoader(file_path)

    elif file_type == 'pdf':
        file = request.files['file']
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(file_path)
        loader = PyMuPDFLoader(file_path, mode="page", images_inner_format="text", images_parser=RapidOCRBlobParser())

    elif file_type == 'link':
        url = request.form['url']
        loader = WebBaseLoader(url)

    elif file_type == 'pasted':
        return request.form['pasted']

    for page in loader.lazy_load():
        content += page.page_content

    return content

@app.route('/', methods=['GET', 'POST'])
def index():
    result = ''

    if request.method == 'POST':
        file_type = request.form['file_type'].lower()
        query = request.form['query'].lower()
        text = file_loader(file_type, request)

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
