from flask import Flask, request, render_template, session, jsonify
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableSequence, RunnablePassthrough
from dotenv import load_dotenv
import secrets
from flask_session import Session
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, 
                    filename='app.log', 
                    filemode='w', 
                    format="%(asctime)s-%(name)s-%(levelname)s-%(message)s")

from file_handler import File
from rag_handler import DocumentQA, DocumentQAError, _vector_store


load_dotenv()

logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = secrets.token_hex(16) 
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

files = File(app)

try:
    logger.info('Initializing Gemini')
    gemini = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.5
    )

except Exception:
    logger.critical("Gemini Couldn't be initialized", exc_info=True, stack_info=True)

parser = StrOutputParser()

summarise_prompt = PromptTemplate(
    template='''Summarise the following topic(s).If the source(s) are all for the same topic or contain parts of overarching topic give a detailed summary 
    covering all the sources. If sources are different topic(s) summarise then in a concise way and cite the name/source of the topic just above its summary.For URL's
    try to infer the topic being discussed and don't talk about the site from where the information is being used.\n {text}''',
    input_variables=['text'],
)

faq_prompt = PromptTemplate(
    template='''Generate some frequently asked questions for the following topic(s).If the topic(s) in the sources are completely unrelated create different questionnaires
    for them and cite the source/topic before listing its questions.For URL's try to infer the topic being discussed and don't talk about the site from where 
    the information is being used.\n {text}''',
    input_variables=['text']
)

guide_prompt = PromptTemplate(
    template='''Create a study guide which is a condensed and focused resource, include key points, definitions, formulas and quick questions for the given topic(s).
    If the text contains unrelated topic(s) then also make a single guide only trying to cover all the sources by cutting on content for each source that ain't of utmost
    importance.For URL's try to infer the topic being discussed and don't talk about the site from where the information is being used.\n {text}''',
    input_variables=['text']
)

timeline_prompt = PromptTemplate(
    template='''Create a timeline for the the topic. Timeline basically is the chronological order in how things would have happened. 
    If any time related stuff like date, month or year is mentioned in the text use that as leverage. 
    For texts that inherently do not have a chronological order try to assume one by breaking it down to fundamental concepts, 
    reaching a point that seems like a good starting point and just warn the user that it doesn't have a temporal order. 
    For code just list the hiearchial structure of the classes or order in which the functions are defined.
    If multiple entries are there for a single year/month club them into a heiarchial points for that year instead of mentioning the year again and again.
    Don't stray to far away from the source text.For text(s) that are are unrelated create different timelines but with lesser detail. For code Warn the user at top that
    codes do not have a timeline.For URL's try to infer the topic being discussed and don't talk about the site from where the information is being used.\n{text}''',
    input_variables=['text']
)

map_prompt = PromptTemplate(
    template='''Create a mind map for the following topic. A mind map is a heiarchial tree containing topic, its sub-topics which can repeat for N times. 
    The root is the name of the Topic or file, then comes the major topics discussed in this which then has sub-topics for those. For this division consider only the
    topics of utmost importance. Each node only mentions the topic name nothing else and if any essential info about that is to be expressed it is done in that topics 
    sub-tree. For Code break it down using class structures or independent functions.For source(s) that are unrelated created different mind maps but a shallower tree.
    For URL's try to infer the topic being discussed and 
    don't talk about the site from where the information is being used.\n{text}''',
    input_variables=['text']
)

index_prompt = PromptTemplate(
    template='''From the given content figure what topic is being talked about in this and only output the Topic name, if it clearly seems like a sub-topic output 
    both topic and the corresponding sub-topic like Topic:sub-topic(s).\n{content}''',
    input_variables=['content']
)

@app.route('/upload', methods=['POST'])
def upload():
    upload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info("Uploading & processing files")
    text = []

    if 'raw_text' not in session:
        session['raw_text'] = []
    if 'is_uploaded' not in session:
        session['is_uploaded'] = []
    if 'upload_meta' not in session:
        session['upload_meta'] = {
            'count': 0,
            'files': []
        }
    
    upload_count = session.get('upload_meta', {'count': 0, 'files': []})['count']
    if upload_count > 10:
        return "Can only upload upto 10 sources.", 500

    input_groups = set()
    for key in request.form.keys():
        if key.startswith('file_type_'):
            group_id = key.split('_')[2]
            input_groups.add(group_id)

    currently_uploaded = session.get('is_uploaded', [])
    new_uploads = []

    print(input_groups)
    
    for group_id in sorted(input_groups):
        file_id = f'file_type_{group_id}'
        file_type = request.form.get(file_id)
        
        if file_type and file_id not in currently_uploaded:
            try:
                content, metadata = files.file_loader(file_type, request, group_id, upload_time)
                if content: 
                    text.extend(content)
                    new_uploads.append(file_id)
                    
                    upload_meta = session.get('upload_meta', {'count': 0, 'files': []})
                    upload_meta['count'] = upload_meta['count'] + 1
                    upload_meta['files'].append(metadata)
                    session['upload_meta'] = upload_meta
                    
            except Exception as e:
                logger.error(f'Could not process file for group {group_id}: {e}')
                continue

    if len(text) == 0:
        return "No content to process", 400

    existing_text = session.get('raw_text', [])
    existing_text.extend(text)
    session['raw_text'] = existing_text
    
    is_uploaded = session.get('is_uploaded', [])
    is_uploaded.extend(new_uploads)
    session['is_uploaded'] = is_uploaded

    logger.info(f"Files uploaded & indexed successfully.")

    try:
        logger.info("Adding documents to vectorstore")
        _vector_store.add_documents(text)
    except Exception as e:
        logger.error(f"Failed to add documents to vectorstore: {e}")
        return f"Error adding documents to vectorstore", 500

    return f"Files uploaded & indexed successfully!"

@app.route('/upload-meta', methods=['GET'])
def get_upload_meta():
    if 'upload_meta' not in session:
        return jsonify({'count': 0, 'files': []}), 200
    return jsonify(session['upload_meta']), 200

@app.route('/', methods=['GET', 'POST'])
def index():
    session['is_uploaded'] = []
    result = ''
    if request.method == 'POST':
        logger.info('Fetching Query')
        query = request.form.get('query', '').lower().strip()

        text = session.get('raw_text', [])

        if len(text) == 0:
            logger.error("No uploaded text found — user needs to upload first!")
            return "No files uploaded yet! Please upload files first.", 400

        input_data = {"mode": query, "text": text}

        summariser_chain = RunnableSequence(summarise_prompt, gemini, parser)
        faq_chain = RunnableSequence(faq_prompt, gemini, parser)
        guide_chain = RunnableSequence(guide_prompt, gemini, parser)
        timeline_chain = RunnableSequence(timeline_prompt, gemini, parser)
        map_chain = RunnableSequence(map_prompt, gemini, parser)

        branch_chain = RunnableBranch(
            (lambda x: x["mode"] == "summarise", summariser_chain),
            (lambda x: x["mode"] == "faq", faq_chain),
            (lambda x: x['mode'] == 'guide', guide_chain),
            (lambda x: x['mode'] == 'timeline', timeline_chain),
            (lambda x: x['mode'] == 'map', map_chain),
            RunnablePassthrough()
        )

        logger.info('Branch Chain Invoked')
        result = branch_chain.invoke(input_data)

    return render_template('index.html', result=result)

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if 'chat_history' not in session:
        session['chat_history'] = []
    
    if request.method == 'POST':
        try:
            user_query = request.get_json().get('message', '').strip()
            
            if not user_query:
                return {"error": "Empty query provided"}, 400
            
            if not session.get('raw_text'):
                return {"error": "No documents uploaded. Please upload documents first."}, 400
            
            try:
                doc_qa = DocumentQA(session)
                logger.info("DocumentQA initialized for chat")
            except DocumentQAError as e:
                return {"error": "Failed to initialize chat system"}, 500
            
            try:
                response = doc_qa.invoke(user_query)
                logger.info("Query processed successfully in chat")
                return {"response": response}
            except DocumentQAError as e:
                logger.error(f"Failed to process query in chat: {e}")
                return {"error": "Failed to process your question"}, 500
                
        except Exception as e:
            logger.error(f"Unexpected error in chat: {e}")
            return {"error": "An unexpected error occurred"}, 500
    
    chat_history = session.get('chat_history', [])
    
    return render_template('chat.html', chat_history=chat_history)
    
@app.route('/session-status', methods=['GET'])
def session_status():
    """Get current session status for debugging"""
    return jsonify({
        'raw_text_count': len(session.get('raw_text', [])),
        'uploaded_files': session.get('is_uploaded', []),
        'upload_meta': session.get('upload_meta', {'count': 0, 'files': []})
    })  

@app.route('/reset-session', methods=['POST'])
def reset_session():
    """Reset session data - useful for testing"""
    session.clear()
    logger.info("Session and vector store cleared")
    return "Session reset successfully!"

if __name__ == '__main__':
    app.run(debug=True, use_evalex=False, use_reloader=False)
