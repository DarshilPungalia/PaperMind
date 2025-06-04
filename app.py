from flask import Flask, request, render_template
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableSequence, RunnablePassthrough
from dotenv import load_dotenv
from file_handler import File
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO, 
                    filename='app.log', 
                    filemode='w', 
                    format="%(asctime)s-%(name)s-%(levelname)s-%(message)s"
                    )

logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

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
    covering all the sources. If sources are different topic(s) summarise then in a concise way and cite the name/source of the topic just above its summary.\n {text}''',
    input_variables=['text'],
)

faq_prompt = PromptTemplate(
    template='''Generate some frequently asked questions for the following topic(s).If the topic(s) in the sources are completely unrelated create different questionnaires
    for them and cite the source/topic before listing its questions.\n {text}''',
    input_variables=['text']
)

guide_prompt = PromptTemplate(
    template='''Create a study guide which is a condensed and focused resource, include key points, definitions, formulas and quick questions for the given topic(s).
    If the text contains unrelated topic(s) then also make a single guide only trying to cover all the sources by cutting on content for each source that ain't of utmost
    importance.\n {text}''',
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
    codes do not have a timeline.\n{text}''',
    input_variables=['text']
)

map_prompt = PromptTemplate(
    template='''Create a mind map for the following topic. A mind map is a heiarchial tree containing topic, its sub-topics which can repeat for N times. 
    The root is the name of the Topic or file, then comes the major topics discussed in this which then has sub-topics for those. For this division consider only the
    topics of utmost importance. Each node only mentions the topic name nothing else and if any essential info about that is to be expressed it is done in that topics 
    sub-tree. For Code break it down using class structures or independent functions.For source(s) that are unrelated created different mind maps but a shallower tree.
    For code if related try to segregate it like frontend-backend or according to the code files.\n{text}''',
    input_variables=['text']
)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = ''

    if request.method == 'POST':
        logger.info('Fetching Query')
        query = request.form.get('query', '').lower().strip()
        text = []
        
        input_groups = set()
        logger.info('Fetching Keys')
        for key in request.form.keys():
            if key.startswith('file_type_'):
                group_id = key.split('_')[2]
                input_groups.add(group_id)
        logger.info('Fetching Files')
        for group_id in sorted(input_groups):
            file_type = request.form.get(f'file_type_{group_id}')
            logger.info(f'Fetched file_type_{group_id}')
            if file_type:
                try:
                    logger.info(f'Sending file_type_{group_id} to loader')
                    content = files.file_loader(file_type, request, group_id)
                    text.extend(content)
                except Exception as e:
                    logger.exception(f"Error processing input {group_id}")

        if len(text)==0:
            logger.error()
            raise ValueError("No content available for processing")

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

        logger.info('Chain Invoked')
        result = branch_chain.invoke(input_data)

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
