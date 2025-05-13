from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, WebBaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders.parsers import RapidOCRBlobParser
from langchain_core.runnables import RunnableBranch, RunnableSequence, RunnablePassthrough

load_dotenv()

gemini = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.3
)

parser = StrOutputParser()

summarise_prompt = PromptTemplate(
    template='Summarise the following text.\n {text}',
    input_variables=['text'],
)

faq_prompt = PromptTemplate(
    template='Generate some frequently asked questions for the topic.\n {topic}',
    input_variables=['topic']
)

def file_loader(file_type, file):
    content = ''

    if file_type == 'text':
        loader = TextLoader(file)
    
    elif file_type == 'pdf':
        loader = PyMuPDFLoader(file,
                               mode="page",
                               images_inner_format="text",
                               images_parser=RapidOCRBlobParser()
                               )
        
    elif file_type == 'link':
        loader = WebBaseLoader(file)

    elif file_type == 'pasted':
        return file
    
    for page in loader.lazy_load():
        content = content + page.page_content

    return content


file_type = str(input("Enter file type (Text, PDF, Web URL, Pasted): "))

file = input("Upload/Paste the file or its URL: ")

text = file_loader(file_type, file)

query = str(input("Summarise or FAQs: "))


summariser_chain = RunnableSequence(summarise_prompt, gemini, parser)
faq_chain = RunnableSequence(faq_prompt, gemini, parser)

choice = RunnablePassthrough(query)

branch_chain = RunnableBranch(
    (lambda x: x == 'query', summariser_chain),
    (lambda x: x == 'faq', faq_chain),
    RunnablePassthrough()
)

final_chain = RunnableSequence(choice, branch_chain)

result = final_chain.invoke(text)

print(result)

