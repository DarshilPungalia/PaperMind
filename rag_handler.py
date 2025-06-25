import chromadb
from langchain_chroma import Chroma
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableSequence, RunnableLambda
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.messages import HumanMessage, AIMessage
import logging
from typing import List, Union, Dict, Literal
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from flask.sessions import SessionMixin
import time


logger = logging.getLogger(__name__)

load_dotenv()


class VectorStore:
    def __init__(self, collection_name="user", persist_directory=None):
        self._vectorstore = None
        self._retriever = None
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        self._init_vectorstore()
        
    def _init_vectorstore(self):
        """
        Create or load the shared persistent vector store.
        Call this ONCE at app startup.
        """

        if self._vectorstore is None:
            try:
                embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
                logger.debug("Google Generative AI Embeddings initialized")
            except Exception as e:
                logger.critical(f"Failed to initialize embeddings: {e}")
                raise VectorStoreError(f"Failed to initialize embeddings: {e}")
                
            try:
                self._vectorstore = Chroma(
                    collection_name="user",
                    embedding_function=embeddings,
                    collection_metadata={"hnsw:space": "cosine"},
                )
                logger.info(f"Vectorstore initialized in memory")
            except Exception as e:
                logger.error(f"Failed to create vector store: {e}")
                raise VectorStoreError(f"Failed to create vector store: {e}")
                
        
    def add_documents(self, documents: List[str])->None:
        """
        Add parsed docs to the shared vector store.
        Call this in /upload route.
        """
        try:
            if not documents:
                raise ValueError("No documents provided to add.")
            if self._vectorstore is None:
                raise RuntimeError("Vectorstore not initialized. Call init_vectorstore() first.")
            try:
                start = time.perf_counter()
                self._vectorstore.add_texts(documents)
                end = time.perf_counter()
                logger.info(f"Successfully added {len(documents)} documents to vector store in {end-start} seconds")
            except Exception as e:
                logger.error(f"Failed to add documents to vector store: {e}")
                raise VectorStoreError(f"Failed to add documents: {e}")
        
        except Exception as e:
            raise VectorStoreError(f'Unable to insert documents: {e}')
        

    def get_retriever(self, search_type:Literal["similarity", "mmr", "similarity_score_threshold"]="similarity")->VectorStoreRetriever:
        """
        Get a retriever from the current vectorstore.
        """
        try:
            if self._vectorstore is None:
                raise RuntimeError("Vectorstore not initialized.")
            if self._retriever is None:
                self._retriever = self._vectorstore.as_retriever(search_type=search_type)
                logger.info("Retriever created from vectorstore.")
            return self._retriever
        
        except Exception as e:
            raise VectorStoreError(f'Unable to create Retriever from Vector Store: {e}')

_vector_store = VectorStore(persist_directory=None)

class DocumentQAError(Exception):
    """Base exception for DocumentQA errors"""
    pass


class ChainBuildError(DocumentQAError):
    """Raised when chain building fails"""
    pass


class VectorStoreError(DocumentQAError):
    """Raised when vector store operations fail"""
    pass


class SessionError(DocumentQAError):
    """Raised when session operations fail"""
    pass

class ChatHistory:
    """Manages chat history using Flask session"""
    
    def __init__(self, session: Union[SessionMixin, Dict]):
        try:
            if not isinstance(session, SessionMixin) and not isinstance(session, dict):
                raise TypeError("Expected a Flask SessionMixin object")
            self.session = session
        except Exception as e:
            raise SessionError(f'Failed to get session: {e}')

    def add_message(self, message: str):
        """Add a message to chat history"""
        if not message or not message.strip():
            raise ValueError('Message is empty.')
        try:
            history = self.session.get('chat_history', [])

            next_index = len(history)

            role = 'user' if next_index%2==0 else 'assistant'

            history.append({'role': role, 'content': message})
            logger.debug(f"Added {role} message to chat history. Total messages: {len(history)}")


            self.session['chat_history'] = history

        except Exception as e:
            logger.error(f"Failed to add message to chat history: {e}")
            raise SessionError(f"Failed to update chat history: {e}")

    def get_history(self) -> List:
        """Get the complete chat history"""
        return self.session.get('chat_history', [])
    
    def format_history_for_chain(self) ->List[Union[HumanMessage, AIMessage]]:
        history = self.get_history()
        formatted_history = []

        for message in history:
            role = message.get('role', '')
            content = message.get('content', '')
            
            if role == 'Human':
                formatted_history.append(HumanMessage(content=content))
            elif role == 'LLM':  
                formatted_history.append(AIMessage(content=content))
            else:
                logger.warning(f"Unexpected role in chat history: {role}")
                formatted_history.append(AIMessage(content=content))
        
        return formatted_history
    
    def clear_history(self):
        """Clear the chat history"""
        self.session['chat_history'] = []


class DocumentQA():
    """A complete document-based question-answering system with RAG capabilities"""
    def __init__(self, session: SessionMixin, vector_store: VectorStore=None):        
        try:
            self.chat_history = ChatHistory(session)
            self.vector_store = vector_store or _vector_store
            self.retriever = self.vector_store.get_retriever(search_type='mmr')
            self.chain = self.build_chain()
            
            logger.info("DocumentQA initialized successfully")
            
        except Exception as e:
            logger.critical(f"Failed to initialize DocumentQA: {e}", exc_info=True)
            raise DocumentQAError(f"Failed to initialize DocumentQA: {e}")

    @staticmethod
    def combine_context(documents: List[Document]) -> str:
        """Combine retrieved documents into a single context string"""
        try:
            context = "\n\n".join(document.page_content for document in documents if document.page_content)
            logger.debug(f"Combined context from {len(documents)} documents, total length: {len(context)}")
            return context if context.strip() else "No relevant context found."
        except Exception as e:
            logger.error(f"Failed to combine context: {e}")
            return "Error processing context."
    
    def build_chain(self) -> RunnableSequence:
        """Build the RAG chain"""
        parser = StrOutputParser()

        try: 
            logger.info('Initializing Gemini for QnA.')
            gemini = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=0.5
            )

        except Exception as e:
            logger.critical("Gemini Couldn't be initialized for QnA.", exc_info=True, stack_info=True)
            raise ChainBuildError(f"Gemini Couldn't be initialized for QnA: {e}")

        prompt = PromptTemplate(
            template='''You are PaperMind — an intelligent and helpful assistant that answers user queries using the provided document context.
            Always prioritize the information found in the document. Use it as your **primary source of truth**. Your goal is to extract and present factual, 
            relevant information clearly and concisely.
            If the document is specific (e.g., a company report, research paper, resume, or technical spec), stick strictly to its contents when answering.
            Use the chat history to follow the flow of the conversation and maintain natural continuity without repeating previous answers unless needed.
            If the user asks something the document does **not** address — and you are confident you can answer it correctly — you **may** do so, 
            but clearly **warn** the user that the information is not present in the document and is based on your own knowledge.
            Avoid guessing, hallucinating, or referencing irrelevant information. Do not include general examples unless they directly help clarify the answer.
            Keep your tone clear, professional, and supportive.

            ---
            
            **Instructions:**

            • Use only the document and chat history to answer the query, unless otherwise specified.  
            • If the document includes structured data (tables, lists, metrics), highlight key points clearly.  
            • Ignore irrelevant document sections.  
            • Maintain conversational flow across follow-up questions.  
            • Be concise — focus on clarity, accuracy, and relevance.

            document:{document}

            Use this chat history to have a context of the chat.
            chat_history:{chat_history}
            
            query:{query}''',
            input_variables=['query', 'document', 'chat_history']
        )

        try:
            input_chain = RunnableParallel({
                'document': self.retriever | RunnableLambda(self.combine_context),
                'query': RunnablePassthrough(),
                'chat_history':RunnableLambda(lambda _ :self.chat_history.format_history_for_chain)
            })

            logger.info('Building QnA chain')
            chain = RunnableSequence(input_chain,
                                    prompt,
                                    gemini,
                                    parser)
        except Exception as e:
            logger.error(f'Failed to build QnA chain: {e}')
            raise ChainBuildError(f'Failed to build QnA chain: {e}')
        
        return chain    
    
    def invoke(self, user_query: str) -> str:
        """Process a user query and return the response"""
        if not user_query or not user_query.strip():
            logger.warning("Empty query provided to invoke")
            raise ValueError("Query cannot be empty")
        
        logger.info('Updating session Chat history with user query')
        self.chat_history.add_message(user_query)
        
        try:
            start = time.perf_counter()
            response = self.chain.invoke(user_query)
            end = time.perf_counter()

        except Exception as e:
            logger.error(f"Chain execution failed: {e}")
            raise DocumentQAError(f"Failed to process query: {e}")
        
        if not response:
            return "I apologize, but I couldn't generate a proper response to your query."
        
        logger.info('Updating session Chat history with LLM response')
        self.chat_history.add_message(response)
        
        logger.info(f"Query processed successfully, response length: {len(response)} and response time: {end-start}")
        return response
    
    def get_vectorstore(self) -> Chroma:
        """Get the underlying vector store"""
        logger.info('Fetching Vector Store')
        return _vector_store

        
    def get_retriever(self) -> VectorStoreRetriever:
        """Get the retriever"""
        logger.info('Fetching Retriever')
        return self.retriever

    
    def get_chat_history(self) -> List:
        """Get the chat history"""
        try:
            logger.info('Fetching Chat history from the session')
            history = self.chat_history.get_history()
            return history
        except Exception as e:
            raise SessionError(f'Can not fetch chat history: {e}')
    
    def clear_chat_history(self):
        """Clear the chat history"""
        try:
            logger.info('Clearing Chat history from the session')
            self.chat_history.clear_history()
        except Exception as e:
            raise SessionError(f'Can not clear chat history: {e}')
