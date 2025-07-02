from langchain_core.prompts import PromptTemplate

class Prompts:
    QAPrompt = PromptTemplate(
            template='''Your name is PaperMind. Answer the user query with the help of the document provided.Ignore parts present in context which are irrelevant to the query. For document 
            specific questions which can be personal info, company info, etc stick to it. If the document doesn't contain any info regarding query and if you are capable of CORRECTLY 
            answering it, do so by WARNING the user clearly that the document didn't have the given info but you think the answer is this. But for the most part try to STICK TO THE DOCUMENT only.
            document:{document}

            Use this chat history to have a context of the chat.
            chat_history:{chat_history}
            
            query:{query}''',
            input_variables=['query', 'document', 'chat_history']
        )
    
    SummaryPrompt = PromptTemplate(
        template='''Summarise the following topic(s).If the source(s) are all for the same topic or contain parts of overarching topic give a detailed summary 
        covering all the sources. If sources are different topic(s) summarise then in a concise way and cite the name/source of the topic just above its summary.For URL's
        try to infer the topic being discussed and don't talk about the site from where the information is being used.\n {text}''',
        input_variables=['text'],
    )

    FAQPrompt = PromptTemplate(
        template='''Generate some frequently asked questions for the following topic(s).If the topic(s) in the sources are completely unrelated create different questionnaires
        for them and cite the source/topic before listing its questions.For URL's try to infer the topic being discussed and don't talk about the site from where 
        the information is being used.\n {text}''',
        input_variables=['text']
    )

    GuidePrompt = PromptTemplate(
        template='''Create a study guide which is a condensed and focused resource, include key points, definitions, formulas and quick questions for the given topic(s).
        If the text contains unrelated topic(s) then also make a single guide only trying to cover all the sources by cutting on content for each source that ain't of utmost
        importance.For URL's try to infer the topic being discussed and don't talk about the site from where the information is being used.\n {text}''',
        input_variables=['text']
    )

    TimelinePrompt = PromptTemplate(
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

    MindMapPrompt = PromptTemplate(
        template='''Create a mind map for the following topic. A mind map is a heiarchial tree containing topic, its sub-topics which can repeat for N times. 
        The root is the name of the Topic or file, then comes the major topics discussed in this which then has sub-topics for those. For this division consider only the
        topics of utmost importance. Each node only mentions the topic name nothing else and if any essential info about that is to be expressed it is done in that topics 
        sub-tree. For Code break it down using class structures or independent functions.For source(s) that are unrelated created different mind maps but a shallower tree.
        For URL's try to infer the topic being discussed and 
        don't talk about the site from where the information is being used.\n{text}''',
        input_variables=['text']
    )

    TextNamingPrompt = PromptTemplate(
        template='''From the given content figure what topic is being talked about in this and only output the Topic name, if it clearly seems like a sub-topic output 
        both topic and the corresponding sub-topic like Topic:sub-topic(s).\n{content}''',
        input_variables=['content']
    )