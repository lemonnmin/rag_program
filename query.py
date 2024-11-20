import os
import logging
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from get_vector_db import get_vector_db
from langchain_core.runnables import RunnableMap
import logging
from langchain_community.llms import Ollama
from langchain.callbacks.base import BaseCallbackHandler
from flask import stream_with_context, Response
from flask import jsonify

LLM_MODEL = os.getenv('LLM_MODEL', 'llama3')

class StreamingCallbackHandler(BaseCallbackHandler):
    def __init__(self, response_stream):
        self.partial_output = ""
        self.response_stream = response_stream

    def on_llm_new_token(self, token: str, **kwargs: any) -> None:
        print(token)
        self.partial_output += token
        if self.response_stream:
            self.response_stream.write(token)
            self.response_stream.flush()

def get_prompt():

    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""Your task is to generate three similar questions based on user questions, provide Chinese questions, and prohibit providing English versions.
        原始问题：{question}""",
    )
     
    relate ="""
            Please generate three related questions for the query provided to the user, which belong to different perspectives. Answer in Chinese, provide questions without explanation, and present answers in JSON format
            query: {question}
            Return format: 
               [[
                {{ question:"..." }},
                {{ question:"..." }} ,
                {{ question:"..." }} 
               ]]
        """
    
    template = """
    #Role: You are a document assistant, searching through document content based on questions as the basis for answering
    ###Skills:
    1. Clear thinking and accurate answering of questions based on the document
    2. Can provide guidance on specific actions and details
    ##Rules
    2. When unable to provide an answer, simply apologize
    3. The final answer must be in Chinese and the sentences must be fluent
    ##document content
    {context}
    ##question:
    {question}
    ##Answer all in Chinese, point by point：
    直接给出答案，全部使用中文：
    1.
    2.
    3.
    """

    prompt = ChatPromptTemplate.from_template(template)
    relatePrompt = ChatPromptTemplate.from_template(relate)
    return QUERY_PROMPT, prompt , relatePrompt

# Function to combine input_context with retrieved context
def combine_contexts(input_context, retrieved_context):
    return f"{input_context}"

def print_context(x):
    print("Combined context:", x["context"])
    return x

def query(input_context, input, response_stream , isRelate):
    if input:
        handler = StreamingCallbackHandler(response_stream)
        llm = ChatOllama(model=LLM_MODEL, callbacks=[handler], disable_streaming='false')
        QUERY_PROMPT, prompt, relatePrompt = get_prompt()

        # 文档模式

       # 获取向量数据库对象
        db = get_vector_db()

        # 将数据库对象转换为检索器对象
        retriever = db.as_retriever()

        # 使用检索器获取相关文档
        relevant_docs = retriever.get_relevant_documents(input)

                # Combine the retrieved documents with the input context


        chain = RunnableMap({
            "context":RunnablePassthrough(),
            "question": RunnablePassthrough()
        }) | prompt | llm | StrOutputParser()

        # 日常模式

        # chaindate = RunnableMap({
        #     "context":RunnablePassthrough(),
        #     "question": RunnablePassthrough()
        # }) | prompt | llm | StrOutputParser()


        # 相关问题
        chainRel = RunnableMap({
            "question": RunnablePassthrough()
        }) | relatePrompt | llm | StrOutputParser()

        try:
            print("开始思考")
            print(isRelate)
            if(isRelate == "false") :
                print("回答内容")
                # related_docs = retriever.retrieve(input_context)
                       # Get the vector database instance
                # "context": combined_context,
                # 文档模式
                response1 = chain.invoke({"context":relevant_docs,"question": input})
                # 日常模式
                # print(input_context)
                # response1 = chaindate.invoke({"context": " ","question": input})
                for token in response1:
                 yield token
            else:
                print("相关问题")
                response2 = chainRel.invoke({"question": input})
                for token in response2:
                 yield token
            
        except Exception as e:
            logging.error(f"Query processing error: {e}")
            yield "An error occurred while processing your request."

    return jsonify({"error": "Invalid input"}), 400    

