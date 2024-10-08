import os

from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from get_vector_db import get_vector_db

LLM_MODEL = os.getenv('LLM_MODEL', 'llama3')

# Function to get the prompt templates for generating alternative questions and answering based on context
def get_prompt():
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""您是一名AI语言模型助理，专门提供关于西北农林大学的准确信息。您的任务是生成给定用户问题的五个不同版本，这将有助于从矢量数据库中检索相关文档。通过对用户问题生成多个视角，您的目标是提高搜索效率，并确保与大学相关的主题得到全面覆盖。请提供这些用换行符分隔的备选问题。
        原始问题：{question}""",
    )

    template = """根据知识库的上下文回答问题:
    {context}
    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    return QUERY_PROMPT, prompt


# Main function to handle the query process
def query(input):
    if input:
        # Initialize the language model with the specified model name
        llm = ChatOllama(model=LLM_MODEL)
        # Get the vector database instance
        db = get_vector_db()
        # Get the prompt templates
        QUERY_PROMPT, prompt = get_prompt()

        # Set up the retriever to generate multiple queries using the language model and the query prompt
        retriever = MultiQueryRetriever.from_llm(
            db.as_retriever(), 
            llm,
            prompt=QUERY_PROMPT
        )

        # Define the processing chain to retrieve context, generate the answer, and parse the output
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        try:
            response = chain.invoke(input)
            return response
        except Exception as e:
            logging.error(f"Query processing error: {e}")
            return "An error occurred while processing your request."

