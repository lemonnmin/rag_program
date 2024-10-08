from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain.chains import create_extraction_chain
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 定义提取信息的提示模板
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a top-tier algorithm for extracting information from text. "
            "Only extract information that is relevant to the provided text. "
            "If no information is relevant, use the schema and output "
            "an empty list where appropriate."
        ),
        ("user",
            "I need to extract information from "
            "the following text: ```\n{text}\n```\n",
        ),
    ]
)

# 定义JSON架构
schema = {
    "type": "object",
    "title": "Recipe Information Extractor",
    "$schema": "http://json-schema.org/draft-07/schema#",
    "required": ["recipes"],
    "properties": {
        "recipes": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "ingredients"],
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name of the recipe."
                    },
                    "ingredients": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["name", "amount", "unit"],
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "The name of the ingredient."
                                },
                                "unit": {
                                    "type": "string",
                                    "description": "The unit of the amount of the ingredient."
                                },
                                "amount": {
                                    "type": "number",
                                    "description": "The numeric amount of the ingredient."
                                }
                            }
                        }
                    }
                }
            }
        }
    },
    "description": "Schema for extracting recipe information from text."
}

# 从PDF加载文档
loader = PyMuPDFLoader("./recipe.pdf")
docs = loader.load()

def split_docs(documents, chunk_size=int(128_000 * 0.8), chunk_overlap=20):
    # 初始化文本分割器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # 将文档分割成多个块
    chunks = text_splitter.split_documents(documents=documents)
    
    return chunks

# 分割文档
documents = split_docs(docs)

# 初始化LLM和提取链
llm = OllamaFunctions(model="llama3", temperature=0)
chain = prompt | create_extraction_chain(schema, llm)

# 准备响应列表
responses = []

# 遍历每个文档块并调用提取链
for document in documents:
    input_data = {
        "text": document.page_content,  # 确保正确访问文本内容
        "json_schema": schema,  
        "instruction": (
            "Each recipe has a name and a list of ingredients. "
            "Ingredients should have a name, a numeric amount, and a unit of measure."
        )
    }
    response = chain.invoke(input_data)
    responses.append(response)

# 打印格式化的JSON响应
for response in responses:
    result = response.get('text', {})  # 使用.get()避免KeyError
    print(json.dumps(result, indent=4))
