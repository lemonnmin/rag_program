import os
import random
from datetime import datetime
from werkzeug.utils import secure_filename
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOllama
import faiss
import logging
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap
from langchain_core.runnables import RunnablePassthrough

TEMP_FOLDER = os.getenv('TEMP_FOLDER', './_temp')
LLM_MODEL = os.getenv('LLM_MODEL', 'llama3.2')

logging.basicConfig(level=logging.INFO)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'pdf'}

def save_file(file):
    ct = datetime.now()
    ts = ct.timestamp()
    filename = str(ts) + "_" + secure_filename(file.filename)
    file_path = os.path.join(TEMP_FOLDER, filename)
    file.save(file_path)
    return file_path

def load_and_split_data(file_path):
    loader = PyPDFLoader(file_path=file_path)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200, separators=["\n", "。", "！", "？", "，", "、", ""], add_start_index=True)
    chunks = text_splitter.split_documents(data)
    return chunks

def multi_embedding_recall(query, vector_stores, top_k=20):
    combined_results = []
    for vector_store in vector_stores:
        results = vector_store.similarity_search(query, k=top_k)
        combined_results.extend(results)
    return combined_results

def rerank_documents(query, documents):
    llm = ChatOllama(model=LLM_MODEL)
    prompt_template = """
    你是一个智能助手，请根据以下查询重新排序文档，并返回最相关的文档。
    查询: {query}
    文档:
    {documents}
    """
    prompt = PromptTemplate(input_variables=["query", "documents"], template=prompt_template)
    chaindate = RunnableMap({
            "query":RunnablePassthrough(),
            "documents": RunnablePassthrough()
        }) | prompt | llm | StrOutputParser()

    print("开始精排")
    ranked_documents = chaindate.invoke({"query": query,"documents": documents})
    return ranked_documents

def embed(file):
    if file.filename != '' and allowed_file(file.filename):
        print("文件传输处理")
        try:
            file_path = save_file(file)
            chunks = load_and_split_data(file_path)
            
            embedding_model_1 = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            embedding_model_2 = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            
            vector_store_1 = FAISS.from_documents(chunks, embedding_model_1)
            vector_store_2 = FAISS.from_documents(chunks, embedding_model_2)
            
            vector_store_1.save_local("faiss_index1")
            vector_store_2.save_local("faiss_index2")

            
            logging.info(f'成功处理并添加了 {len(chunks)} 个块来自 {file.filename}。')
        except Exception as e:
            logging.error(f'处理文件 {file.filename} 时出错：{e}')
            return False
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
                logging.info(f'已删除临时文件 {file_path}。')
        # 示例查询
        main("爱丽丝的身世")        
        return True
    logging.warning('无效文件或未上传文件。')
    return False

import faiss

def main(query_text):
    # 假设 FAISS 类有一个接受 faiss.Index 对象的构造函数
    embedding_model_1 = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    embedding_model_2 = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
      
    vector_store_1 = FAISS.load_local("faiss_index1", embedding_model_1, allow_dangerous_deserialization=True)
    vector_store_2 = FAISS.load_local("faiss_index2", embedding_model_2, allow_dangerous_deserialization=True)
    
    vector_stores = [vector_store_1, vector_store_2]
    
      # 召回文档
    recalled_docs = multi_embedding_recall(query_text, vector_stores)
    print("召回的文档:")
    for doc in recalled_docs:
        print(doc)
    
    # 精排文档
    final_docs = rerank_documents(query_text, recalled_docs)
    print("最终排序的文档:")
    for doc in final_docs:
        print(doc)

# 确保 multi_embedding_recall 和 rerank_documents 函数已正确定义

