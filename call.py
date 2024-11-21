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

# 2.2 文本分割优化
# 文本分割对应流程图中的步骤3和4，它是决定文档检索准确度的重要因素，甚至比使用的embedding模型还重要。为什么这样说呢？主要有以下几点原因：

# 文本块是文本检索中最基础的单元。受制于大语言模型输入的最大长度限制，我们很难直接将整个文档输入给大语言模型，因此现有的知识库问答应用都是将检索的文本块作为大语言模型的输入。分割后的文本块需要是一个能表达完整意思的一段文本，如果检索到的文本块内容不完整，将影响LLM回答的准确率。

# 我们在实验中发现，现有的中文embedding模型能力还不是很强，尤其是长文本经过embedding后，向量的区分度不高，所以分割后文本的长度对于后续向量检索的效果有很大影响。
# 综合上述两点， 我们会发现，分割后的文本块过长，embedding区分度低，检索效果差；文本块过短，可能导致信息不完整，最终的回答效果同样不好。

# 文本分割容易出现上下文缺失的问题，影响检索准确度。如图3所示，如果用户的问题是：企业直播自定义鉴权的前提条件是什么？由于文本块2中只包含”前提条件“，而没有”企业直播自定义鉴权“，使用向量召回很难检索到文本块2。


# 图3 文本分割带来的上下文缺失问题
# 基于上述观察和分析，我们对文本分割流程做了以下优化：

# 单文本块多向量表示
# 针对文本块长度的矛盾，我们首先将文本分割为长度300-500 token的文本块，这些文本块被用于向量检索。但是在构建向量时，我们不是利用这个文本块得到一个向量，而是将文本块拆分为一个个句子，得到每个句子的向量表示，如果其中一个句子的向量表示和用户的query向量相似度足够高，整个文本块就会被检索回来。我们将这一方法称为单文本块的多向量表示。

# 上下文信息补充
# 为了解决文本分割带来的上下文缺失的问题，我们对分割后的文本进行了以下信息的补充。

# 章节标题信息补充
# 针对Markdown这种有明显章节结构的文档，我们按章节进行文本分割，如果某个章节的文本太长，我们会进一步进行切分，并且在文本开头补充对应章节的标题信息。

# 全文核心词信息补充
# 全文的核心词也是重要的上下文信息，例如在图3所示的例子中，我们用核心词提取算法，提取出全文讨论的核心词为“企业直播“和”自定义鉴权“，作为文本块信息的补充，那么文本块2就很容易被检索回来。

# 2.3 向量检索优化
# 向量检索对应于流程图中的步骤5-7和11，其中的embedding模型是文档向量检索的核心。关于embedding模型的介绍，可以参考这篇文章——5分钟搭建基于LLM的智能文档助手，文中对现有的开源中英文embedding模型做了对比，其中最好的模型，在内部文档的测评集上，top3召回率只能达到64.8%。我们在实践中也有类似的观察，中文开源embedding模型的能力还不是很强，并且模型在不同语言上也有不同的表现，有的模型在纯中文文本上召回率高，有的模型在中英混合的文本上召回率高。

# 基于这些实验观察，我们对向量检索流程进行了以下优化：

# 多embedding召回
# 既然单个embedding模型效果不好，且不同的模型有不同的擅长方向，我们可以同时使用多个embedding模型，综合多个embedding模型的结果来衡量query和文本块的相似度。

# 召回-精排两阶段检索
# 我们参考信息检索领域常用的召回-精排两阶段的模式，在召回阶段召回较多的文本块（例如20个），保证相关的文本尽可能多地被召回，然后在精排阶段，综合文本块的各方面信息，对文本块重新进行打分，最终得到最相关的k个文本块。可以参考的信息包括：文本块的各级标题信息、全文的核心词、文本块中每个句子与query的相似度等。

# 2.4 query增强优化
# query增强是对流程图中的步骤8的完善。在大部分embedding模型中，用户query中的每个词在最终的embedding中所占的权重是一样的，这就导致文档召回的效果可能受到query中无关词的影响。

# 为了解决这一问题，我们在用户的query转换为embedding前，额外增加了query增强的优化。

# 具体的，我们调用Venus平台部署的Ziya-13B模型，通过few-shot prompting，让模型从用户的query中提取关键词，例如从”告诉我错误码3003的含义是什么“，我们可以提取出关键词”“错误码”和“3003”。提取后关键词将拼接到用户的query中，这样，最终的embedding中，“错误码”和“3003”由于出现次数更多，所占的权重就越大，更有利于检索到相关的文档。图4和图5显示了优化前后这一问题的检索和回答效果。


# 图4 query增强优化演示（优化前）

# 图5 query增强优化演示（优化后）
# 可以看到，使用query增强前，检索出的文档是错误的，导致LLM回答错误；使用query增强后，“错误码3003”相关的文本成功被检索回来。