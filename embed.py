import os
from datetime import datetime
from werkzeug.utils import secure_filename
from langchain_community.document_loaders import PyPDFLoader,UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from get_vector_db import get_vector_db
import logging
TEMP_FOLDER = os.getenv('TEMP_FOLDER', './_temp')

# Function to check if the uploaded file is allowed (only PDF files)
# 文件类型检查
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'pdf'}

# Function to save the uploaded file to the temporary folder
# 唯一的时间戳临时目录
def save_file(file):
    # Save the uploaded file with a secure filename and return the file path
    ct = datetime.now()
    ts = ct.timestamp()
    filename = str(ts) + "_" + secure_filename(file.filename)
    file_path = os.path.join(TEMP_FOLDER, filename)
    file.save(file_path)

    return file_path

# Function to load and split the data from the PDF file
def load_and_split_data(file_path):
    # Load the PDF file and split the data into chunks
    loader = PyPDFLoader(file_path=file_path)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200, separators=["\n", "。", "！", "？", "，", "、", ""],add_start_index=True)
    chunks = text_splitter.split_documents(data)

    return chunks

# Main function to handle the embedding process


# 设置日志记录
logging.basicConfig(level=logging.INFO)

def embed(file):
    # 检查文件是否合法，保存文件，加载并拆分数据，添加到数据库，并删除临时文件
    if file.filename != '' and allowed_file(file.filename):
        print("文件传输处理")
        try:
            file_path = save_file(file)
            chunks = load_and_split_data(file_path)
            db = get_vector_db()
            print("文本",chunks)
            db.add_documents(chunks)
            db.persist()
            logging.info(f'成功处理并添加了 {len(chunks)} 个块来自 {file.filename}。')
        except Exception as e:
            logging.error(f'处理文件 {file.filename} 时出错：{e}')
            return False
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
                logging.info(f'已删除临时文件 {file_path}。')
        return True

    logging.warning('无效文件或未上传文件。')
    return False
