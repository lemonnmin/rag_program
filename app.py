import os
from dotenv import load_dotenv
from flask import Response, Flask, request, jsonify, stream_with_context
from flask_cors import CORS  # 导入 CORS
from embed import embed
from query import query
from get_vector_db import get_vector_db

load_dotenv()

TEMP_FOLDER = os.getenv('TEMP_FOLDER', './_temp')
os.makedirs(TEMP_FOLDER, exist_ok=True)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})  # 启用 CORS

@app.route('/embed', methods=['POST'])
def route_embed():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
   
    embedded = embed(file)
    if embedded:
        return jsonify({"message": "File embedded successfully"}), 200

    return jsonify({"error": "File embedded unsuccessfully"}), 400

@app.route('/query', methods=['POST'])
def route_query():
    data = request.get_json()
    contents = [msg['content'] for msg in data['messages']]
    def generate_response():
        response_stream = Response()
        return query(contents[0], contents[1], response_stream, contents[2])

    return Response(stream_with_context(generate_response()), content_type='text/plain')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)