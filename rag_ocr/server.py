from flask import Flask, request, jsonify

app = Flask(__name__)


# 定义一个可被Java调用的接口
@app.route('/process/ocr', methods=['POST'])
def process_data():
    data = request.json  # 获取Java传递的JSON数据





    return jsonify({"result": result})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)  # 启动服务
