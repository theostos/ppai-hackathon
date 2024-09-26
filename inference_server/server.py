import os

from flask import Flask, request, jsonify

from .models.base_llm import BaseLLM

INFERENCE_SERVER_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(INFERENCE_SERVER_ROOT, 'config', 'llama.yaml')

app = Flask(__name__)
llm = BaseLLM(CONFIG_PATH)
@app.route('/inference', methods=['GET', 'POST'])
def inference():
    content = request.json
    llm_output = llm(**content)
    return jsonify({"outputs": llm_output})

if __name__ == '__main__':
    app.run(host= '0.0.0.0')