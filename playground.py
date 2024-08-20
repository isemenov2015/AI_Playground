import os
import json

from flask import Flask, render_template, request, jsonify
from langchain.llms import OpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chains import LLMChain

from dotenv import load_dotenv


app = Flask(__name__)

# Replace with your OpenAI API key
load_dotenv()
openai_api_key = os.environ.get("OPENAI_API_KEY")
llm = OpenAI(openai_api_key=openai_api_key)

def generate_response(system_prompt, user_prompt):
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    llm_chain = LLMChain(llm=llm, prompt=messages)
    response = llm_chain.run("")
    return response

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    system_prompt = request.json['system_prompt']
    user_prompt = request.json['user_prompt']
    user_message = request.json['user_message']

    # Update user prompt with previous messages
    user_prompt += f"\nUser: {user_message}"

    response = generate_response(system_prompt, user_prompt)
    return jsonify({'response': response})

@app.route('/save', methods=['POST'])
def save_dialog():
    dialog = request.json['dialog']
    with open('dialog.txt', 'w') as f:
        f.write(json.dumps(dialog))
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True)