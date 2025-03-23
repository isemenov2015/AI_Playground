import os
import json, ast

from flask import Flask, render_template, request, jsonify
from openai import OpenAI

from dotenv import load_dotenv


app = Flask(__name__, static_folder='.')


load_dotenv()
openai_api_key = os.environ.get("OPENAI_API_KEY")


def generate_response(system_prompt="You are an AI assistant of general purpose", 
                      user_prompt="Greet me, please!", 
                      chat_history=[],
                      llm_version="gpt-4o-mini", 
                      llm_temperature=30):
    try:
        client = OpenAI(api_key=openai_api_key)

        # Build the conversation history in the format expected by OpenAI
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(chat_history)  # Append full chat history
        messages.append({"role": "user", "content": user_prompt})  # Add the latest user prompt


        # print('MESSAGES: \n\n', messages)

        if not llm_version in ["gpt-4o-mini", "gpt-3.5-turbo"]:
            response = client.chat.completions.create(
                model=llm_version,
    #            temperature=1.0 * llm_temperature / 100,  # Scale to OpenAI's range
                timeout=None,
                messages=messages,
                reasoning_effort="medium",
            )
            output = response.choices[0].message.content
        else:
            response = client.chat.completions.create(
                model=llm_version,
                temperature=1.0 * llm_temperature / 100,  # Scale to OpenAI's range
                timeout=None,
                messages=messages,
            )
            # print(f"RESPONSE: \n\n{response.choices[0].message.content}")
            output = response.choices[0].message.content

        return output  # Return response

    except Exception as e:
        print(f"An error occurred: {e}")
        return json.dumps({"error": "Bad LLM response: something went wrong"})  # Return JSON error


@app.route('/')
def index():
    system_prompt_content = ""
    try:
        with open('prompt_system.txt', 'r') as file:
            system_prompt_content = file.read()
    except FileNotFoundError:
        system_prompt_content = "System prompt file not found."    
    return render_template('index.html', system_prompt_content=system_prompt_content)


@app.route('/chat', methods=['POST'])
def chat():
    system_prompt = request.json['system_prompt']
    user_prompt = request.json['user_prompt']
    user_message = request.json['user_message']
    llm_version = request.json['llm_version']
    llm_temperature = request.json['llm_temperature']
    chat_history = request.json['chat_history']
    history_type = request.json['history_type']

    if history_type == "summary" and 'gpt-' in user_prompt:
        try:
            history = user_prompt.split(': ')[2]
            history = json.loads(history)
            chat_history = history['summary']
        except:
            pass

    # Update user prompt with previous messages
    user_prompt += f"\nUser: {user_message}"

#    print(f"Chat history_type: {history_type}\nChat history: {chat_history}")

    response = generate_response(system_prompt, user_prompt, chat_history, llm_version, llm_temperature)

#    print(f"LLM response : {response}")

    if "gpt-3.5-turbo" in response:  # workaround for ChatGPT 3.5 incorrect JSON output
        response = response.split(": ")[1]

    response = json.loads(response)

    return response


@app.route('/save', methods=['POST'])
def save_dialog():
#    print(request.json)
    dialog = request.json['chat_history']
    with open('dialog.txt', 'w') as f:
        f.write(json.dumps(dialog))
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True)