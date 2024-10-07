import os
import json

from flask import Flask, render_template, request, jsonify
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder

from dotenv import load_dotenv


app = Flask(__name__, static_folder='.')


load_dotenv()
openai_api_key = os.environ.get("OPENAI_API_KEY")


def generate_response(system_prompt="You are an AI assistant of general purpose", 
                      user_prompt="Greet me, please!", 
                      chat_history="",
                      llm_version="gpt-3.5-turbo", 
                      llm_temperature=30):
    try:
        # Initialize the LLM with the given parameters
        llm = ChatOpenAI(
            model=llm_version,
            temperature=1.0 * llm_temperature / 100,  # Pass the temperature value
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key=openai_api_key
        )

        # Create a ChatPromptTemplate object to format the prompt
        formatted_prompt = f"""System: {system_prompt} \nChat history: {chat_history}\n Human: {user_prompt}"""

#        print(formatted_prompt)

        # Generate a response using the formatted prompt
        response = llm.generate([formatted_prompt])
        llm_text_response = response.generations[0][0].text

        # Extract the generated response text from the LLM output
        generated_response = llm_text_response if response.generations else "Bad LLM response"#

        return generated_response

    except Exception as e:
        print(f"An error occurred: {e}")
        return f"{llm_version}: Bad LLM response: something went wrong"


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