# Flask Chat Application

This is a simple Flask-based web application that allows users to send messages and interact with a language model. The application also lets users save the chat history as a text file.

## Features

- A web-based interface to chat with the language model.
- Save chat history to a `.txt` file.
- Select different language model versions and set temperature for the model's response generation.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- **Python 3.x** installed on your machine.
- A **GitHub** account to clone the repository.
- **PyCharm** (or any other Python IDE) installed.
- An **OpenAI API key** (store this in a `.env` file).

## Setup Instructions in PyCharm

Follow these steps to set up the project in PyCharm:

### 1. Clone the Repository

1. Open **PyCharm**.
2. Navigate to `File` > `New Project` > `Get from Version Control`.
3. In the `Get from Version Control` dialog, paste the GitHub repository URL:

    ```bash
    https://github.com/your-username/flask-chat-app.git
    ```

4. Click **Clone**.

### 2. Create a Virtual Environment

1. Go to `File` > `Settings` > `Project: <project_name>` > `Python Interpreter`.
2. Click on the gear icon ⚙️ and select **Add** > **New Virtual Environment**.
3. Select a base interpreter (Python 3.x) and click **OK** to create the virtual environment.

### 3. Install Required Dependencies

1. Once the project is open, install the required packages by running the following command in the terminal or adding it to your PyCharm environment:

    ```bash
    pip install -r requirements.txt
    ```

   If the `requirements.txt` file does not exist, you can manually install the dependencies:

    ```bash
    pip install Flask python-dotenv langchain-openai
    ```

2. After installation, ensure that all the necessary libraries are installed properly by checking the `Python Interpreter` in the PyCharm settings.

### 4. Set Up the `.env` File

1. In the root directory of the project, create a `.env` file.
2. Add your OpenAI API key to the `.env` file:

    ```env
    OPENAI_API_KEY=your_openai_api_key
    ```

### 5. Run the Flask Application

1. In PyCharm, open the `py script` (playground.py).
2. In the top right corner, click on the **Run/Debug Configurations** dropdown and select **Edit Configurations**.
3. Click the **+** icon to add a new configuration.
4. Choose **Flask Server** and configure it:
    - **Host**: `127.0.0.1`
    - **Port**: `5000`
    - **Environment Variables**: Point to your `.env` file to load the OpenAI API key.
5. Click **OK** and **Run** the Flask application.

### 6. Access the Web Interface

1. Once the Flask server is running, open a browser and navigate to:

    ```bash
    http://127.0.0.1:5000
    ```

2. You should see the web interface where you can interact with the chat application.

### 7. Save Chat History

- After chatting with the model, click the **Save Dialog** button to download the chat history as a `.txt` file.

---

## Additional Commands

### Run the Application

If you're running the application manually from the terminal:

```bash
flask run
