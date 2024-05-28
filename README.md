# ScamDetect: A Scam Detection Chatbot

a Streamlit application powered by Large Language Model "LLAMA3", helps you identify potential scam texts. This scam detection chatbot analyzes user-provided text using a fine-tuned DistilBERT model, assigning a "scam" or "safe" label with likelihood. Leveraging Langchain and Ollama frameworks, it explains why a text might be risky, offering an interactive and user-friendly experience through Streamlit's user interface.

### Features:

- Analyze suspicious text and receive a classification (scam or safe) with a likelihood score.
- Get explanations for the classification to understand why the text might be a scam.
- Learn tips on how to identify scam texts in the future.
- Maintain a chat history for previous interactions.

### Requirements:

- Python 3.x
- Streamlit
- TensorFlow
- Transformers
- Langchain

### Installation:

1. Clone this repository.
2. Install required libraries using pip install -r requirements.txt.
3. Install Ollama locally for faster response times (https://github.com/ollama/ollama).

### Usage:

- Run Ollama with ollama serve.
- Run the application using streamlit run app.py.
- Enter a suspicious text in the chat window.
- Click "Enter" to receive the analysis from the chatbot.

### Technical Details:

- The application utilizes a pre-trained text classification model (pippinnie/scam_text_classifier) to classify input text as scam or safe.
- It leverages Ollama (specifically LLM model "llama3") to analyze the text and provide explanations for the classification.
- Chat history is maintained using Streamlit's session state for a more interactive experience.

### Additional Notes:

- This is a basic example, and the prompt template and explanation generation can be further customized for better performance.
- Consider adding functionalities like integrating with external scam databases.
