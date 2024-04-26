# from flask import Flask, render_template, request
# import requests
# import json

# app = Flask(__name__)

# # Function to generate response using Mistral
# def generate_response(prompt):
#     url = "http://localhost:11434/api/generate"
#     headers = {'content-type': 'application/json'}
#     data = {
#         "model": "Brahmmy",
#         "stream": False,
#         "prompt": prompt
#     }
#     response = requests.post(url, headers=headers, data=json.dumps(data))
#     if response.status_code == 200:
#         response_text = response.text
#         data = json.loads(response_text)
#         return data.get("response", "No response found in data.")
#     else:
#         error_msg = f"Error: {response.status_code} - {response.text}"
#         return error_msg

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     chat_history = []
#     if request.method == 'POST':
#         user_question = request.form['user_question']
#         if user_question.strip():
#             try:
#                 response = generate_response(user_question)
#                 message = {'human': user_question, 'AI': response}
#                 chat_history.append(message)
#             except Exception as e:
#                 error_msg = "Error:", e
#                 return render_template('offline_index.html', error_msg=error_msg, chat_history=chat_history)
#     return render_template('offline_index.html', chat_history=chat_history)

# if __name__ == '__main__':
#     app.run(debug=True)



from flask import Flask, render_template, request, jsonify
import requests
import json

app = Flask(__name__)

# Function to generate response using Mistral
def generate_response(prompt):
    url = "http://localhost:11434/api/generate"
    headers = {'content-type': 'application/json'}
    data = {
        "model": "Brahmmy",
        "stream": False,
        "prompt": prompt
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        response_text = response.text
        data = json.loads(response_text)
        return data.get("response", "No response found in data.")
    else:
        error_msg = f"Error: {response.status_code} - {response.text}"
        return error_msg

@app.route('/', methods=['GET', 'POST'])
def index():
    chat_history = []
    if request.method == 'POST':
        user_question = request.form['user_question']
        if user_question.strip():
            try:
                response = generate_response(user_question)
                message = {'human': user_question, 'AI': response}
                chat_history.append(message)
            except Exception as e:
                error_msg = "Error:", e
                return render_template('offline_index.html', error_msg=error_msg, chat_history=chat_history)
    return render_template('offline_index.html', chat_history=chat_history)

@app.route('/pregenerated_questions', methods=['GET'])
def pregenerated_questions():
    # Pregenerated questions
    pregenerated_questions = [
        "What is this?",
        "Who created this?",
        "What can we do with this?"
    ]
    return jsonify(pregenerated_questions)

if __name__ == '__main__':
    app.run(debug=True)
