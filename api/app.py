import openai
from flask import Flask, render_template, request

app = Flask(__name__)

openai.api_key = "your-api-key"

def get_gpt3_response(prompt):
    try:
        response = openai.Completion.create(
          engine="gpt-3.5-turbo-instruct",
          prompt=prompt,
          max_tokens=150
        )
        return response.choices[0].text.strip()
    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        question = request.form['question']
        answer = get_gpt3_response(question)
        return render_template('index.html', answer=answer)
    return render_template('index.html', answer=None)

if __name__ == '__main__':
    app.run(debug=True)
