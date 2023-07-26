import openai
import cohere
import os
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

openai.api_key = os.getenv("OPENAI_API_KEY")
co = cohere.Client(os.getenv("COHERE_API_KEY"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/gpt-3.5-turbo", methods=["POST"])
def openai_call():
    prompt = request.json["prompt"]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,

    )

    answer = response.choices[0].message.content

    return jsonify({"answer": answer})


@app.route("/command-nightly", methods=["POST"])
def cohere_call():
    prompt = request.json["prompt"]

    response = co.generate(
        model="command-nightly",
        prompt=prompt,
        max_tokens=200,
    )

    answer = response.generations[0].text.strip()

    return jsonify({"answer": answer, })


if __name__ == "__main__":
    app.run(debug=True)
