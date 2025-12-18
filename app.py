from flask import Flask, render_template, request, jsonify
import joblib
import re
import math
import requests

app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load("math_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def extract_numbers(text):
    """Extract all numbers from the question text."""
    return list(map(int, re.findall(r'\d+', text)))


def ollama_mistral_answer(question):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "mistral:latest",
        "prompt": (
            f"{question}"
        ),
        "system": (
            "You are a math assistant for Class 1–5 students. "
            "Always answer in one or two short sentences. "
            "Do not give long explanations. "
            "Explain in simple words and show steps if possible."
        ),
        "stream": False
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "No answer received.")
    except Exception as e:
        return f"Ollama error: {str(e)}"


def solve_question(question):
    """Solve structured math or send to Ollama if not clear."""
    try:
        # Try predicting intent
        prediction = model.predict(vectorizer.transform([question]))
        label = prediction[0] if len(prediction) > 0 else "unknown"
        numbers = extract_numbers(question)

        if label == "addition" and len(numbers) >= 2:
            return str(sum(numbers))
        elif label == "subtraction" and len(numbers) >= 2:
            return str(numbers[0] - numbers[1])
        elif label == "multiplication" and len(numbers) >= 2:
            return str(numbers[0] * numbers[1])
        elif label == "division" and len(numbers) >= 2:
            return "Cannot divide by zero." if numbers[1] == 0 else str(numbers[0] / numbers[1])
        elif label == "area_rectangle" and len(numbers) >= 2:
            return str(numbers[0] * numbers[1])
        elif label == "perimeter_square" and len(numbers) >= 1:
            return str(4 * numbers[0])
        elif label == "even_odd" and len(numbers) >= 1:
            return "Even" if numbers[0] % 2 == 0 else "Odd"
        elif label == "place_value" and len(numbers) >= 2:
            digit = numbers[0]
            number = str(numbers[1])
            if str(digit) not in number:
                return "Digit not found."
            pos = number[::-1].index(str(digit))
            return str(digit * (10 ** pos))
        elif label == "table" and len(numbers) >= 1:
            num = numbers[0]
            return "\n".join([f"{num} x {i} = {num * i}" for i in range(1, 11)])
        elif label == "lcm" and len(numbers) >= 2:
            return str(math.lcm(numbers[0], numbers[1]))
        elif label == "hcf" and len(numbers) >= 2:
            return str(math.gcd(numbers[0], numbers[1]))
        elif label == "prime_check" and len(numbers) >= 1:
            n = numbers[0]
            if n < 2:
                return "Not Prime"
            for i in range(2, int(n**0.5) + 1):
                if n % i == 0:
                    return "Not Prime"
            return "Prime"
        elif label == "square" and len(numbers) >= 1:
            return str(numbers[0] ** 2)
        elif label == "cube" and len(numbers) >= 1:
            return str(numbers[0] ** 3)
        elif label == "factorial" and len(numbers) >= 1:
            return str(math.factorial(numbers[0]))
        elif label == "area_circle" and len(numbers) >= 1:
            return str(round(math.pi * (numbers[0] ** 2), 2))
        elif label == "volume_cube" and len(numbers) >= 1:
            return str(numbers[0] ** 3)
        elif label == "volume_cuboid" and len(numbers) >= 3:
            return str(numbers[0] * numbers[1] * numbers[2])
        else:
            # Fallback to Ollama if intent unclear
            return ollama_mistral_answer(question)

    except Exception:
        # If classifier fails, fallback to Ollama
        return ollama_mistral_answer(question)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/solve", methods=["POST"])
def solve():
    data = request.get_json()
    question = data.get("question", "")
    answer = solve_question(question)
    return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run(debug=True)
