from flask import Flask, request, jsonify 
import subprocess

app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.json
        prompt = data.get("prompt", "Hello, DeepSeek!")

        # Call Ollama using subprocess
        result = subprocess.run(
            ["ollama", "run", "deepseek", prompt],
            capture_output=True,
            text=True
        )

        return jsonify({"response": result.stdout.strip()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
