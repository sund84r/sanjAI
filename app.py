from flask import Flask, render_template, request, jsonify
from function import generate_response

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat(): #hello
    data = request.get_json(silent=True) or {}
    user_message = (data.get("message") or "").strip()

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    try:
        reply = generate_response(
            user_message,
            max_new_tokens=int(data.get("max_new_tokens", 200)),
            temperature=float(data.get("temperature", 0.7)),
            top_p=float(data.get("top_p", 0.9)),
            repetition_penalty=float(data.get("repetition_penalty", 1.05)),
        )

        # 🔹 Clean reply (strip whitespace/newlines)
        clean_reply = reply.strip()

        return jsonify({
            "success": True,
            "user_message": user_message,
            "reply": clean_reply
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, threaded=True)
