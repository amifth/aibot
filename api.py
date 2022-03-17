import re
from flask import Flask, render_template, request, jsonify
from app import get_response

app = Flask(__name__)

# define app routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get")
def response_to_user():
    userText = request.args.get('msg')
    print(userText)
    res = get_response(userText)
    return str(res[0])

@app.route("/predict", methods=['POST'])
def get_bot_response():
    json_ = request.json
    res = get_response(json_['request'])
    print(res[0])
    print(type(res[0]))
    res0 = res[0]
    res1 = res[1]
    data = [{"ai_response": str(res0), "accuracy": str(res1)}]
    return jsonify({"status" : 200, "message" : "Successfully!", "errors": None,"data" : data})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
