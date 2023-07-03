from flask import Flask, render_template, request, redirect
import numpy as np
import pickle

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        with open("model.pkl", "rb") as model_file:
            model = pickle.load(model_file)

        variance = float(request.form['variance'])
        skewness = float(request.form['skewness'])
        curtosis = float(request.form['curtosis'])
        entropy = float(request.form['entropy'])

        inputs = np.array([[variance, skewness, curtosis, entropy]])
        output = {0: "Authentic", 1: "Counterfeit"}
        pred = output[model.predict(inputs)[0]].lower()

        return render_template("index.html", output=f"The banknote is {pred}")
    else:
        return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)