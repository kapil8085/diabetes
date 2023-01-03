from flask import Flask, render_template, request
import config
import utils

app= Flask(__name__)

@app.route("/")
def hello_flask():
    return render_template("index.html")


@app.route("/predict_diabetes",methods=["GET", "POST"])
def get_pred():
    user_data= request.form

    prediction = utils.get_prediction(user_data)
    return render_template("index.html",probability=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=config.PORT_NUMBER,debug=True)    