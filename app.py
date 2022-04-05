from flask import Flask,render_template,request
import pickle
import numpy as np

model = pickle.load(open('Wine_Quality_predictor.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/index1')
def index1():
    return render_template('index1.html')

@app.route('/predict',methods=['POST'])
def predict():
    type = float(request.form.get('type'))
    fixed_acidity = float(request.form.get('fixed_acidity'))
    volatile_acidity = float(request.form.get('volatile_acidity'))
    citric_acid = float(request.form.get('citric_acid'))
    residual_sugar = float(request.form.get('residual_sugar'))
    chlorides = float(request.form.get('chlorides'))
    free_sulphur_dioxide = float(request.form.get('free_sulphur_dioxide'))
    total_sulphur_dioxide = float(request.form.get('total_sulphur_dioxide'))
    density = float(request.form.get('density'))
    ph = float(request.form.get('ph'))
    sulphates = float(request.form.get('sulphates'))
    alcohol = float(request.form.get('alcohol'))

    result = model.predict(np.array([type,fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulphur_dioxide,total_sulphur_dioxide,density,ph,sulphates,alcohol]).reshape(1, 12))

    return render_template('result.html', result=result)



if __name__ == '__main__':
    app.run(debug=True)