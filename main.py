from flask import Flask ,request,jsonify
import numpy as np
import pickle
import sklearn
model = pickle.load(open('pipe2.pkl','rb'))

app =Flask(__name__)

@app.route('/predict',methods = ['POST'])
def index():
    preg = int(request.form.get('Pregnancies'))
    glu = int(request.form.get('Glucose'))
    bp = int(request.form.get('BloodPressure'))
    st = int(request.form.get('SkinThickness'))
    ins = int(request.form.get('Insulin'))
    bmi = float(request.form.get('BMI'))
    dpf = float(request.form.get('DiabetesPedigreeFunction'))
    age = int(request.form.get('Age'))

    input = np.array([[preg,glu,bp,st,ins,bmi,dpf,age]])

    result = model.predict_proba(input)[0][1]

    return jsonify({'risk status' : result})

if __name__ == '__main__' :
    app.run(debug=True)