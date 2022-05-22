from flask import Flask ,request,jsonify
import numpy as np
import pickle
import sklearn
print(sklearn.__version__)
model = pickle.load(open('pipe2.pkl','rb'))

app =Flask(__name__)

@app.route('/predict',methods = ['POST'])
def index():
    preg = request.form.get('Pregnancies')
    glu = request.form.get('Glucose')
    bp = request.form.get('BloodPressure')
    st = request.form.get('SkinThickness')
    ins = request.form.get('Insulin')
    bmi = request.form.get('BMI')
    dpf = request.form.get('DiabetesPedigreeFunction')
    age = request.form.get('Age')

    input = np.array([[preg,glu,bp,st,ins,bmi,dpf,age]])

    result = model.predict_proba(input)[0][1]

    return jsonify({'risk status' : result})

if __name__ == '__main__' :
    app.run(debug=True)