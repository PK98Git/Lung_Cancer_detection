import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('lung_modelKNN', 'rb'))


@app.route('/')
def home():
  return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
  input_features = [int(x) for x in request.form.values()]
  features_value = [np.array(input_features)]
  #print(features_value);

  features_name = ['Gender','Age','smoking', 'yellow_fingers', 'anxiety','peer_pressure', 'chronic_disease', 'fatigue','allergy', 'wheezing', 'alcohol_consiuming', 'coughing', 'shortness_of_breath', 'swallowing difficulty', 'chest_pain']

  #df = pd.DataFrame([[80, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2]])
  df = pd.DataFrame(features_value,columns=features_name)

  #output = model.predict([[25, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,2,2]])
  output = model.predict(df)
  if output == 1:
      res_val = " cancer"
  if output == 0:
      res_val = "no cancer"

  return render_template('index.html', prediction_text='Patient has {}'.format(res_val))


if __name__ == "__main__":
  app.run()(debug=false,host='0,0,0,0')
