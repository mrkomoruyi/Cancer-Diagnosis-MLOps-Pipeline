from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            age=request.form['age'],
            gender=request.form['gender'],
            bmi=request.form['bmi'],
            smoking=request.form['smoking'],
            genetic_risk=request.form['genetic_risk'],
            physical_activity=request.form['physical_activity'],
            alcohol_intake=request.form['alcohol_intake'],
            cancer_history=request.form['cancer_history']
        )

        pred_df = data.get_data_as_dataframe()

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html', results=results[0])


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
