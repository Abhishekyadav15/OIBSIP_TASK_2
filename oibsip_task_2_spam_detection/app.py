# app.py

from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('model_svm.pkl')


label_mapping = {
    'ham': 'Not Spam',
    'spam': 'Spam'
}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        
        email_text = request.form['email_text']
        
        prediction = model.predict([email_text])[0]
        
        result = label_mapping.get(prediction, 'Unknown') 
        
        return render_template('predict.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
