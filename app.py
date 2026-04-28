from flask import Flask, render_template, request, make_response
import joblib as jb
import pandas as pd

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


# SIMPLE ENCODING (fit + transform)
def encode(df):
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for col in df.columns:
        df[col] = le.fit_transform(df[col].astype(str))
    return df


@app.route('/predict', methods=['POST'])
def predict():

    file = request.files['file']
    df = pd.read_csv(file)

    data = encode(df)
    kbest = jb.load('kbest.pkl')
    data = kbest.transform(data)

    model = jb.load('model.pkl')
    result = model.predict(data)

    # convert to DataFrame
    result_df = pd.DataFrame(result, columns=['Prediction'])

    # label convert
    result_df['Prediction'] = result_df['Prediction'].replace({
        1: 'Poisonous',
        0: 'Edible'
    })

    # convert to csv
    csv = result_df.to_csv(index=False)

    # response
    response = make_response(csv)
    response.headers['Content-Disposition'] = 'attachment; filename=predictions.csv'
    response.headers['Content-type'] = 'text/CSV'

    return response


if __name__ == "__main__":
    app.run(debug=True)