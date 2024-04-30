from flask import Flask, render_template, request
import pickle
import numpy as np

import flask
import pickle
# Use pickle to load in the pre-trained model.
with open(f'model/journalmodel_xgboost.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)


@app.route('/')
def main():
    @app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))
    if flask.request.method == 'POST':
        ArtcileID = flask.request.form['id']
        Journal = flask.request.form['journal_class']
        Year = flask.request.form['year']
        input_variables = pd.DataFrame([[ArticleID. Journal, Year]],
                                       columns=['id', 'journal_class', 'year'],
                                       dtype=float)
        prediction = y_pred[0]
        return flask.render_template('home.html',
                                     original_input={'Article ID':id,
                                                     'Journal':journal_class,
                                                     'Year':year},
                                     result=prediction,
                                     )
    
    
if __name__ == '__main__':
    app.run(debug=True)
        