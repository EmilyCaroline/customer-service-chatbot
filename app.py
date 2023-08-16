from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from chat import get_response
import pyodbc
import pandas as pd
import numpy as np; 

app = Flask(__name__)
CORS(app)

def connection():
    s = 'localhost' #Your server name 
    d = 'chatagent' 
    u = 'chatadmin' #Your login
    p = '12345' #Your login password
    cstr = 'DRIVER={ODBC Driver 17 for SQL Server};SERVER='+s+';DATABASE='+d+';UID='+u+';PWD='+ p
    conn = pyodbc.connect(cstr)
    return conn

# @app.get("/")
# def index_get():
#     return render_template("base.html")

@app.get('/')
def index():
    csv_file = "data/dataset.csv"
    df = pd.read_csv(csv_file)
    print(len(df))
    return 'Hello, World!'

def insert_data():
    conn = connection()
    cursor = conn.cursor()
    csv_file = "data/dataset.csv"
    df = pd.read_csv(csv_file)

    for i, row in df.iterrows():
        flags = row['flags']
        utterance = row['utterance']
        category = row['category']
        intent = row['intent']
        print(intent)
        cursor.execute("INSERT INTO dbo.intents_data (flags, utterance, category, intent) VALUES (?, ?, ?, ?)", flags, utterance, category, intent)
    
    # the connection is not autocommitted by default, so we must commit to save our changes
    conn.commit()
    conn.close()
    

@app.post("/predict")
def predict():
    text = request.get_json().get("message")
    response = get_response(text)
    message = {"answer":response}
    return jsonify(message)

if (__name__) == "__main__":
    app.run(debug=True)