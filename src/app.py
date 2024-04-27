from flask import Flask, render_template, request, redirect, url_for, session
import csv
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']
    if uploaded_file and uploaded_file.filename.endswith('.csv'):
        file_path = os.path.join('static', uploaded_file.filename)
        uploaded_file.save(file_path)
        session['csv_data_file_path'] = file_path
        return redirect(url_for('display'))
    return redirect(url_for('index'))

@app.route('/display', methods=['GET', 'POST'])
def display():
    file_path = session.get('csv_data_file_path')
    file_data = pd.read_csv(file_path)
    lines = file_data['ID урока'].unique()
    lines = lines.astype(int)
    search_query = request.form.get('search', '')
    if search_query:
        lines = [str(line) for line in list(lines) if str(search_query) in str(line)]
    return render_template('display.html', lines=lines, search_query=search_query)

@app.route('/details/<id>')
def details(id):
    file_path = session.get('csv_data_file_path')
    file_data = pd.read_csv(file_path)
    line = file_data.loc[file_data['ID урока'] == int(id)]['Текст сообщения'].to_list()
    if line:
        return render_template('details.html', line=line)
    else:
        return 'Строка с таким ID не найдена', 404

if __name__ == '__main__':
    app.run(debug=True)
