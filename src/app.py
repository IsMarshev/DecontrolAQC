from flask import Flask, render_template, request, redirect, url_for, session
import csv
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
        session['csv_data'] = uploaded_file.read().decode('utf-8').splitlines()
        return redirect(url_for('display'))
    return redirect(url_for('index'))


@app.route('/display', methods=['GET', 'POST'])
def display():
    lines = sorted(list(set([line[0] for line in list(csv.reader(
        session.get('csv_data', [])))[1:]])), key=lambda x: int(x[0]))
    search_query = request.form.get('search', '')
    if search_query:
        lines = list(
            set([str(line[0]) for line in lines if str(search_query) in str(line[0])]))
    return render_template('display.html', lines=lines, search_query=search_query)


@app.route('/details/<id>')
def details(id):
    lines = list(csv.reader(session.get('csv_data', [])))
    line = next((line for line in lines if line[0] == id), None)
    if line:
        return render_template('details.html', line=line)
    else:
        return 'Строка с таким ID не найдена', 404


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
