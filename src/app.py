from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
import torch
import pandas as pd
import os
from transformers import pipeline, BertForSequenceClassification, AutoTokenizer
import asyncio
from tqdm import tqdm

app = Flask(__name__)
app.secret_key = 'your_secret_key'
base_model_name = "cointegrated/rubert-tiny2" 
device = 'cuda' if torch.cuda.is_available() else 'cpu'

binary_classification_model = pipeline('text-classification', model=BertForSequenceClassification.from_pretrained("models/binary_classification_model"), tokenizer=AutoTokenizer.from_pretrained(base_model_name), device=device)
usefull_classification_model = pipeline('text-classification', model=BertForSequenceClassification.from_pretrained("models/usefull_classification_model"), tokenizer=AutoTokenizer.from_pretrained(base_model_name), device=device)
useless_classification_model = pipeline('text-classification', model=BertForSequenceClassification.from_pretrained("models/useless_classification_model"), tokenizer=AutoTokenizer.from_pretrained(base_model_name), device=device)


async def predict(text_data):
    mp_binary_label = {'LABEL_1':1, 'LABEL_0':0}
    mp_usefull_label = {'LABEL_0':'Вопрос о профессии','LABEL_1':'Вопросе о группе','LABEL_2':'Вопросы о ходе обучения','LABEL_3':'Недопонимание','LABEL_4':'Понимание','LABEL_5':'Мнение о мероприятии','LABEL_6':'Опыт','LABEL_7':'Технические вопросы','LABEL_8':'Технические проблемы','LABEL_9':'Обсуждение кода','LABEL_10':'Полезные ссылки'}
    mp_useless_label = {'LABEL_3':'Приветствие','LABEL_0':'Неформальные нейтральные сообщения','LABEL_4':'Позитивные неформальные сообщения','LABEL_2':'Комплименты','LABEL_1':'Флуд','LABEL_6':'Токсичные сообщения','LABEL_7':'Политика','LABEL_8':'Фишинговые ссылки','LABEL_5':'Прощания'}
    predicts = []
    scores = []
    for i in tqdm(range(0, len(text_data))):
        predicts_binary_class =     binary_classification_model(text_data[i], padding=True, truncation=True, max_length=128)
        label = mp_binary_label[predicts_binary_class[0]['label']]
        if label:
            predict = usefull_classification_model(text_data[i], padding=True, truncation=True, max_length=128)
            score = round(predict[0]['score'],3)
            predict = mp_usefull_label[predict[0]['label']]
        else:
            predict = useless_classification_model(text_data[i], padding=True, truncation=True, max_length=128)
            score = round(predict[0]['score'],3)
            predict = mp_useless_label[predict[0]['label']]
        scores.append(score)    
        predicts.append(predict)
    return {'predictions': predicts, 'scores': scores}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']
    if uploaded_file and uploaded_file.filename.endswith('.csv'):
        file_path = os.path.join('src/static', uploaded_file.filename)
        uploaded_file.save(file_path)
        file = pd.read_csv(file_path)
        file = file.dropna(subset='Текст сообщения')
        file.reset_index(drop=True,inplace=True)
        file.to_csv(file_path, index=False)
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
async def details(id):
    file_path = session.get('csv_data_file_path')
    file_data = pd.read_csv(file_path)
    text_data = file_data.loc[file_data['ID урока'] == int(id)]['Текст сообщения']
    text_data.reset_index(inplace=True,drop=True)
    result = await predict(text_data)
    print(result)
    line = text_data.to_list()
    if line:
        return render_template('details.html', line=line)
    else:
        return 'Строка с таким ID не найдена', 404

if __name__ == '__main__':
    app.run(debug=True)
