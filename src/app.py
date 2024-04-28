from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
import torch
import pandas as pd
import math
import os
from transformers import pipeline, BertForSequenceClassification, AutoTokenizer
import plotly.graph_objs as go
import plotly
import plotly.offline as py
import asyncio
from tqdm import tqdm

app = Flask(__name__)
app.secret_key = 'your_secret_key'
base_model_name = "cointegrated/rubert-tiny2" 
device = 'cuda' if torch.cuda.is_available() else 'cpu'

binary_classification_model = pipeline('text-classification', model=BertForSequenceClassification.from_pretrained("./src/models/binary_classification_model"), tokenizer=AutoTokenizer.from_pretrained(base_model_name), device=device)
usefull_classification_model = pipeline('text-classification', model=BertForSequenceClassification.from_pretrained("./src/models/usefull_classification_model"), tokenizer=AutoTokenizer.from_pretrained(base_model_name), device=device)
useless_classification_model = pipeline('text-classification', model=BertForSequenceClassification.from_pretrained("./src/models/useless_classification_model"), tokenizer=AutoTokenizer.from_pretrained(base_model_name), device=device)


async def predict(text_data):
    mp_binary_label = {'LABEL_1':1, 'LABEL_0':0}
    mp_usefull_label = {'LABEL_0':'Вопрос о профессии','LABEL_1':'Вопросе о группе','LABEL_2':'Вопросы о ходе обучения','LABEL_3':'Недопонимание','LABEL_4':'Понимание','LABEL_5':'Мнение о мероприятии','LABEL_6':'Опыт','LABEL_7':'Технические вопросы','LABEL_8':'Технические проблемы','LABEL_9':'Обсуждение кода','LABEL_10':'Полезные ссылки'}
    mp_useless_label = {'LABEL_3':'Приветствие','LABEL_0':'Неформальные нейтральные сообщения','LABEL_4':'Позитивные неформальные сообщения','LABEL_2':'Комплименты','LABEL_1':'Флуд','LABEL_6':'Токсичные сообщения','LABEL_7':'Политика','LABEL_8':'Фишинговые ссылки','LABEL_5':'Прощания'}
    predicts = []
    scores = []
    for i in tqdm(range(0, len(text_data))):
        predicts_binary_class =   binary_classification_model(text_data[i], padding=True, truncation=True, max_length=128)
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

def calculate_metrics(class_values):
    ru_eng = {
    'Неформальные нейтральные сообщения': 'non_formal_neutral',
    'Технические вопросы': 'tech_questions',
    'Обсуждение кода': 'code_discussion',
    'Флуд': 'flood',
    'Вопросы о ходе обучения': 'course_progress',
    'Недопонимание': 'misunderstanding',
    'Технические проблемы': 'tech_problems',
    'Комплименты': 'compliments',
    'Понимание': 'understanding',
    'Приветствие': 'greetings',
    'Токсичные сообщения': 'toxic',
    'Позитивные неформальные сообщения': 'positive_informal',
    'Полезные ссылки': 'useful_links',
    'Прощания': 'goodbyes',
    'Мнение о мероприятии': 'event_opinion',
    'Вопрос о профессии': 'professions',
    'Вопросе о группе': 'groups'
    }
    new = {}
    for k, v in class_values.items(): #переводим ключи на русский так как код писала лЛама, коммент удалить
        if k != 'Итог':
            new[ru_eng[k]] = v

    class_values = new
    class_values['misunderstanding'] = 1 - class_values['misunderstanding']
    class_values['flood'] = 1 - class_values['flood']
    class_values['toxic'] = 1 - class_values['toxic']
    class_values['tech_problems'] = 1 - class_values['tech_problems']

    successfulness_weights = {
        'understanding': 2,
        'compliments': 1,
        'professions': 1,
        'groups': 1,
        'course_progress': 1, 
        'useful_links': 1,
        'event_opinion': 1,
        'tech_problems': 1,
        'misunderstanding': 1,
        'toxic': 1
    }
    successfulness = math.sqrt(sum([
        (class_values[param] * weight)**2
        for param, weight in successfulness_weights.items()
    ])) / math.sqrt(sum(successfulness_weights.values()))

    # Calculate Student Discipline
    discipline_weights = {
        'positive_informal': 0.3,
        'compliments': 0.4,
        'professions': 0.10,
        'groups': 0.5,
        'course_progress': 0.15,
        'flood': 0.18,
        'toxic': 0.201,
    }
    discipline = math.sqrt(sum([
        (class_values[param] * weight)**2
        for param, weight in discipline_weights.items()
    ])) / math.sqrt(sum(discipline_weights.values()))

    # Calculate Teacher Professionalism
    professionalism_weights = {
        'understanding': 5,  
        'compliments': 4, 
        'useful_links': 1,
        'code_discussion': 1,
        'tech_questions': 2,
        'misunderstanding': 2
    }
    professionalism = math.sqrt(sum([
        (class_values[param] * weight)**2
        for param, weight in professionalism_weights.items()
    ])) / math.sqrt(sum(professionalism_weights.values()))

    return {
        'successfulness': successfulness,
        'discipline': discipline,
        'professionalism': professionalism
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
async def upload_file():
    uploaded_file = request.files['file']
    if uploaded_file and (uploaded_file.filename.endswith('.csv') or uploaded_file.filename.endswith('.xlsx')):
        file_path = os.path.join('./src/static', uploaded_file.filename)
        print(file_path[-4:])
        uploaded_file.save(file_path)
        if file_path[-4:]=='.csv':
            file = pd.read_csv(file_path)
        else:
            file = pd.read_excel(file_path)
        file = file.dropna(subset='Текст сообщения')
        file.reset_index(drop=True,inplace=True)
        result = await predict(file['Текст сообщения'])
        file['label'] = result['predictions']
        if file_path[-4:]!='.csv':
            file_path = file_path.replace('.xlsx','.csv')
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
    def show_plot(file_data):
        data = {'Hour': []}
        for i in range(len(file_data['Дата сообщения'])):
            try:
                data['Hour'].append(file_data['Дата сообщения'][i].split()[1][:2])
            except:
                pass
        data = pd.DataFrame(data)
        sorted_counts = data['Hour'].value_counts().sort_index()
        print(sorted_counts.values)
        fig = go.Figure(data=[go.Bar(x=sorted_counts.index, y=sorted_counts)])
        fig.update_layout(
            title='Гистограмма по часам',
            xaxis_title='Часы', 
            yaxis_title='Количество записей'
        )

        # Создаем HTML-код графика  
        plot_div = py.plot(fig, include_plotlyjs=False, output_type='div')
        return plot_div
    classification_result = file_data['label'].value_counts()
    def view_predictions(classification_result):
        classification_result=classification_result.to_dict()
        data = [go.Bar(x=list(classification_result.keys()), y=list(classification_result.values()))]
        layout = go.Layout(title='Результаты классификации',
        paper_bgcolor='rgba(0,0,0,0)',  # <--- Add this line
        plot_bgcolor='rgba(0,0,0,0)', autosize=True)
        fig = go.Figure(data=data, layout=layout)
        plot_div = plotly.offline.plot(fig, include_plotlyjs=False, output_type='div')
        return plot_div


    search_query = request.form.get('search', '')
    if search_query:
        lines = [str(line) for line in list(lines) if str(search_query) in str(line)]
    return render_template('display.html', lines=lines, search_query=search_query, show_plot = show_plot(file_data) ,view_predictions = view_predictions(classification_result))

@app.route('/details/<id>')
async def details(id):
    file_path = session.get('csv_data_file_path')
    file_data = pd.read_csv(file_path)
    data = file_data.loc[file_data['ID урока'] == int(id)]
    data.reset_index(inplace=True,drop=True)
    result = await predict(data['Текст сообщения'])
    # print(result['predictions'])
    classification_result = pd.DataFrame(result['predictions'], columns = ['label'])['label'].value_counts()
    classification_result_sum = sum(classification_result.to_list())
    
    pattern  = {'Неформальные нейтральные сообщения': 0,
                                'Технические вопросы': 0,
                                'Обсуждение кода': 0,
                                'Флуд': 0,
                                'Вопросы о ходе обучения': 0,
                                'Недопонимание': 0,
                                'Технические проблемы': 0,
                                'Комплименты': 0,
                                'Понимание': 0,
                                'Приветствие': 0,
                                'Токсичные сообщения': 0,
                                'Позитивные неформальные сообщения': 0,
                                'Полезные ссылки': 0,
                                'Прощания': 0,
                                'Мнение о мероприятии': 0,
                                'Вопрос о профессии': 0,
                                'Вопросе о группе': 0
                                }
    
    for k,v in classification_result.to_dict().items():
        pattern[k] = v/classification_result_sum
    def calculate_metrics(class_values):
        ru_eng = {
        'Неформальные нейтральные сообщения': 'non_formal_neutral',
        'Технические вопросы': 'tech_questions',
        'Обсуждение кода': 'code_discussion',
        'Флуд': 'flood',
        'Вопросы о ходе обучения': 'course_progress',
        'Недопонимание': 'misunderstanding',
        'Технические проблемы': 'tech_problems',
        'Комплименты': 'compliments',
        'Понимание': 'understanding',
        'Приветствие': 'greetings',
        'Токсичные сообщения': 'toxic',
        'Позитивные неформальные сообщения': 'positive_informal',
        'Полезные ссылки': 'useful_links',
        'Прощания': 'goodbyes',
        'Мнение о мероприятии': 'event_opinion',
        'Вопрос о профессии': 'professions',
        'Вопросе о группе': 'groups'
        }
        new = {}
        for k, v in class_values.items(): #переводим ключи на русский так как код писала лЛама, коммент удалить
            if k != 'Итог':
                new[ru_eng[k]] = v

        class_values = new
        class_values['misunderstanding'] = 0 - class_values['misunderstanding']
        class_values['flood'] = 0 - class_values['flood']
        class_values['toxic'] = 0 - class_values['toxic']
        class_values['tech_problems'] = 0 - class_values['tech_problems']
        print(class_values)
        successfulness_weights = {
            'understanding': 50,
            'compliments': 50,
            'professions': 15,
            'groups': 11,
            'course_progress': 15, 
            'useful_links': 11,
            'event_opinion': 14,
            'tech_problems': 11,
            'misunderstanding': 10,
            'toxic': 10,
            'code_discussion': 50, 
            'tech_questions': 35,
            'positive_informal': 15

        }
        numerator_sum = sum([
            (class_values[param]/abs(class_values[param]))*((class_values[param] * weight)**2)
            for param, weight in successfulness_weights.items()
            if class_values[param] != 0 
        ])
        if numerator_sum < 0:
            numerator_sum = 0
        denominator_sum = sum([
            weight
            for param, weight in successfulness_weights.items()
            if class_values[param] != 0
        ])

        successfulness = math.sqrt(numerator_sum) / math.sqrt(denominator_sum)
        discipline_weights = {
            'non_formal_neutral': 15,
            'positive_informal': 20,
            'compliments': 18,
            'professions': 12,
            'groups': 10,
            'course_progress': 10,
            'flood': 25,
            'toxic': 30,
        }
        numerator_sum = sum([
            (class_values[param]/abs(class_values[param]))*((class_values[param] * weight)**2)
            for param, weight in discipline_weights.items()
            if class_values[param] != 0  
        ])
        print(numerator_sum) 
        if numerator_sum < 0:
            numerator_sum = 0
        denominator_sum = sum([
            weight
            for param, weight in discipline_weights.items()
            if class_values[param] != 0  
        ])
        discipline = math.sqrt(numerator_sum) / math.sqrt(denominator_sum)

        professionalism_weights = {
            'understanding': 50,  
            'compliments': 50, 
            'useful_links': 15,
            'code_discussion': 20,
            'tech_questions': 40,
            'misunderstanding': 10
        }
        numerator_sum = sum([
            (class_values[param]/abs(class_values[param]))*((class_values[param] * weight)**2)
            for param, weight in professionalism_weights.items()
            if class_values[param] != 0 
        ])
        if numerator_sum < 0:
            numerator_sum = 0

        denominator_sum = sum([
            weight
            for param, weight in professionalism_weights.items()
            if class_values[param] != 0  
        ])
        professionalism = math.sqrt(numerator_sum) / math.sqrt(denominator_sum)
        return {
            'successfulness': float(str(successfulness * 100)[:5]),
            'discipline': float(str(discipline * 100)[:5]),
            'professionalism': float(str(professionalism * 100)[:5])
        }

    proxi_metrics = calculate_metrics(pattern)
    def view_predictions(classification_result):
        classification_result=classification_result.to_dict()
        data = [go.Bar(x=list(classification_result.keys()), y=list(classification_result.values()))]
        layout = go.Layout(title='Результаты классификации',
        paper_bgcolor='rgba(0,0,0,0)',  # <--- Add this line
        plot_bgcolor='rgba(0,0,0,0)', autosize=True)
        fig = go.Figure(data=data, layout=layout)
        plot_div = plotly.offline.plot(fig, include_plotlyjs=False, output_type='div')
        return plot_div
    def view_positive_predictions(classification_result):
        mp_usefull_label = ['Вопрос о профессии','Вопросе о группе','Вопросы о ходе обучения','Мнение о мероприятии','Технические вопросы','Обсуждение кода','Полезные ссылки','Комплименты']
        classification_result=classification_result.to_dict()
        classification_result = {k: v for k, v in classification_result.items() if k in mp_usefull_label}
        data = [go.Bar(x=list(classification_result.keys()), y=list(classification_result.values()))]
        layout = go.Layout(title='Результаты классификации',
        paper_bgcolor='rgba(0,0,0,0)',  # <--- Add this line
        plot_bgcolor='rgba(0,0,0,0)', autosize=True)
        fig = go.Figure(data=data, layout=layout)
        plot_div = plotly.offline.plot(fig, include_plotlyjs=False, output_type='div')
        return plot_div
    def view_negative_predictions(classification_result):
        mp_useless_label = ['Недопонимание','Флуд','Токсичные сообщения','Политика','Фишинговые ссылки','Технические проблемы']
        classification_result=classification_result.to_dict()
        classification_result = {k: v for k, v in classification_result.items() if k in mp_useless_label}
        data = [go.Bar(x=list(classification_result.keys()), y=list(classification_result.values()))]
        layout = go.Layout(title='Результаты классификации',
        paper_bgcolor='rgba(0,0,0,0)',  # <--- Add this line
        plot_bgcolor='rgba(0,0,0,0)', autosize=True)
        fig = go.Figure(data=data, layout=layout)
        plot_div = plotly.offline.plot(fig, include_plotlyjs=False, output_type='div')
        return plot_div
    
    def view_message_density(data):
        data['Дата старта урока'] = pd.to_datetime(data['Дата старта урока'])
        data['Дата сообщения'] = pd.to_datetime(data['Дата сообщения'])
        start_time = data['Дата старта урока'].min()+ pd.Timedelta(minutes=10)
        end_time = data['Дата сообщения'].max()+ pd.Timedelta(minutes=10)
        time_intervals = pd.date_range(start=start_time, end=end_time, freq='5T')
        message_counts = data.groupby(pd.cut(data['Дата сообщения'], bins=time_intervals)).size()
        trace = go.Scatter(x=time_intervals, y=message_counts, mode='lines')
        layout = go.Layout(title='Количество сообщений за каждые 5 минут',
                        xaxis=dict(title='Время'),
                        yaxis=dict(title='Количество сообщений'),
                        paper_bgcolor='rgba(0,0,0,0)',  # <--- Add this line
                        plot_bgcolor='rgba(0,0,0,0)')
        fig = go.Figure(data=[trace], layout=layout)
        plot_div = fig.to_html(full_html=False, include_plotlyjs=False)
        return plot_div
    print(classification_result)

    def toxic_flood(result, data):
        mp_useless_label = ['Флуд','Токсичные сообщения']
        text = data["Текст сообщения"]
        result = result['predictions']
        output = {'id':[], 'text':[], 'result':[]}
        for i in range(len(text)):
            if result[i] in mp_useless_label:
                output['id'].append(i)
                output['text'].append(text[i])
                output['result'].append(result[i])
            if 'nill kiggers' in text[i].lower() or '1488' in text[i].lower() or '1377' in text[i].lower():
                output['id'].append(i)
                output['text'].append(text[i])
                output['result'].append('Токсичные сообщения')
        
        return output
        # classification_result = {k: v for k, v in classification_result.items() if k in mp_useless_label}
        
        # return classification_result

    line = data['Текст сообщения'].to_list()
    if line:
        return render_template('details.html', line=line, view_predictions=view_predictions(classification_result) , view_message_density=view_message_density(data), view_negative_predictions= view_negative_predictions(classification_result) , view_positive_predictions= view_positive_predictions(classification_result), proxi_metrics= proxi_metrics, toxic_flood = toxic_flood(result,data))
    else:
        return 'Строка с таким ID не найдена', 404

if __name__ == '__main__':
    app.run(debug=True)
