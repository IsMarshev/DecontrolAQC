FROM pytorch/pytorch

#Создаем папку хранения кода
RUN mkdir /usr/src/decontrol_aqc

#Копируем папки с кодом в хранилище
COPY ./src /usr/src/decontrol_aqc/src
COPY ./requirement.txt /usr/src/decontrol_aqc/requirements.txt

#Установка рабочего каталога
WORKDIR /usr/src/decontrol_aqc

#Устновка зависимостей 
RUN python -m pip install -r /usr/src/decontrol_aqc/requirements.txt

#Привяжем порт
EXPOSE 5000

#Команда запуска приложения внутри контейнера
CMD ["python", "/usr/src/decontrol_aqc/src/app.py", "0.0.0.0:5000"]