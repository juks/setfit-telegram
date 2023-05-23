# Что это такое?
![Картинка](https://user-images.githubusercontent.com/147685/240291033-f5a1f20c-f7f2-4b79-bcc8-7c603273ff4d.png)
Это микро-фреймворк, позволяющий предельно легко обучить модель, способную классифицировать образцы текста. С помощью этой модели можно реализовать элементарную логику ответов на известные типы обращений в чате или в трекере задач.

Возможности:
* Получение образцов текста для разметки из dump-файла telegram методом кластеризации.
* Few-shot обучение модели с использованием библиотеки Setfit.
* Тест модели на образцах из текстового файла.
* Тест модели на dump-файле телеграм-чата.
* Веб сервер поддерживающий классификацию передаваемых строк или ответ на них.

# От состояния «я полный ноль» до работающей модели

#### 1. Установка необходимого:
    
    pip install pipenv 
    pipenv install

#### 2. Кластеризация из dump-файла в формате json в набор образцов (результаты в ./samples_clustered)
    pipenv run python main.py -m cluster -tf test_dump.json -v

#### 3. Подготовка нужных категорий в директорию ./samles (удаление мусорных строк, удаление, объединение файлов).

#### 4. Обучение модели:
    pipenv run python main.py -m train -s ./samples -f model.joblib -v

#### 5. Тест на файле с примерами:
    pipenv run python main.py -m test -t test.txt -f model.joblib -v

#### 6. Тест на дампе телеграма:
    pipenv run python main.py -m history -f model.joblib -dl 01-01-2023 -tf test_dump.json

#### 7. Запуск-веб сервера:
    pipenv run python main.py -m serve -f model.joblib -v

#### 8. Проверка веб-сервера:
    curl -X POST "http://localhost:8080/cats" -H "accept: application/json" -H "Content-Type: application/json" -d "{\"text\": [\"курьер не может запустить приложение\", \"приложение зависло\", \"смена не открывается\", \"какая-то ерунда\"]}"

Результат:
`{"category": ["courier_app_problem", "courier_app_problem", "open_shift", "other"]}`
