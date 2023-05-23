from datasets import Dataset
from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel, SetFitTrainer
import pandas as pd
import os
import re
import joblib
import logging as log
import pprint

class Ml:
    data = {}
    labels_dict = {}
    labels_dict_inv = {}
    global pp

    def __init__(self, base_model=None):
        if base_model:
            self.model = SetFitModel.from_pretrained(base_model)
        self.trainer = None

    # Добавление образцов текстов
    def add_samples(self, items, labels):
        if not isinstance(labels, list):
            labels = [labels]

        for label in labels:
            if label not in self.labels_dict:
                new_id = len(self.labels_dict) + 1
                self.labels_dict[label] = new_id
                self.labels_dict_inv[new_id] = label

            label_id = self.labels_dict[label]

            if label not in self.data:
                self.data[label_id] = []

        for item in items:
            self.data[label_id].append(item)

    # Загрузка образцов текстов
    def load_samples(self, path_samples):
        sample_files = []

        for path in os.listdir(path_samples):
            if os.path.isfile(os.path.join(path_samples, path)):
                if m := re.search('sample_([^.]+)\.txt', path):
                    sample_files.append({'name': path, 'labels': m[1]})

        for file in sample_files:
            file_name = './samples/' + file['name']
            with open(file_name) as f:
                lines = f.read().splitlines()
                labels = file['labels']

                # Ищем название категории в первой строке файла в формате # ИМЯ, если его нет, оставляем исходное
                if m := re.search('^#\s*(\w+)', lines[0]):
                    lines.pop(0)
                    labels = m[1]

                self.add_samples(lines, labels=labels)
                log.info('Loaded samples from {file} ({labels})'.format(labels=labels, file=file_name))

    # Обучение и сохранение модели
    def train(self):
        data_train = []

        for label_id in dict.keys(self.data):
            for item in self.data[label_id]:
                data_train.append({'text': item, 'label': label_id})

        df = pd.DataFrame(data_train, columns=['text', 'label'])
        dataset = Dataset.from_pandas(df)

        self.trainer = SetFitTrainer(
            model=self.model,
            train_dataset=dataset,
            eval_dataset=dataset,
            loss_class=CosineSimilarityLoss,
            batch_size=16,
            num_iterations=20, # Number of text pairs to generate for contrastive learning
            num_epochs=1,  # Number of epochs to use for contrastive learning
            column_mapping={"text": "text", "label": "label"}
        )

        self.trainer.train()
        metrics = self.trainer.evaluate()
        log.info(metrics)

        self.trainer.labels_dict = self.labels_dict
        self.trainer.labels_dict_inv = self.labels_dict_inv

    # Загрузка модели
    def load(self, trained_model_name):
        self.trained_model_name = trained_model_name
        self.trainer = joblib.load(self.trained_model_name)

        labels_list = 'None'

        if hasattr(self.trainer, 'labels_dict'):
            self.labels_dict = self.trainer.labels_dict
            labels_list = ', '.join(dict.keys(self.labels_dict))

        if hasattr(self.trainer, 'labels_dict_inv'):
            self.labels_dict_inv = self.trainer.labels_dict_inv

        log.info('Loaded model {model_name} with labels: {labels_list}'.format(model_name=trained_model_name, labels_list=labels_list))

    # Сохранение модели
    def save(self, trained_model_name=False):
        file_name = trained_model_name if trained_model_name else self.trained_model_name
        self.model = self.model.to('cpu')
        joblib.dump(self.trainer, file_name)

    # Получение категорий
    def cats(self, items):
        if not isinstance(items, list):
            items = [items]

        #t = self.trainer.model.predict_proba(items, as_numpy=True)
        #print(t)

        results = self.trainer.model(items)

        cats = []

        for result in results:
            label_id = int(result)

            if label_id in self.labels_dict_inv:
                cats.append(self.labels_dict_inv[label_id])
            else:
                cats.append('Unknown')

        return cats
