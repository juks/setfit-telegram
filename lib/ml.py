import collections

from datasets import Dataset
from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel, SetFitTrainer
import pandas as pd
import statistics
import os
import re
import joblib
import logging as log


class Ml:
    data = {'train': {}, 'test': {}}
    labels_dict = {}
    labels_dict_inv = {}
    global pp

    def __init__(self, base_model=None):
        if base_model:
            self.model = SetFitModel.from_pretrained(base_model)
        self.trainer = None

    # Добавление образцов текстов
    def add_samples(self, items, labels, mode='train'):
        if not isinstance(labels, list):
            labels = [labels]

        for label in labels:
            # При обучении храним словарь целочисленных идентификаторов, для теста используем строки
            if mode == 'train':
                if label not in self.labels_dict:
                        new_id = len(self.labels_dict) + 1

                self.labels_dict[label] = new_id
                self.labels_dict_inv[new_id] = label

                label_id = self.labels_dict[label]
            else:
                label_id = label

            if label not in self.data[mode]:
                self.data[mode][label_id] = []

            for item in items:
                self.data[mode][label_id].append(item)

    # Загрузка образцов текстов
    def load_samples(self, source_path, mode='train'):
        source_files = []

        pattern = 'sample_([^.]+)\.txt' if mode == 'train' else 'test_([^.]+)\.txt'

        for path in os.listdir(source_path):
            if os.path.isfile(os.path.join(source_path, path)):
                if m := re.search(pattern, path):
                    source_files.append({'name': path, 'labels': m[1]})

        for file in source_files:
            file_name = './samples/' + file['name']
            with open(file_name) as f:
                lines = f.read().splitlines()
                labels = file['labels']

                # Ищем название категории в первой строке файла в формате # ИМЯ, если его нет, оставляем исходное
                if m := re.search('^#\s*(\w+)', lines[0]):
                    lines.pop(0)
                    labels = m[1]

                self.add_samples(lines, labels=labels, mode=mode)
                log.info('Loaded {mode} samples from {file} ({labels})'.format(labels=labels, file=file_name, mode=mode))

    # Обучение и сохранение модели
    def train(self):
        data_train = []

        for label_id in dict.keys(self.data['train']):
            for item in self.data['train'][label_id]:
                data_train.append({'text': item, 'label': label_id})

        df = pd.DataFrame(data_train, columns=['text', 'label'])
        dataset = Dataset.from_pandas(df)

        self.trainer = SetFitTrainer(
            model=self.model,
            train_dataset=dataset,
            eval_dataset=dataset,
            loss_class=CosineSimilarityLoss,
            batch_size=14,
            num_iterations=20, # Number of text pairs to generate for contrastive learning
            num_epochs=1,  # Number of epochs to use for contrastive learning
            column_mapping={"text": "text", "label": "label"}
        )

        self.trainer.train()
        metrics = self.trainer.evaluate()
        log.info(metrics)

        self.trainer.labels_dict = self.labels_dict
        self.trainer.labels_dict_inv = self.labels_dict_inv
    # Тест модели
    def test(self,  prob_limit=None, default_label_name=None):
        col_ok = '\033[92m'
        col_fail = '\033[91m'
        col_end = '\033[0m'

        total_correct = 0
        total = 0
        summary = {}
        confidence_stat = collections.defaultdict(list)

        for label in self.data['test']:
            correct = 0

            for item in self.data['test'][label]:
                cats = self.cats(item, prob_limit, default_label_name)
                confidence_stat[label].append(cats[0]['p'])

                if cats[0]['label'] == label:
                    col = col_ok
                    correct += 1
                    total_correct += 1
                else:
                    col = col_fail
                print(item)
                print(f"[{col}{cats[0]['label']}{col_end}] {label}")

                total += 1

            p = round(correct * 100 / len(self.data['test'][label]), 2) if correct else 0
            print(f"\nTag: {label}:")
            print("Total: {total} Correct: {correct} ({p}%) Confidence: {conf}\n".format(total=len(self.data['test'][label]), correct=correct, p=p, conf=round(statistics.mean(confidence_stat[label]), 4)))

            summary[label] = {'total': len(self.data['test'][label]), 'correct': correct, 'p': p}

        for line in summary:
            print("{label}: {correct}/{total} ({p}%) Confidence: {conf}".format(label=line, correct=summary[line]['correct'], total=summary[line]['total'], p=summary[line]['p'], conf=round(statistics.mean(confidence_stat[line]), 4)))

        p = round(total_correct * 100 / total, 2) if total_correct else 0
        print("\nAll total: {total} Correct: {correct} ({p}%)".format(total=total, correct=total_correct, p=p))
    # Тест на истории из dump-файла telegram

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

    # Получение названия тега по идентификатору
    def get_tag_name(self, label_id):
        if label_id in self.labels_dict_inv:
            return self.labels_dict_inv[label_id]
        else:
            return False

    # Получение идентификатора тега по названию
    def get_tag_id(self, label_name):
        if label_name in self.labels_dict:
            return self.labels_dict[label_name]
        else:
            return None

    # Получение категорий
    def cats(self, items, prob_limit=None, default_label_name=None):
        if not isinstance(items, list):
            items = [items]

        t = self.trainer.model.predict_proba(items)
        probs, results = t.max(dim=1)

        cats = []

        for i, result in enumerate(results):
            label_id = int(result) + 1

            if prob_limit and probs[i] < prob_limit:
                label_id = self.get_tag_id(default_label_name)

            if label_id in self.labels_dict_inv:
                cats.append({'label': self.labels_dict_inv[label_id], 'p': round(float(probs[i]), 5)})
            else:
                cats.append('Unknown')

        return cats
