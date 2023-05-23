import json
import os.path
import glob
import pprint
import re
import datetime
import logging as log
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.feature_extraction import text
import nltk
import pymorphy2

class Clusterer:
    stop_words = ['и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 'то', 'все', 'она', 'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же', 'вы', 'за', 'бы', 'по', 'только', 'ее', 'мне', 'было', 'вот', 'от', 'меня', 'еще', 'нет', 'о', 'из', 'ему', 'теперь', 'когда', 'даже', 'ну', 'вдруг', 'ли', 'если', 'уже', 'или', 'ни', 'быть', 'был', 'него', 'до', 'вас', 'нибудь', 'опять', 'уж', 'вам', 'ведь', 'там', 'потом', 'себя', 'ничего', 'ей', 'может', 'они', 'тут', 'где', 'есть', 'надо', 'ней', 'для', 'мы', 'тебя', 'их', 'чем', 'была', 'сам', 'чтоб', 'без', 'будто', 'чего', 'раз', 'тоже', 'себе', 'под', 'будет', 'ж', 'тогда', 'кто', 'этот', 'того', 'потому', 'этого', 'какой', 'совсем', 'ним', 'здесь', 'этом', 'один', 'почти', 'мой', 'тем', 'чтобы', 'нее', 'сейчас', 'были', 'куда', 'зачем', 'всех', 'никогда', 'можно', 'при', 'наконец', 'два', 'об', 'другой', 'хоть', 'после', 'над', 'больше', 'тот', 'через', 'эти', 'нас', 'про', 'всего', 'них', 'какая', 'много', 'разве', 'три', 'эту', 'моя', 'впрочем', 'хорошо', 'свою', 'этой', 'перед', 'иногда', 'лучше', 'чуть', 'том', 'нельзя', 'такой', 'им', 'более', 'всегда', 'конечно', 'всю', 'между', 'пожалуйста', 'пожалйста', 'пжлст', 'пож', 'пжл', 'плиз', 'плз', 'спасибо', 'сорри', 'привет', 'здравствуйте', 'дратути', 'добрый', 'день', 'добрый', 'вечер', 'доброе', 'утро', 'это', 'всем', 'можете', 'очень', 'всё', 'маркет', 'market', 'сегодня', 'вчера', 'заранее', 'благодарю', 'спасибо', 'парни', 'друзья', 'коллеги', 'товарищи', 'ребята', 'ребят', 'блин', 'короче', 'кажется', 'видимо', 'вроде', 'наверное', 'мочь', 'хотеть', 'посмотреть', 'сюда', 'нужно', 'коллега', 'весь', 'почему']
    stop_words_dict = dict.fromkeys(stop_words, True)
    freq_limit = 86400 / 2
    min_message_length = 40
    max_message_length = 5000

    def __init__(self):
        self.messages = []
        self.morph = pymorphy2.MorphAnalyzer(lang='ru')

    def load_messages(self, file_name, date_limit=False, messages_count_limit=0, options={}):
        self.messages = []
        member_time = {}

        min_message_length = options['min_message_length'] if 'min_message_length' in options else self.min_message_length
        max_message_length = options['max_message_length'] if 'max_message_length' in options else self.max_message_length

        if date_limit:
            time_start = int(datetime.datetime.strptime(date_limit, '%d-%m-%Y').strftime("%s"))
        else:
            time_start = 0

        data = json.load(open(file_name))
        messages_count = 0

        for m in data['messages']:
            t = int(m['date_unixtime'])

            if t < time_start or 'reply_to_message_id' in m:
                continue

            if 'from_id' in m and t:
                #if m['from_id'] in locals_dict:
                #    continue

                if m['from_id'] not in member_time or t - member_time[m['from_id']] > self.freq_limit:
                    msg = ''
                    msg_full = ''
                    links = []

                    for item in m['text_entities']:
                        if item['type'] == 'plain':
                            msg += item['text'] + ' '
                            msg_full += item['text'] + ' '
                        elif item['type'] == 'link':
                            links.append(item['text'])
                            if 'url_placeholders' in options:
                                msg += 'гиперссылка' + ' '

                            if 'url_placeholders_full' in options:
                                msg_full += 'гиперссылка' + ' '
                            else:
                                msg_full += item['text'] + ' '

                    msg = re.sub("\s{1,}", " ", msg, re.S)
                    msg = msg.strip("\n ")
                    msg_full = re.sub("\s{1,}", " ", msg_full, re.S)
                    msg_full = msg_full.strip("\n ")

                    if len(msg) > min_message_length and len(msg) < max_message_length:
                        #msg = re.sub(" {1,}", " ", msg)
                        self.messages.append({'msg': msg, 'msg_full': msg_full, 'links': links})

                        messages_count += 1
                        if messages_count_limit and messages_count >= messages_count_limit:
                            break

                member_time[m['from_id']] = t

        log.info('Messages loaded: %s' % messages_count)

    def preprocess(self, text):
        text = re.sub("[.,\-!?:()\[\]]+", " ", text)
        text = re.sub("\n", ' ', text)

        result = ''

        for w in text.split():
            w = self.morph.parse(w.lower())[0].normal_form

            if w in self.stop_words_dict:
                continue

            if result: result += ' '
            result += w

        return result

    def cluster(self, cluster_count, max_samples=50, max_terms=10, out_folder=False, clean_folder=False):
        vectorizer = TfidfVectorizer()
        processed_messages = []

        log.info('Clustering messages to {out_folder}'.format(out_folder=out_folder))

        for m in self.messages:
            if type(m['msg']) is str:
                processed_messages.append(self.preprocess(m['msg']))

        x = vectorizer.fit_transform(processed_messages)

        model = KMeans(n_clusters=cluster_count, init='k-means++', max_iter=3000, n_init=50)
        model.fit(x)

        clustered = {}

        for m in self.messages:
            if type(m['msg']) is str:
                y = vectorizer.transform([self.preprocess(m['msg'])])
                p = model.predict(y)[0]

                if p not in clustered:
                    clustered[p] = [m['msg_full']]
                else:
                    clustered[p].append(m['msg_full'])

        order_centroids = model.cluster_centers_.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names_out()

        # Подготовка директории (создать если нет, удалить все файлы, если есть)
        if out_folder:
            if not os.path.exists(out_folder):
                os.mkdir(out_folder)
            elif clean_folder:
                files = glob.glob(out_folder + '/*')
                for f in files:
                    os.remove(f)

        for i in range(0, cluster_count):
            if out_folder:
                file_name = '{folder}/sample_cluster_{id}.txt'.format(folder=out_folder, id=i)
                fh = open(file_name, 'w')

            log.info("**Кластер {n} ({l})**".format(n=i, l=len(clustered[i])))

            w = ''
            for c in order_centroids[i, :max_terms]:
                w += terms[c] + ' '

            if out_folder:
                fh.write('# %s\n' % w)

            log.info('Cluster top terms: %s' % w)

            for m in clustered[i][0:max_samples]:
                sample_line = '%s' % m[0].upper() + m[1:]
                log.info(sample_line)

                if out_folder:
                    fh.write(sample_line + "\n")

            log.info('')
            log.info('')

            fh.close()