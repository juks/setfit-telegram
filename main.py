from lib.ml import Ml
from lib.my_responder import MyResponder
from lib.clusterer import Clusterer
import argparse
import logging as log
import re
from simple_http_server import route, server
from simple_http_server import JSONBody
from simple_http_server import HttpError
import pprint

parser = argparse.ArgumentParser(description="Just an example",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-m",  "--mode",                    help="operation mode", choices=['train', 'history', 'cluster', 'serve', 'test'], required=True)
parser.add_argument('-bm', '--base_model',              help='base model to use', default='RussianNLP/ruRoBERTa-large-rucola')
parser.add_argument('-tf', '--telegram_file',           help='telegram messages dump file to use')
parser.add_argument('-dl', '--date_limit',              help='date limit for messages to cluster')
parser.add_argument('-nc', '--num_clusters',            help='number of clusters', default=20)
parser.add_argument('-o',  '--output_folder',           help='number of clusters', default='./samples_clustered')
parser.add_argument('-s',  '--samples_folder',          help='number of clusters', default='./samples')
parser.add_argument('-cf', '--clean_samples_folder',    help='clean samples folder', default=True)
parser.add_argument('-f',  '--model_file',              help='local model file')
parser.add_argument('-t',  '--test_file',               help='file with test examples'),
parser.add_argument('-v',  '--verbose',                 help='verbose mode', action='store_true'),
parser.add_argument('-p',  '--listen_port',             help='http port to listen on', default=8080)
parser.add_argument('-ml', '--max_message_len',         help='maximum message len', default=1000)

args = parser.parse_args()
config = vars(args)

if config['mode'] in ['train', 'test', 'serve', 'history'] and config['model_file'] == None:
    parser.error('The --model_file argument is required')

if config['mode'] in ['test'] and config['test_file'] == None:
    parser.error('The --test_file argument is required')

if config['mode'] == 'history' and config['telegram_file'] == None:
    parser.error('The --telegram_file argument is required')

config = vars(args)

# Включаем журналирование
if config['verbose']:
    log.basicConfig(format="%(levelname)s: %(message)s", level=log.DEBUG)
    log.info("Verbose output.")
else:
    log.disable(log.CRITICAL)

# Обучение
if config['mode'] == 'train':
    ml = Ml(base_model=config['base_model'])
    ml.load_samples(config['samples_folder'])

    ml.train()
    log.info("Training complete, saving model to %s" % config['model_file'])
    ml.save(config['model_file'])
# Тест на наборе предложение из файла
elif config['mode'] == 'test' and config['test_file']:
    ml = Ml()
    ml.load(config['model_file'])

    col_ok = '\033[92m'
    col_fail = '\033[91m'
    col_end = '\033[0m'

    correct = 0

    f = open(config['test_file'])
    test_lines = f.read().splitlines()

    for item in test_lines:
        parts = item.split(' ', 1)
        cat = ml.cats(parts[1])

        if cat[0] == parts[0]:
            col = col_ok
            correct += 1
        else:
            col = col_fail

        print(f'[{col}{cat[0]}{col_end}] {parts[1]}')

    p = round(correct * 100 / len(test_lines), 2) if correct else 0
    print("\nTotal: {total} Correct: {correct} ({p}%)".format(total=len(test_lines), correct=correct, p=p))
# Тест на истории из dump-файла telegram
elif config['mode'] == 'history':
    ml = Ml()
    ml.load(config['model_file'])

    resp = MyResponder()
    clusterer = Clusterer()

    reply_count = 0
    total_count = 0

    clusterer.load_messages(config['telegram_file'], date_limit=config['date_limit'], options={'max_message_length': config['max_message_len']})

    for m in clusterer.messages:
        cats = ml.cats(m['msg'])
        cat_name = cats[0]
        response = resp.respond(cat_name, m['msg_full'], m['links'])

        total_count += 1

        if response:
            print('M: ' + m['msg_full'])
            print('R: ' + response)
            print()

            reply_count += 1

    ratio = round((reply_count / total_count) * 100, 2) if total_count else 0
    print("Total messages: {total}. Replies: {reply_count}. Ratio: {ratio} %".format(total=total_count, reply_count=reply_count, ratio=ratio))
# Кластеризация набора сообщений с сохранением набора образцов в директорию
elif config['mode'] == 'cluster':
    clusterer = Clusterer()

    clusterer.load_messages(config['telegram_file'], config['date_limit'], options={'url_placeholders_full': True})
    clusterer.cluster(cluster_count=config['num_clusters'],
                      out_folder=config['output_folder'],
                      clean_folder=config['clean_samples_folder'])
# Классификация в режиме http-сервера
elif config['mode'] == 'serve':
    ml = Ml()
    ml.load(config['model_file'])

    # Определить категорию для набора текстов
    @route("/cats", method=["POST"])
    def index(data: JSONBody):
        if 'text' in data and len(data['text']):
            cats = ml.cats(data['text'])
            return {'category': cats}
        else:
            raise HttpError(400, "Missing JSON body parameter: text")

    # Ответить на текстовое сообщение
    @route("/reply", method=["POST"])
    def index(data: JSONBody):
        resp = MyResponder()

        if 'text' in data and len(data['text']):
            cats = ml.cats(data['text'])
            if m := re.search("(https?://[^\s]+)", data['text']):
                urls = m[1]
                text = re.sub("(https?://[^\s]+)", '', data['text'])
            else:
                urls = []
                text = data['text']

            response = resp.respond(cats[0], text, urls)
            return {'response': response}
        else:
            raise HttpError(400, "Missing JSON body parameter: text")

    server.start(port=config['listen_port'])