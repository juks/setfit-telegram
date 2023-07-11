import unicodedata
import re


def strip_accents(text):
    result = ''
    enumerated = unicodedata.normalize('NFD', text)

    for i, c in enumerate(enumerated):
        if unicodedata.category(c) == 'Mn':
            continue

        if (i < len(enumerated) - 1):
            nxt = enumerated[i + 1]
        else:
            nxt = ''

        if c.lower() in ['и', 'е'] and nxt != '' and unicodedata.category(nxt) == 'Mn':
            if c == 'и':
                c = 'й'
            elif c == 'И':
                c = 'Й'
            elif c == 'е':
                c = 'ё'
            elif c == 'Е':
                c = 'Ё'

        result += c

    return result


def clean(text):
    text = re.sub('https?://[^ ]+', '', text)
    text = re.sub('[a-z0-9_.\-]@[a-z0-9\.\-_]+', '', text)
    text = re.sub('[^Ёёа-яa-z0-9. ,\?!\-\(\)/:]', '', text, flags=re.IGNORECASE)
    text = re.sub('\s([?!\.,])', '\\1', text)
    text = re.sub('([?!\.,])([а-яa-z])', '\\1 \\2', text, flags=re.IGNORECASE)
    text = re.sub('!{2,}', '!', text)
    text = re.sub('\s{2,}', '', text)
    text = re.sub('^(добрый день|день добрый|добрый вечер|вечер добрый|доброе утро|привет|приветствую|здравствуйте|доброй ночи)[,.?! ]*', '', text,
                  flags=re.IGNORECASE)
    text = strip_accents(text).lower()

    return text


