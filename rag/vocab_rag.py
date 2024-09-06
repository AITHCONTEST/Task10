import copy
import random
import numpy as np
import pandas as pd

from pymystem3 import Mystem
from tqdm.notebook import tqdm
from uralicNLP import uralicApi
from collections import Counter
from typing import Union, List

system = Mystem()


def create_vocab():
    vocab_1 = pd.read_csv("DICTIONARY_MANS_RUS_new.csv")
    vocab_2 = pd.read_csv("DICTIONARY_MANS_RUS_2.csv")
    vocab_2 = vocab_2.dropna(subset=["rus"])
    rus_translate = vocab_2.rus.tolist()
    mans_translate = vocab_2.mans.tolist()
    note_translate = vocab_2.note.tolist()
    for i in range(len(rus_translate)):
        if "гл. прист." in rus_translate[i]:
            # print(i)
            rus_translate[i] = rus_translate[i].replace("гл. прист.", note_translate[i])
            note_translate[i] = ""
        if "гл. приставка" in rus_translate[i]:
            rus_translate[i] = rus_translate[i].replace(
                "гл. приставка", note_translate[i]
            )
            note_translate[i] = ""
        if "гл.прист." in rus_translate[i]:
            # print(i)
            rus_translate[i] = rus_translate[i].replace("гл.прист.", note_translate[i])
            note_translate[i] = ""
    for i in range(len(rus_translate)):
        for n in ["III.", "II.", "I.", "I", "II", "III"]:
            # print(i)
            if n in rus_translate[i]:
                # print(i)
                rus_translate[i] = rus_translate[i].replace(n, "")
            if not isinstance(note_translate[i], float):
                # print(note_translate[i])
                if n in note_translate[i]:
                    note_translate[i] = note_translate[i].replace(n, "")
        rus_translate[i] = rus_translate[i].strip()
        if not isinstance(note_translate[i], float):
            note_translate[i] = note_translate[i].strip()
    bad = [
        "4)",
        ") 2)",
        "),2)",
        ") ,2)",
        ") 3)",
        ");2)",
        "); 2)",
        "2)",
        "),1)",
        ");",
        ")",
    ]
    for i in range(len(note_translate)):
        if isinstance(note_translate[i], float):
            continue
        clean_last = False
        for n in bad:
            if n in note_translate[i]:
                # print(note_translate[i])
                note_translate[i] = note_translate[i].replace(n, ",")
                clean_last = True
        if clean_last:
            # print(note_translate[i])
            res = note_translate[i].split(",")[:-1]
            note_translate[i] = []
            for elem in res:
                if elem != " " or elem != "":
                    note_translate[i].append(elem)
            note_translate[i] = (", ".join(note_translate[i])).strip()
            if note_translate[i][-1] == ",":
                note_translate[i] = note_translate[i][:-1]

    vocab_2.rus = rus_translate
    vocab_2.mans = mans_translate
    vocab_2.note = note_translate

    rus_translate = vocab_1.rus.tolist()
    mans_translate = vocab_1.mans.tolist()
    note_translate = vocab_1.note.tolist()

    rus_translate[47] = "один"
    note_translate[47] = "акв хум - один мужчина"

    rus_translate[903] = "ещё"
    note_translate[903] = "неа̄сюм иӈыт ёхты - отец ещё не приехал"

    vocab_1.rus = rus_translate
    vocab_1.mans = mans_translate
    vocab_1.note = note_translate

    vocab = pd.concat([vocab_1, vocab_2])

    rus_translate = vocab.rus.tolist()
    mans_translate = vocab.mans.tolist()
    note_translate = vocab.note.tolist()

    words = {}
    for i in range(len(rus_translate)):
        s = f"{mans_translate[i]}@@@{rus_translate[i]}"
        if s not in words:
            words[s] = []
        if isinstance(note_translate[i], float):
            continue
        words[s].append(note_translate[i])

    rus_translate = []
    mans_translate = []
    note_translate = []
    for key in words:
        m, r = key.split("@@@")
        mans_translate.append(m)
        rus_translate.append(r)
        if len(words[key]) > 0:
            note_translate.append(", ".join(words[key]))
        else:
            note_translate.append(np.nan)
    data = {"mans": mans_translate, "rus": rus_translate, "note": note_translate}

    vocab = pd.DataFrame(data)
    vocab_mans_dict = {}
    vocab_rus_dict = {}
    full_rus_dict = {}
    ids = vocab.index
    mans = vocab.mans.tolist()
    rus = vocab.rus.tolist()
    note = vocab.note.tolist()

    for i in range(len(ids)):
        rus_word = rus[i]
        mans_word = mans[i]
        if note[i] is not np.nan:
            rus_word += f" ({note[i]})"
        if mans_word not in vocab_mans_dict:
            vocab_mans_dict[mans_word] = []
        if rus_word not in vocab_rus_dict:
            vocab_rus_dict[rus_word] = []

        vocab_mans_dict[mans_word].append(rus_word)
        vocab_rus_dict[rus_word].append(mans_word)

    for i in range(len(ids)):
        rus_word = rus[i]
        words = rus_word.split()
        if note[i] is not np.nan:
            rus_word += f" ({note[i]})"
        for word in words:
            if word in [
                "с",
                "но",
                "и",
                "а",
                "что",
                "в",
                "на",
                "от",
                "не",
                "то",
                "это",
                "тот",
                "же",
                "это",
                "к",
                "у",
                "мы",
                "я",
                "он",
                "оно",
                "ты",
                "вы",
                "она",
                "они",
                "нет",
                "нас",
                "быть",
                "о",
                "том",
                "как",
            ]:
                continue
            if word not in full_rus_dict:
                full_rus_dict[word] = []
            add_word = True
            for item in full_rus_dict[word]:
                if item[0] == rus_word:
                    add_word = False
            if add_word:
                full_rus_dict[word].append([rus_word, vocab_rus_dict[rus_word], i])

    return vocab_mans_dict, full_rus_dict


def replace_symbols(text, f=True):
    for symbol in "!@#$%,.1234567890/?'\"\\«»_:;_…()”":
        text = text.replace(symbol, " ")
    first_words = set((" ".join(text.lower().split())).split())

    for symbol in "!@#$%,.1234567890-/?'\"\\«»_-:;_…–()”":
        text = text.replace(symbol, " ")
    second_words = set((" ".join(text.lower().split())).split())
    return list(first_words | second_words)


def lemmatize_mans_sentence(sentence):
    sentence = replace_symbols(sentence)
    words = []
    for word in sentence:
        norm_form = uralicApi.lemmatize(word, "mns")
        if len(norm_form) == 0:
            words.append(word)
        else:
            words.extend(sorted(uralicApi.lemmatize(word, "mns")))
    return list(set(words) | set(sentence))


def lemmatize_rus_sentence(sentence):
    lemmas = system.lemmatize(sentence)
    lemmas_clean = []
    for word in lemmas:
        if word in "!@#$%,.1234567890-/?'\"\\«»_-:;_…–()”":
            continue
        lemmas_clean.append(word)
    sentence = replace_symbols(sentence)
    return list(set(" ".join(lemmas_clean).split()) | set(sentence))


mans_vocab, rus_vocab = create_vocab()


def get_rus_translate(words: Union[List[str], str], vocab=mans_vocab, sep=" | "):
    if isinstance(words, str):
        words = lemmatize_mans_sentence(words)
    translate = []
    for word in words:
        if word in vocab:
            translate.append(word + " - " + ", ".join(vocab[word]))
    return sep.join(translate)


def get_mans_translate(words: Union[List[str], str], vocab=rus_vocab, sep=" | "):
    if isinstance(words, str):
        words = lemmatize_rus_sentence(words)
    translate = []
    used = set()
    for word in words:
        if word in vocab:
            for example in vocab[word]:
                if example[2] not in used:
                    translate.append(example[0] + " - " + ", ".join(example[1]))
                    used.add(example[2])
    return sep.join(translate)
