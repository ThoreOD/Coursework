import contractions
import emot
import numpy as np
import pandas as pd
import re
import torch

from functools import reduce
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from textblob import TextBlob
from transformers import pipeline
from tqdm import tqdm
from unidecode import unidecode

def compute_metrics(p):
    pred, labels = p
    return {"balanced_accuracy": balanced_accuracy_score(y_true = labels, y_pred = np.argmax(pred, axis = 1))}

class dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels = None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

class sentiment():
    def __init__(self, csv, title, text, score):
        self.df = pd.read_csv(filepath_or_buffer = csv)
        self.title = self.df[title]
        self.text = self.df[text]
        self.score = self.df[score].astype(int)
        self.descriptions = ["titles", "texts"]
    
    def del_emoticon(self, text_input):
        return [re.compile("|".join(map(re.escape, list(emot.EMOTICONS_EMO.keys())))).sub("", text)
                for text in text_input]
        
    def fix_symbols(self, text_input):
        patterns = [(r"\n", ""), (r'‘(.*?)’', r'"\g<1>"'), (r"`", r"'"), (r'„', r'"')]
        return [unidecode(reduce(lambda x, y: re.sub(y[0], y[1], x), patterns, text))
                for text in text_input]
    
    def score_mapping(self, sentiment):
        return 1 + ((5 - 1) / (1 - (-1))) * (sentiment - (-1))

    def grad_sen(self, identifier, title_sens, title_weight, text_sens):
        score_input = np.round((title_sens * title_weight) + (text_sens * (1 - title_weight)))
        return score_input, {"model": identifier,
                             "accuracy": accuracy_score(y_true = self.score, y_pred = score_input),
                             "balanced_accuracy": balanced_accuracy_score(y_true = self.score, y_pred = score_input)}
    
    def txtblb_sen(self, identifier, title_weight = 0.5):
        txtblb_sens = np.array([
            [
                self.score_mapping(TextBlob(text_input).sentiment.polarity)
                for text_input in tqdm(text_type, desc = f"txtblb_sens_{description}", unit = "texts")
            ]
            for text_type, description in zip([self.title, self.text], self.descriptions)
        ])
        return self.grad_sen(identifier, txtblb_sens[0], title_weight, txtblb_sens[1])

    def bert_prep(self, text_input):
        return [
            contractions.fix(text) 
            for text in self.fix_symbols(self.del_emoticon(text_input))
            ]

    def bert_sen(self, identifier, model, title_weight = 0.5):
        pipe = pipeline(task = "text-classification", 
                        model = model)
        bert_sens = np.array([
            [
                int(pipe(text_input, padding = True, truncation = True, max_length = 512)[0]["label"][0]) 
                for text_input in tqdm(self.bert_prep(text_type), desc = f"bert_sens_{description}", unit = "texts")
            ]
            for text_type, description in zip([self.title, self.text], self.descriptions)
        ])
        return self.grad_sen(identifier, bert_sens[0], title_weight, bert_sens[1])