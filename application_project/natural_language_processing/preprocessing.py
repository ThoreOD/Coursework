import contractions
import os
import re
import spacy
import string

from collections import namedtuple
from functools import reduce
from itertools import chain
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
from thefuzz import fuzz, process, utils
from unidecode import unidecode

class prep():
    def __init__(self, df, included_punctuation = None):
        self.df = df.reset_index()
        symbols = set(string.punctuation).union({"`"})
        self.excluded_punctuations = list(symbols - set(included_punctuation)) if included_punctuation is not None else list(symbols)

    def elongation_contraction(self, doc, add = []):
        patterns = [(r"(\w)\1+", r"\1\1"), (r"(\w)(-\1)+", r"\1"), (r"(\b\w+\b)(-\1)+", r"\1"), (r"(\w+)in'", r"\1ing"), (r"'til", "until")] + add
        return reduce(lambda doc, pattern: re.sub(pattern[0], pattern[1], doc), patterns, contractions.fix(doc.lower()))   
    
    def lemmatization(self, doc, nlp):
        return " ".join([word.lemma_ for word in nlp(doc)])
    
    def punctuation(self, doc):
        return [word for word in WordPunctTokenizer().tokenize((doc.translate(str.maketrans({"&": "and", "$": "dollars", "%": "percent"})))) 
                if word not in self.excluded_punctuations + [punctuation * 2 for punctuation in self.excluded_punctuations]]
    
    def lwe_corpus(self, text, tags):
        self.lwe_train_corpus = [
            namedtuple("lyric_doc", ["words"] + ["tags"])(
                self.punctuation(unidecode(self.elongation_contraction(lyric))),
                [f"{self.df[tags[0]][i]}_{self.df[tags[1]][i]}"]
            )
            for i, lyric in enumerate(self.df[text])
        ]
        self.vocabular = len(set([word for doc in self.lwe_train_corpus for word in doc.words]))
        return
    
    def words_per_sentence(self, text):
        sentences = [sentence for sentence in re.sub(r"\n\n|\n|!|\?", ". ", ". ".join(self.df[self.df[text].str.contains(r"\.(?!\.)|\n")][text])).split(".") if sentence != " "]
        self.average_words_per_sentence = len(self.punctuation(unidecode(self.elongation_contraction(" ".join(sentences))))) / len(sentences)
        return
    
    def emo_se_corpus(self, title, artist, text, nlp):
        self.titles, self.artists, self.emo_se_analysis_corpus = zip(*[
            (self.df[title][i], self.df[artist][i], self.lemmatization(unidecode(self.elongation_contraction(self.df[text][i])), nlp)) 
            for i in range(len(self.df))])
        return
    
    def interpreter_recognition(self, tags, artists, threshold):
        return [
            [
                "".join(artist[0].split()) if (utils.full_process(tag) and (artist := process.extractOne(query = tag, choices = artists, score_cutoff = threshold))) else tag
                for tag in sublist
            ]
            for sublist in tags
        ]
    
    def person_recognition(self, tags, nlp):
        return [
            ["".join(
                [
                    person.text for person in nlp(tag)
                ]) 
                if nlp(tag).ents and nlp(tag).ents[0].label_ == "PERSON" 
                else tag for tag in sublist
            ]
            for sublist in tags
        ]
    
    def genre_tok(self, tags, genres):
        return [
            [
                re.sub("|".join(genres), r" \g<0> ", tag.lower()) for tag in sublist] 
                for sublist in tags
        ]
    
    def global_tok(self, tags, language):
        return [list(chain.from_iterable(
        [
            [
                tag for tag in self.punctuation(tag)
                if tag not in stopwords.words(language.lower())
            ]
            for tag in sublist
        ]))
        for sublist in tags
    ]

    def year_correction(self, tags):
        return [
            [
                re.sub(r"^\d{4}$", f"{tag[2]}0", tag) 
                if len(tag) >= 3 else tag
                for tag in sublist
            ]
            for sublist in tags
        ]
    
    def grouping(self, tags, threshold):
        tag_stems = {}
        for tag in set(chain(*tags)):
            added = False
            for tag_stem in tag_stems:
                for tag_within_stem in tag_stems[tag_stem]:
                    if fuzz.ratio(tag, tag_within_stem) > threshold:
                        tag_stems[tag_stem].append(tag)
                        added = True
                        break
            if not added:
                tag_stems[tag] = [tag]
        return [
            [
                next((key for key, values in tag_stems.items() if tag in values), tag)
                for tag in sublist
            ]
            for sublist in tags
        ]
    
    def tag_prep(self, tags, artists, genres, nlp, language, add, thresholds):
        nlp = spacy.load(nlp)
        tags_ner = self.person_recognition(self.interpreter_recognition(self.df[tags], artists, thresholds[0]), nlp)
        tags_norm = [
            [
                self.lemmatization(self.elongation_contraction(tag, add), nlp)
                for tag in tags
            ]
            for tags in tags_ner
        ]
        tags_tok = self.global_tok(self.genre_tok(tags_norm, genres), language)
        self.tags = self.grouping(self.year_correction(tags_tok), thresholds[1])
        return
    
def get_path(file):
    return os.path.join("..", "datasets", file)