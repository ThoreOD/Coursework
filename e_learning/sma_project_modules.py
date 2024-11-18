#################################
# 0 - General Libraries
#################################

from nltk.stem import LancasterStemmer
from tqdm import tqdm

import pandas as pd


#################################
# 2 - Data
#################################

from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager

import time

class scraper():
    def __init__(self, username, password, word, language, date_pattern, start_date, end_date):
        self.username = username
        self.password = password
        self.word = word
        self.language = language
        self.date_pattern = date_pattern
        self.start_date = start_date
        self.end_date = end_date

    #construct input dates
    def date_range(self):
        output_pattern = "%Y-%m-%d"
        start_date_obj, end_date_obj = map(lambda date: datetime.strptime(date, self.date_pattern), (self.start_date, self.end_date))
        date_pairs = [(start_date_obj + timedelta(days = days), start_date_obj + timedelta(days = days + 1)) for days in range((end_date_obj - start_date_obj).days)]
        return [(date[0].strftime(output_pattern), date[1].strftime(output_pattern)) for date in date_pairs]
    
    #HTML reference shortcuts
    def text_path(self, text):
        return f"//span[contains(text(), '{text}')]"
    def data_path(self, data, current = False):
        return f"{'.' if current else ''}//*[@data-testid = '{data}']"
    
    #start Google Chrome and access X
    def initiation(self):
        self.driver = webdriver.Chrome(service = ChromeService(ChromeDriverManager().install()))
        self.driver.maximize_window()
        self.driver.get("https://twitter.com/?lang=en")
        self.driver.implicitly_wait(10)

    #log in to X
    def log_in(self):
        self.driver.find_element(By.XPATH, self.text_path("Sign in")).click()
        for path, input in zip(["//*[@autocomplete = 'username']", "//*[@name = 'password']"], [self.username, self.password]):
            input_field = self.driver.find_element(By.XPATH, path)
            input_field.send_keys(input)
            input_field.send_keys(Keys.RETURN)

    #scrape tweets
    def search(self):
        self.tweet_dicts = []
        for start_date, end_date in self.date_range():

            #complete form 
            input_field = self.driver.find_element(By.XPATH, "//*[@placeholder = 'Search']")
            input_field.send_keys(f"{self.word} lang:{self.language} until:{end_date} since:{start_date} -filter:replies")
            input_field.send_keys(Keys.RETURN)
            for path in [self.text_path("Latest"), "//*[@enterkeyhint = 'search']", self.data_path("clearButton"), self.text_path("Clear all"), self.data_path("confirmationSheetConfirm")]:
                self.driver.find_element(By.XPATH, path).click()
            time.sleep(5)

            #scroll and scrape
            while True:
                # Get the current scroll position
                old_scroll_position = self.driver.execute_script("return window.scrollY")

                # Scroll down the page
                self.driver.execute_script("window.scrollBy(0, 500)")

                # Wait for some time for the new content to load
                time.sleep(0.5)

                [
                    self.tweet_dicts.append({
                        "user_identifier": next((userid for userid in tweet.find_element(By.XPATH, self.data_path("User-Name", True)).text.split() if userid.startswith("@")), None),
                        "timestamp": tweet.find_element(By.CSS_SELECTOR, "time").get_attribute("datetime"),
                        "text": tweet.find_element(By.XPATH, self.data_path("tweetText", True)).text
                    })
                    for tweet in self.driver.find_elements(By.XPATH, self.data_path("tweet"))
                ]

                # Get the new scroll position
                new_scroll_position = self.driver.execute_script("return window.scrollY")

                # Check if the scroll position hasn't changed, indicating that we've reached the bottom
                if new_scroll_position == old_scroll_position:
                    break

    #drop double scraped tweets
    def transform(self):
        self.tweets_df = pd.DataFrame(data = self.tweet_dicts).drop_duplicates().reset_index(drop = True)

    #terminate Google Chrome
    def close(self):
        self.driver.close()

    #wrapper
    def scraping(self):
        for step in ["initiation", "log_in", "search", "transform", "close"]:
            getattr(self, step)()


#################################
# 3 - Processing
#################################

from functools import reduce
from transformers import logging, pipeline
from unidecode import unidecode
from urlextract import URLExtract
from urllib.parse import urlparse

import contractions
import re
import spacy

class transformer_pipe():
    def __init__(self, csv, text):
        self.df = pd.read_csv(filepath_or_buffer = csv)
        self.text = self.df[text]

    #lowercase input
    def lowercase(self, text):
        return text.lower()

    #translate input's Unicode into ASCII, when possible
    def unicode_fix(self, text):
        return unidecode(string = text,
                         errors = "preserve")
    
    #expand input's contractions
    def contractions_fix(self, text):
        return contractions.fix(text)
    
    #remove input's URLs
    def url_fix(self, text):
        urls = set(URLExtract().find_urls(text))
        urls.update([word for word in text.split() if urlparse(word).scheme])
        return re.sub("|".join(map(re.escape, urls)), "", text)
    

    #raw string shortcuts 
    #pattern with whitespace (pw)
    def pw(self, pattern):
        return (r"{}".format(pattern), " ")
    
    #pattern with no whitespace (pnw)
    def pnw(self, pattern):
        return (r"{}".format(pattern), "")
    

    #regex pattern pool
    def pattern_init(self):
        self.alph_seq = self.pw("\w*\d+\w*")
        self.colloq_short = (r"'til", "until")
        self.email = self.pw(r"[\w.-]+@[\w-]+\.([a-z]{2,})+")
        self.g_drop = (r"(\w+)in'", r"\1ing")
        self.lead_wspace = self.pnw("^\s")
        self.newline = self.pw("\n")
        self.mention = self.pw("(?<!\w)@\w+")
        self.redup = (r"(\w)\1+", r"\1\1")
        self.trail_wspace = self.pnw("\s$")
        self.wspaces = self.pw("\s{2,}")

    #replace in input contained regex patterns
    def pattern_fix(self, text, patterns):
        return reduce(lambda text, patterns: re.sub(patterns[0], patterns[1], text), patterns, text)
    
    #predict sentiments
    def sens(self):
        logging.set_verbosity_error()
        pipe = pipeline(task = "text-classification",
                        model = "cardiffnlp/twitter-roberta-base-sentiment-latest")
        
        self.pattern_init()

        #apply RoBERTa transformer along preprocessing
        self.df["sentiment"] = [
            pipe(
                self.pattern_fix(
                    self.url_fix(self.contractions_fix(self.unicode_fix(self.lowercase(text)))),
                    [self.newline, self.alph_seq, self.mention, self.email, self.g_drop, self.colloq_short, self.wspaces, self.lead_wspace, self.trail_wspace]
                ),
                padding = True, truncation = True, max_length = 512
            )[0]["label"]
            for text in tqdm(
                self.text, desc = "sen_clas", unit = "texts"
            )
        ]

    #extract part-of-speech
    def lexemes(self, lexeme):
        model = spacy.load("en_core_web_trf")
        stemmer = LancasterStemmer()
        self.pattern_init()

        #apply spaCy transformer along preprocessing
        self.df[f"{lexeme.lower()}_stems"] = [
            {
                stemmer.stem(stem)
                for word in model(
                    self.pattern_fix(
                        self.url_fix(self.unicode_fix(text)),
                        [self.mention, self.email, self.g_drop, self.redup]
                    )
                )
                if word.pos_ == lexeme
                for stem in [word.text]
            }
            for text in tqdm(self.text, desc = f"{lexeme.lower()}_rec", unit = "texts")
        ]


#################################
# 4 - Analysis
#################################
        
from thefuzz import fuzz, process

import ast

class analysis:
    def __init__(self, csv, cols):
        self.df = pd.read_csv(filepath_or_buffer = csv)
        self.cols = cols
        for col in self.cols:
            self.df[col] = self.df[col].apply(ast.literal_eval)

    #combine columns into single set
    def retrieve_sets(self):
        return [
            [
                sets
                for sets in self.df[col] 
            ]
            for col in self.cols
        ]
    
    #filter for keywords
    def match(self, context, search_terms):
        sets = self.retrieve_sets()
        search_terms = [LancasterStemmer().stem(term) for term in search_terms]
        search_terms_min = min([len(term) for term in search_terms])

        #fuzzy match
        self.df["match"] = [
            [
                f"{matching[0]}${word}"
                for coll in sets
                for word in coll[i]
                if len(word) >= search_terms_min
                and
                (matching := process.extractOne(
                    query = word, 
                    choices = search_terms, 
                    score_cutoff = 100, 
                    scorer = fuzz.partial_ratio
                ))  
            ]
            for i in tqdm(range(len(self.df)), desc = "matching", unit = "texts")
        ]

        #evaluate relevancy
        self.df[f"{context}_relevancy"] = [
            any(match)
            for match in self.df["match"]
        ]