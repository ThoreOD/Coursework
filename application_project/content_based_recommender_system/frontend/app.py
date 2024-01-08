import os
import sys
from dotenv import load_dotenv
path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.append(os.path.join(path, "content_based_recommender_system", "backend"))

import pandas as pd
import recommender
from flask import Flask, render_template, request


application = Flask(__name__)

@application.route("/", methods = ["GET"])
def open_html():
    return render_template("test_framework.html")

load_dotenv(os.path.join(path, "content_based_recommender_system", "frontend", "variable.env"))
df = [pd.read_csv(filepath_or_buffer = os.path.join(path, "datasets", f"{df}.csv"))
      for df in [os.getenv("dataset_main"), os.getenv("dataset_genre"), os.getenv("dataset_lyrics_emo_sen"), os.getenv("dataset_lyrics_we"), os.getenv("dataset_tags")]]

r = recommender.Recommender(df[0], df[1], df[4], df[3], df[2])
@application.route("/recommend", methods = ["POST"])
def recommend():
    inputs = [input for input in request.form.values()]

    if "" in inputs:
        return render_template("test_framework.html", error_message = "Please ensure all required fields are completed before proceeding.")
    elif inputs[0].isnumeric():
        return render_template("test_framework.html", error_message = "Please include at least one alphabetical character in the song field.")
    elif inputs[1].isnumeric():
        return render_template("test_framework.html", error_message = "Please include at least one alphabetical character in the artist field.")
    try:
        repute = int(inputs[2])
        if repute < 0:
            return render_template("test_framework.html", error_message = "Please input only positive whole numbers in the repute field.")
        elif repute >= 100:
            return render_template("test_framework.html", error_message = "Please input only whole numbers smaller than 100 in the repute field")
    except ValueError:
        return render_template("test_framework.html", error_message = "Please include only whole numbers in the repute field.")
    
    output = r.recommend(inputs[0],
                         inputs[1],
                         len(df[0]),
                         inputs[2])
    
    spotify_ids = output["spotify_id"]
    input_song = f"https://open.spotify.com/embed/track/{spotify_ids[0]}?utm_source=generator"
    output_songs = [f"https://open.spotify.com/embed/track/{sid}?utm_source=generator" for sid in spotify_ids[1:]]
    output_songs_similarity = [f"{int(similarity)}% MATCH" for similarity in output["similarity"][1:]]

    return render_template("test_framework.html", rec = [input_song, output_songs, output_songs_similarity])

if __name__ == "__main__":
    application.run(debug = True)