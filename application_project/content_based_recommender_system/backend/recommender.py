import numpy as np
import pandas as pd
import copy
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from thefuzz import fuzz, process

class Recommender:
    def __init__(self, df_main, df_genre, df_tags, df_embedding, df_sentiment, stored_recommendations=1):
        self.df_main = df_main
        self.df_genre = df_genre
        self.df_tags = df_tags
        self.df_embedding = df_embedding
        self.df_sentiment = df_sentiment
        self.stored_recommendations = stored_recommendations
        self.recommendations = []
        self.euclidean_distance = [0.012070768195597258, 3.164371987699992, 1.448376165768231]#min, max, mean
        self.weights = {"sound": {
                            "SELF": 1,
                            "danceability": 1.4,
                            "energy": 0.9,
                            "loudness": 2,
                            "mode": 0.3,
                            "speechiness": 2,
                            "acousticness": 0.65,
                            "instrumentalness": 0.7,
                            "liveness": 1,
                            "valence": 1,
                            "tempo": 3.5,
                            "key": 0.5,
                            "mode_key": 0.3,
                            "time_signature": 0.6
                        },
                        "g_popularity": {
                            "SELF": 0,
                            "popularity": 1,
                            "chart": 0.5,
                            "Listeners": 0.5,
                            "Playcount": 0.5
                        },
                        "lyrics": {
                            "SELF": 1,
                            "lyrics_length": 0.1,
                            "unique_words": 0.1,
                            "sentiment": 0.9,
                            "embedding": 0.1
                        },
                        "g_genre": {
                            "SELF": 1,
                            "genre": 3,
                            "tags": 0.1
                        },
                        "language": {
                            "SELF": 0.5
                        },
                        "other": {
                            "SELF": 0.05,
                            "duration_ms": 0.2,
                            "country": 0.5,
                            "year": 1
                        }
                        }

    def set_stored_recommendations(self, stored_recommendations):
        #check if its a positive integer
        if isinstance(stored_recommendations, int) & (stored_recommendations >= 0):
            self.stored_recommendations = stored_recommendations
        else:
            print("Error: stored_recommendations must be a positive integer.")

    def get_stored_recommendations(self):
        return self.stored_recommendations

    def get_recommendation(self, index=None):
        if index == None:
            return self.recommendations[len(self.recommendations)-1]
        else:
            return self.recommendations[index]
        
    def get_number_recommendations(self):
        return len(self.recommendations)
    
    def set_weight(self, name, weight):
        for i in self.weights.keys():
            if i == name:
                    self.weights[i]["SELF"] = weight
                    break
            for j in self.weights[i].keys():
                if j == name:
                    self.weights[i][j] = weight
                    break

    def get_weight(self, name):
        for i in self.weights.keys():
            if j == name:
                    return self.weights[i]["SELF"]
            for j in self.weights[i].keys():
                if j == name:
                    return self.weights[i][j]
                
    def set_weights(self, weights):
        try:
            for i in weights.keys():
                for j in weights[i].keys():
                    self.weights[i][j] = weights[i][j]
        except:
            print("Error: Wrong format for weights. Please use the following format: {'sound': {'SELF': 1, 'danceability': 1.4, ...},... 'g_popularity': {'SELF': 0.1, 'popularity': 1, ...}, ...}")

    def get_weights(self):
        return self.weights

    def store_recommendation(self, distances):
        if self.stored_recommendations == 0:
            return
        self.recommendations.append(distances)
        if len(self.recommendations) > self.stored_recommendations:
            self.recommendations.pop(0)

    def set_recommendations(self, recommendations):
        self.recommendations = recommendations
    
    def clear_recommendations(self):
        self.recommendations = []

    def prepare_df(self, df):
        not_necessary_features = ["isrc",
                                "album",
                                "release_date",
                                "release_date_precision",
                                "uri",
                                "country_code",
                                "Tags",
                                "clear_name",
                                "lyrics_url",
                                "lyrics",
                                "rank",
                                "chart_weeks",
                                "chart_rank"]

        #only compare if song was in charts or not
        df["chart"] = df["chart_peak"].apply(lambda x: 1 if x > 0 else 0)
        df = df.drop(["chart_peak"], axis=1)

        #values under 0.8 is not live 
        #self.df_main["liveness"] = self.df_main["liveness"].apply(lambda x: 1 if x >= 0.8 else 0)
        
        df = df.drop(not_necessary_features, axis=1)

        return df

    def key_distance(self, key1, key2):
        if (key1 < 0) | (key2 < 0):
            return 6
        return min(abs(key1 - key2), 12 - abs(key1 - key2))
    
    def mode_key_distance(self, mode1, key1, mode2, key2):
        k_d = self.key_distance(key1, key2)
        if k_d == 1 & mode1 == mode2:
            return 0
        if k_d == 0:
            return 0
        return 1
    
    #RECCOMENDATION

    def recommend(self, song_title, song_artist, recommendation_number=10, min_rank=0, same_artist=True):

        ###################### INIT ######################

        scaler = MinMaxScaler()

        #deepcopy to not change the original df
        df_main = copy.deepcopy(self.df_main)
        df_genre = copy.deepcopy(self.df_genre)
        df_tags = copy.deepcopy(self.df_tags)
        df_embedding = copy.deepcopy(self.df_embedding)
        df_sentiment = copy.deepcopy(self.df_sentiment)

        min_rank = float(min_rank)
        if min_rank != 0:
            min_rank = min_rank / 100
            min_rank = int(min_rank*len(df_main))
            #only songs with rank >= min_rank
            df_main["rank"] = scaler.fit_transform(df_main["rank"].to_numpy().reshape(-1, 1))
            df_main =df_main.sort_values(by="rank", ascending= False)
            df_main = df_main.iloc[:-min_rank, :]
            df_genre = df_genre.drop(df_genre.index.difference(df_main.index))
            df_tags.drop(df_tags.index.difference(df_main.index))


        song_artist = process.extract(song_artist, 
                                  df_main["artists"], 
                                  limit = 1, 
                                  scorer = fuzz.token_sort_ratio)[0][0]
        song_title = process.extract(song_title, 
                                 df_main[df_main["artists"] == song_artist]["name"], 
                                 limit = 1, 
                                 scorer = fuzz.ratio)[0][0]
        
        

        input_language = df_main[(df_main["name"] == song_title) & (df_main["artists"] == song_artist)]["language"].values[0]
        # recommendation_number = int(recommendation_number)
        min_rank = float(min_rank)

        #input song english -> only english songs because of nlp
        if input_language == "en":
            df_main = df_main[df_main["language"] == "en"]
            df_genre = df_genre.drop(df_genre.index.difference(df_main.index))
            df_tags = df_tags.drop(df_tags.index.difference(df_main.index))
        
        input_song = df_main[(df_main["name"] == song_title) & (df_main["artists"] == song_artist)]
        input_has_tags = not(input_song['Tags'].isnull().values.any())

        #if same artist -> drop songs exept of the input song
        if not same_artist:
            df_main = df_main[df_main["artists"] != input_song["artists"].values[0]]
            #add input song to df
            df_main = pd.concat([df_main, input_song])
            df_genre = df_genre.drop(df_genre.index.difference(df_main.index))

        if input_has_tags:
            df_tags = df_tags.drop(df_tags.index.difference(df_main.index))

        
        #if input language is english -> drop songs also from embedding and sentiment df
        if input_language == "en":
            df_embedding = pd.merge(df_embedding, df_main[["name", "artists"]], on=["name", "artists"], how="inner")
            df_sentiment = pd.merge(df_sentiment, df_main[["name", "artists"]], on=["name", "artists"], how="inner")
            
        #sort all and rest index
        df_main = df_main.sort_values(by=['artists', 'name']).reset_index(drop=True)
        df_genre = df_genre.sort_values(by=['artists', 'name']).reset_index(drop=True)
        df_tags = df_tags.sort_values(by=['artists', 'name']).reset_index(drop=True)
        df_embedding = df_embedding.sort_values(by=['artists', 'name']).reset_index(drop=True)
        df_sentiment = df_sentiment.sort_values(by=['artists', 'name']).reset_index(drop=True)

        input_index = df_main[(df_main["name"] == song_title) & (df_main["artists"] == song_artist)].index[0]

        df_main = self.prepare_df(df_main)

        df_distance = copy.deepcopy(df_main)
        
        ###################### CALC DISTANCE ######################

        # MAIN
        #special features
        df_distance["mode_key"] = df_distance.apply(lambda x: self.mode_key_distance(x["mode"], x["key"],
                                                                                     input_song["mode"].values[0],
                                                                                     input_song["key"].values[0]), axis=1)    #is alrady scaled
        df_distance["key"] = df_distance.apply(lambda x: self.key_distance(x["key"], input_song["key"].values[0]), axis=1)
        df_distance["key"] = df_distance["key"].div(6)  #scale key
        df_distance["language"] = df_distance["language"].apply(lambda x: 0 if x == input_language else 1)
        df_distance["country"] = df_distance["country"].apply(lambda x: 0 if x == input_song["country"].values[0] else 1)
        df_distance["time_signature"] = df_distance["time_signature"].apply(
            lambda x: 0 if x == input_song["time_signature"].values[0] else 1)
        special_features = ["key", "mode_key", "language", "country", "time_signature"]

        #other features
        only_sub_features = [x for x in df_main.columns if x not in (special_features +
                                                                     ["name", "artists", "genres", "spotify_id"])]  

        all_features = only_sub_features + special_features + ["genre", "tags", "embedding", "sentiment"]
        
        #scale

        df_distance[only_sub_features] = scaler.fit_transform(df_distance[only_sub_features])

        #difference to input song by substracting the input song value from each song value and take the absolute value
        for feature in only_sub_features:
            df_distance[feature] = df_distance[feature].sub(df_distance.loc[input_index, feature])
            df_distance[only_sub_features] = df_distance[only_sub_features].abs()

        # GENRE
        #scale
        genre = scaler.fit_transform(df_genre.drop(["name", "artists"], axis=1))
        #calculate distance
        df_distance["genre"] = cosine_distances(genre, genre[input_index, None])
        #scale again
        df_distance["genre"] = scaler.fit_transform(df_distance["genre"].to_numpy().reshape(-1, 1))

        # TAGS
        #no scaling needed
        if input_has_tags:
            tags = df_tags.drop(["name", "artists"], axis=1).to_numpy()
            df_distance["tags"] = cosine_distances(tags, tags[input_index, None])
            df_distance["tags"] = scaler.fit_transform(df_distance["tags"].to_numpy().reshape(-1, 1))
        else:
            df_distance["tags"] = 0

        if input_language == "en":
        # EMBEDDING
            # embedding = scaler.fit_transform(df_embedding.drop(["name", "artists"], axis=1))
            embedding = df_embedding.drop(["name", "artists"], axis=1).to_numpy()
            df_distance["embedding"] = cosine_distances(embedding, embedding[input_index, None]) 
            df_distance["embedding"] = scaler.fit_transform(df_distance["embedding"].to_numpy().reshape(-1, 1))

        # SENTIMENT
            sentiment = scaler.fit_transform(df_sentiment.drop(["name", "artists"], axis=1))
            df_distance["sentiment"] = cosine_distances(sentiment, sentiment[input_index, None])
            df_distance["sentiment"] = scaler.fit_transform(df_distance["sentiment"].to_numpy().reshape(-1, 1))
        else:
            df_distance["sentiment"] = 0
            df_distance["embedding"] = 0

        # square all
        df_distance[all_features] = df_distance[all_features].pow(2)

        ###################### WEIGHTING ######################

        w = copy.deepcopy(self.weights)
        
        if input_language != "en":
            del w["lyrics"]["embedding"]
            del w["lyrics"]["sentiment"]
        
        #calculate weighted distance for each feature
        for i in w.keys():
            for j in w[i].keys():
                if j == "SELF":
                    if i == "language":
                        df_distance[i] = df_distance[i].mul(w[i][j])
                        continue
                    group_weight = w[i][j]
                else:
                    df_distance[j] = df_distance[j].mul(w[i][j]).mul(group_weight)
        

        #calculate euclidian distance for each song (sum up and sqrt)
        df_distance["euclidean_distance"] = df_distance[all_features].sum(axis=1)
            
        df_distance["euclidean_distance"] = df_distance["euclidean_distance"].pow(0.5)

        df_distance["similarity"] = 100 - (df_distance["euclidean_distance"] -
                                     self.euclidean_distance[0]) / (self.euclidean_distance[1] - self.euclidean_distance[0]) * 100

        #sort by distance
        df_distance.sort_values(by=["euclidean_distance"],
                                ascending=True,
                                inplace=True,
                                ignore_index=True)
        if recommendation_number == "all":
            recommendation_number = len(df_distance) - 1
        else:
            recommendation_number = int(recommendation_number)
        similar_songs = df_distance[0:recommendation_number + 1][["name",
                                                                    "artists",
                                                                    "spotify_id",
                                                                    "euclidean_distance",
                                                                    "similarity"]]#change
        
        self.store_recommendation(df_distance)
        
        return similar_songs

    def disp_weights(self):
        total_group = 0
        total_subgroup = {}
        for key in self.weights:
            for subkey in self.weights[key]:
                if subkey == "SELF":
                    total_group = total_group + self.weights[key][subkey]
                    total_subgroup[key] = 0
                else:              
                    total_subgroup[key] = total_subgroup[key] + self.weights[key][subkey]
        df_weights = pd.DataFrame(columns=["group", "subgroup", "global_weight", "group_weight"])
        list_weights = []
        for key in self.weights:
            for subkey in self.weights[key]:
                if subkey == "SELF":
                    group_percent = self.weights[key][subkey] / total_group
                    list_weights.append({"group": key,
                                        "subgroup": subkey,
                                        "global_weight": round(self.weights[key][subkey] / total_group * 100, 2),
                                        "group_weight": 100})
                else:
                    list_weights.append({"group": key,
                                        "subgroup": subkey,
                                        "global_weight": round((self.weights[key][subkey] / total_subgroup[key]) * group_percent * 100, 2),
                                        "group_weight": round(self.weights[key][subkey] / total_subgroup[key] * 100, 2) })
        df_weights = pd.DataFrame(list_weights)

        return df_weights

    def feature_importance(self, recommendation_number=1, stored_recommendations=-1):
        df = self.recommendations[stored_recommendations]
        df_importance = pd.DataFrame(columns=["feature", "values", "importance"])
        df_importance["feature"] = df.columns[4:-2]
        #take values from df by recommendation_number as index
        df_importance["values"] = df.iloc[recommendation_number][4:-2].values
        #calculate importance
        df_importance["importance"] = df_importance["values"].div(df_importance["values"].sum()).mul(100)
        df_importance.sort_values(by=["importance"],
                                ascending=False,
                                inplace=True,
                                ignore_index=True)
        return df_importance
    
    def feature_statistics(self, stored_recommendations=-1, head=None):
        df = self.recommendations[stored_recommendations]
        df_statistic = pd.DataFrame(columns=["feature", "mean", "min", "max"])
        df_statistic["feature"] = df.columns[4:-2]
        if head == None:
            df_statistic["mean"] = df.iloc[:, 4:-2].mean().values
            df_statistic["min"] = df.iloc[:, 4:-2].min().values
            df_statistic["max"] = df.iloc[:, 4:-2].max().values
        elif head >= 0:
            df_statistic["mean"] = df.iloc[0:head, 4:-2].mean().values
            df_statistic["min"] = df.iloc[0:head, 4:-2].min().values
            df_statistic["max"] = df.iloc[0:head, 4:-2].max().values
        else:
            df_statistic["mean"] = df.iloc[head:, 4:-2].mean().values
            df_statistic["min"] = df.iloc[head:, 4:-2].min().values
            df_statistic["max"] = df.iloc[head:, 4:-2].max().values

        return df_statistic.sort_values(by=["mean"], ascending=False)
    
    def feature_statistics_all(self, head=None):
        df_statistic = pd.DataFrame(columns=["feature", "mean", "min", "max"])
        for i in range(len(self.recommendations)):
            df_statistic = pd.concat([df_statistic, self.feature_statistics(i, head)])
        df_statistic_mean = df_statistic.groupby("feature").mean().reset_index()
        df_statistic_mean.columns = ["feature", "mean", "mean_min", "mean_max"]
        #min and max values
        df_statistic_min = df_statistic.groupby("feature").min().reset_index()
        df_statistic_max = df_statistic.groupby("feature").max().reset_index()
        df_statistic_min.drop(["mean", "max"], axis=1, inplace=True)
        df_statistic_max.drop(["mean", "min"], axis=1, inplace=True)
        #merge
        df_statistic = df_statistic_mean.merge(df_statistic_min, on="feature")
        df_statistic = df_statistic.merge(df_statistic_max, on="feature")
        
        return df_statistic.sort_values(by=["mean"], ascending=False)
    
    def recommend_random(self, cnt_recommendations=1, recommendation_number=10, min_rank=0, same_artist=True):
        for i in range(cnt_recommendations):
            sample = self.df_main.sample()
            song_title = sample["name"].values[0]
            song_artist = sample["artists"].values[0]
            if cnt_recommendations == 1:
                return self.recommend(song_title, song_artist, recommendation_number, min_rank, same_artist)
            self.recommend(song_title, song_artist, recommendation_number, min_rank, same_artist)

    def recommend_random_sequential(self, min_rank, same_artist):
        sample = self.df_main.sample()
        song_title = sample["name"].values[0]
        song_artist = sample["artists"].values[0]
        self.recommend(song_title, song_artist, min_rank=min_rank, same_artist=same_artist)
        return self.recommendations[-1]
    
    def recommend_random_parallel(self, cnt_recommendations, min_rank=0, same_artist=True, n_jobs=-1):
        result = Parallel(n_jobs=n_jobs)(delayed(self.recommend_random_sequential)(min_rank, same_artist) for i in range(cnt_recommendations))
        self.set_recommendations(result)

    def feature_pie_chart(self, all_recs=False, stored_recommendations=-1, head=None, recommendation_number=None):
        plt.figure(figsize=(10, 10))
        if all_recs:
            df =self.feature_statistics_all(head)
            plt.pie(df["mean"], labels=df["feature"], autopct='%1.1f%%')
        elif recommendation_number != None:
            df = self.feature_importance(recommendation_number, stored_recommendations)
            plt.pie(df["values"], labels=df["feature"], autopct='%1.1f%%')
        else:
            df = self.feature_statistics(stored_recommendations, head)
            plt.pie(df["mean"], labels=df["feature"], autopct='%1.1f%%')
        plt.show()

    def update_similarity(self, n_jobs=-1):
        self.clear_recommendations()
        self.set_stored_recommendations(0)
        def recommend_parallel(i):
            result = self.recommend(self.df_main["name"].values[i], self.df_main["artists"].values[i], "all")
            return result["euclidean_distance"].values[1:].min(), result["euclidean_distance"].values[1:].max(), result["euclidean_distance"].values[1:].mean()
        result = Parallel(n_jobs=n_jobs)(delayed(recommend_parallel)(i) for i in range(len(self.df_main)))
        # min  und max euclidean distance
        df_result = pd.DataFrame(result, columns=["min", "max", "mean"])
        max_euclidean_distance = df_result["max"].max()
        min_euclidean_distance = df_result["min"].min()
        mean_euclidean_distance = df_result["mean"].mean()
        self.euclidean_distance = [min_euclidean_distance, max_euclidean_distance, mean_euclidean_distance]
        return self.euclidean_distance
    
    def generate_recommender_file(self, path, max_rank = 0, n_jobs=-1):
        self.clear_recommendations()
        self.set_stored_recommendations(0)
        def recommend_parallel(i):
            result = self.recommend(self.df_main["name"].values[i], self.df_main["artists"].values[i], 20, max_rank)
            last_id = result[1:]["spotify_id"].values[-1]
            result = self.recommend(self.df_main["name"].values[i], self.df_main["artists"].values[i], "all")
            result = result[1:]
            #drop all songs after last_id
            last_id_index = result[result["spotify_id"] == last_id].index[0]
            # print(last_id_index)
            result = result.drop(result.index[last_id_index+1:])
            return result["spotify_id"].to_list(), result["similarity"].to_list(), result["rank"].to_list()
        # result = []
        # for i in range(10):
        #     result.append(recommend_parallel(i))
        result = Parallel(n_jobs=n_jobs)(delayed(recommend_parallel)(i) for i in range(len(self.df_main)))
        result_df = pd.DataFrame(result, columns=["spotify_ids", "similaritys", "ranks"])
        result_df = pd.concat([self.df_main[["name", "artists", "spotify_id"]], result_df], axis=1)
        result_df.to_csv(path, index=False)
        return result_df


