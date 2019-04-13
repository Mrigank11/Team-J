#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predict classes
"""

import os
import argparse
import numpy as np
np.random.seed(13)

from src.request_prediction_xgb import prepare_data, predict
from sklearn.externals import joblib

parser = argparse.ArgumentParser()
parser.add_argument("-id", "--movie-id", type=int, help="Movie ID")
parser.add_argument("-g", "--genres", default="Comedy,Sci-Fi")
parser.add_argument("-rf", "--ratings-file", default="data/ratings.csv")
parser.add_argument("-mf", "--movies-file", default="data/movies.csv")
parser.add_argument("-pf", "--persistence-folder", default="tmp", help="folder to save trained model")
args = parser.parse_args()

genre_list = args.genres.split(',')

if args.movie_id:
    import pandas as pd
    movies = pd.read_csv(args.movies_file)
    movie = movies[movies['movieId']==args.movie_id]

    print("Using movie: {}".format(movie['title'].values[0]))
    genre_list = movie['genres'].values[0].split("|")

if not os.path.exists(args.persistence_folder):
    os.mkdir(args.persistence_folder)

data_file = os.path.join(args.persistence_folder, "data.pkl")

if os.path.exists(data_file):
    print("Using saved data.")
    mlb, scaler, model = joblib.load(data_file)
else:
    mlb, scaler, model = prepare_data(args.ratings_file, args.movies_file)
    joblib.dump((mlb, scaler, model), data_file)

prediction = predict(model, scaler, np.array([
    [0.156] + list(mlb.transform([genre_list])[0]) + [5]
]))
print(prediction)

