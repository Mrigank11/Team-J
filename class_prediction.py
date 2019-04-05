#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predict classes
"""

import os
import argparse
import numpy as np
np.random.seed(13)

from src.class_prediction_lstm import prepare_data, predict
from sklearn.externals import joblib

parser = argparse.ArgumentParser()
parser.add_argument("-id", "--movie-id", default=356, type=int, help="Movie ID")
parser.add_argument("-rf", "--ratings-file", default="data/ratings.csv")
parser.add_argument("-mf", "--movies-file", default="data/movies.csv")
parser.add_argument("-pf", "--persistence-folder", default="tmp", help="folder to save trained model")
args = parser.parse_args()

if not os.path.exists(args.persistence_folder):
    os.mkdir(args.persistence_folder)

data_file = os.path.join(args.persistence_folder, "{}.data.pkl".format(args.movie_id))

if os.path.exists(data_file):
    print("Using saved data.")
    mlb, scaler, model = joblib.load(data_file)
else:
    mlb, scaler, model = prepare_data(args.ratings_file, args.movies_file, args.movie_id)
    joblib.dump((mlb, scaler, model), data_file)

prediction = predict(model, scaler, np.array([
    [356,56,0.156] + list(mlb.transform([['Comedy', 'Drama', 'Romance', 'War']])[0]) + [1,0,0,0]
]))
print(prediction)

