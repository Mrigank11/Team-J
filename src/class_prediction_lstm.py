import datetime
import matplotlib
matplotlib.use('agg')
import pandas as pd
import numpy as np
np.random.seed(13)

def prepare_data(ratings_file, movies_file, movie_id):
    """
    :param ratings_file: csv file containing ratings
    :param movies_file: csv file containing movies with genres
    :param movie_id: movie id

    :returns processed numpy matrix, mlb
    """

    """# Load data"""

    rating_data = pd.read_csv(ratings_file, index_col='timestamp', parse_dates=['timestamp'], date_parser = lambda x: datetime.datetime.fromtimestamp(int(x)))
    movie_data = pd.read_csv(movies_file, index_col='movieId')

    """## Process data
    **Assumption**: one rating = one request
    """

    del rating_data['userId']
    rating_data.head()

    del movie_data['title']
    movie_data.head()

    def assign_class(n_req, total_req):
        """
        assign class based on the no. of request received
        """
        
        # top 1% of the day are Class 1
        if n_req > 0.01 * total_req:
            return 1
        # top 0.5% of the day are Class 2
        elif n_req > 0.005 * total_req:
            return 2
        # top 0.01% of the day are Class 3
        elif n_req > 0.001 * total_req:
            return 3
        # rest are Class 4
        else:
            return 4

    def group_movies(daywise_group):
        """
        add request probablity to input pandas DataFrame
        
        :param daywise_group: a DataFrame containing movieId and corresponding ratings for a single day.
        """

        df = daywise_group.groupby("movieId").count()   
        # `rating` column now holds the no. of requests, so rename the column to avoid confusion
        df.rename(columns={"rating":"req"}, inplace=True)
        df['totalReq'] = daywise_group.shape[0]
        # request probablity for that day
        df['reqProb'] = daywise_group.groupby("movieId").count()['rating'] / df['totalReq']
        return df

    grouped_by_movie = rating_data.resample("1y").apply(group_movies)
    grouped_by_movie['class'] = grouped_by_movie[['req', 'totalReq']].apply(lambda x:assign_class(x['req'], x['totalReq']), axis=1)
    grouped_by_movie.reset_index(level='movieId', inplace=True)
    grouped_by_movie.head()

    grouped_by_movie.pivot(columns='movieId', values='class').fillna(0).iloc[:100,:5].plot()

    """### Add genres"""

    grouped_by_movie = pd.merge(grouped_by_movie.reset_index(), movie_data.reset_index())
    grouped_by_movie.set_index('timestamp', inplace = True)
    grouped_by_movie.head()

    # Convert genres(str) to a binary matrix
    # because string are good for humans but
    # machines like numbers!
    from sklearn.preprocessing import MultiLabelBinarizer
    splitted = grouped_by_movie.apply(lambda x:x['genres'].split("|"), axis=1)
    mlb = MultiLabelBinarizer()
    genre_matrix = mlb.fit_transform(splitted)

    genre_df = pd.DataFrame(genre_matrix)

    with_genre = pd.merge(grouped_by_movie.reset_index(), genre_df.reset_index(), left_index=True, right_index=True)
    del with_genre['genres'], with_genre['index']

    # shift class column to the end
    with_genre = with_genre[ list(filter(lambda x: x!='class', with_genre.columns)) + ['class']]

    with_genre.set_index('timestamp', inplace = True)
    with_genre.head()

    # We are selecting a single movie here, selecting 
    # multiple movies will yield bad results because 
    # of difference in data points.
    movie_history = with_genre[with_genre['movieId'] == movie_id]
    movie_history['Y'] = np.roll(movie_history['class'], -1)
    movie_history.pivot(columns='movieId', values='req').plot(kind='bar', title="Request Counts/day")
    del movie_history['totalReq']
    movie_history.head()

    """# Crunching numbers!"""

    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer, make_column_transformer
    # One-hot-encode classes
    preprocess = make_column_transformer([OneHotEncoder(categories=[np.array([1,2,3,4])]), ['class', 'Y']])
    values = movie_history.values
    #print(values)
    one_hot_encoded = np.concatenate((list(map(lambda x:np.eye(4)[x-1], values[:,23].astype('int')) ),
                   list(map(lambda x:np.eye(4)[x-1], values[:,24].astype('int')))), axis=1)
    values = np.concatenate((movie_history.values[:,:23], one_hot_encoded), axis=1)

    # Scale 
    from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(values)
    scaled_values.shape

    """## Make X and Y"""

    n_input_cols = 27
    n_output_cols = 4
    X = scaled_values[:,:n_input_cols]
    y = scaled_values[:,n_input_cols:]
    # reshape
    X = X.reshape(X.shape[0], 1, n_input_cols)
    y = y.reshape(y.shape[0], 1, n_output_cols)
    print(X.shape)
    print(y.shape)

    """## train_test_split"""

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=False)

    """# Make Model"""

    from keras.models import Sequential
    from keras.layers import Dropout, LSTM, BatchNormalization, TimeDistributed, Dense, Activation
    model = Sequential()
    model.add(BatchNormalization(input_shape=(1,X_train.shape[2])))
    model.add(LSTM(4, return_sequences=True))
    model.add(LSTM(1, dropout=0.2, return_sequences=True))
    model.add(TimeDistributed(Dense(y_train.shape[2])))
    model.add(Activation('softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    history = model.fit(X_train, y_train, batch_size=10, epochs=10)
    score, acc = model.evaluate(X_test, y_test, batch_size=100)
    print('Test score:', score)
    print('Test accuracy:', acc)

    return mlb, scaler, model

def predict(model, scaler, input_array):
    n_output_cols = model.output_shape[2]
    n_input_cols = model.input_shape[2]
    dat = np.concatenate((input_array, [[0]*n_output_cols]*input_array.shape[0]), axis=1)
    dat = scaler.transform(dat)[:,:n_input_cols]
    dat = dat.reshape(dat.shape[0], 1, n_input_cols)
    predicted = model.predict(dat)

    predicted = predicted.reshape(1,n_output_cols)
    x = np.concatenate(([[0]*n_input_cols]*input_array.shape[0], predicted), axis=1)
    return scaler.inverse_transform(x)[:,n_input_cols:]
