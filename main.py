"""Problems on dataset
    1. 800 "0" in "gross" attribute
    2. 908 "0" in "director_facebook_likes"
    3. Need budget currency correction
    4. Fix budget and gross according to fix year

"""
# DONE: color                       --one hot encoding
# DONE: director_name	            --instead of names use smth else
# TODO: num_critic_for_reviews
# TODO: duration
# TODO: director_facebook_likes
# TODO: actor_3_facebook_likes
# DONE: actor_2_name                --instead of names use smth else
# TODO: actor_1_facebook_likes
# TODO: gross
# DONE: genres                      --split and one hot encoding
# DONE: actor_1_name                --instead of names use smth else
# DONE: movie_title                 --instead of names use smth else
# TODO: num_voted_users
# TODO: cast_total_facebook_likes
# DONE: actor_3_name                --instead of names use smth else
# TODO: facenumber_in_poster
# DONE: plot_keywords               --split - bag of words- one hot encoding || drop
# DONE: movie_imdb_link             --drop
# TODO: num_user_for_reviews
# DONE: language                    --one hot encoding
# DONE: country                     --one hot encoding
# DONE: content_rating              --one hot encoding
# DONE: budget                      --need currency correction
# TODO: title_year
# TODO: actor_2_facebook_likes
# TODO: imdb_score
# DONE: aspect_ratio                --drop
# TODO: movie_facebook_likes

# TODO: handle NANs


import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import feature_preprocessing
import feature_statistic
import predict_imdb_score

from sklearn import svm
from sklearn import neural_network
from sklearn import model_selection
from sklearn import ensemble
from sklearn import feature_selection

from sklearn import linear_model
from sklearn import metrics




def get_data():
    dataset = pd.read_csv("movie_metadata.csv", "r", encoding="utf8", delimiter=",")
    # drop duplicates
    dataset = dataset.drop_duplicates(['movie_title', 'title_year'])
    dataset = dataset.reset_index(drop=True)
    # before doing smth lets give IDs to every movie
    dataset['ID'] = dataset.index
    return dataset

def load_processed_data(path):
    return pd.read_pickle(path)



def normalize(X):
    # # Data Normalization
    from sklearn.preprocessing import StandardScaler,MinMaxScaler
    scaler = MinMaxScaler()
    return X.apply(lambda x: scaler.fit_transform(x))

def calculate(estimator, X_train, y_train, X_test, y_test, verbose=False):
    X_train = X_train.as_matrix()
    y_train = y_train.as_matrix()
    X_test = X_test.as_matrix()
    y_test = y_test.as_matrix()

    estimator.fit(X_train,y_train)
    mse = metrics.mean_squared_error(y_test, estimator.predict(X_test))
    mae = metrics.mean_absolute_error(y_test, estimator.predict(X_test))

    score_total = 0
    for X_, y_ in zip(X_test, y_test):
        prediction = estimator.predict(X_.reshape(1,-1))
        if abs(prediction - y_) < 1:
            score_total +=1
    print("score_total", score_total / y_test.shape[0])

    if verbose:
        for X_,y_ in zip(X_test,y_test):
            print(estimator.predict(X_.reshape(1,-1)), y_)

    print("mse", mse)
    print("mae", mae)
    print("score", estimator.score(X_test, y_test))






def main():
    # dataset_raw = get_data()
    # dataset = feature_processing(dataset, save=True, save_path='dataset_after_preprocessing.pickle')

    dataset = load_processed_data('dataset_after_clf_fill.pickle')

    # handle na values
    # dataset = feature_preprocessing.fillna_with_clf(dataset)
    # pd.to_pickle(dataset,'dataset_after_clf_fill_extra_tree.pickle')
    #

    dataset = dataset.drop('ID', axis=1)
    dataset = dataset.dropna()

    X = dataset.drop('imdb_score', axis=1)
    y = dataset['imdb_score']

    # X = normalize(X)
    import math


    log_subset = ['fixed_budget', 'num_voted_users', 'gross', 'movie_facebook_likes', 'num_user_for_reviews', 'num_critic_for_reviews', 'duration']
    X[log_subset] = X[log_subset].applymap(lambda x: math.log(x,2) if x != 0 else 0)


    # select important feature
    important_features = feature_statistic.feature_importance(X, y, how_many=11, verbose=False, plot=False)
    X = X[important_features]

    norm_dataset = X.copy(deep=True)
    norm_dataset['imdb_score'] = y

    # hexbin
    # feature_statistic.draw_hexbins(norm_dataset, ['imdb_score'])

    # feature_statistic.test(norm_dataset)




    svr = svm.SVR()
    linear_regressor = linear_model.LinearRegression()
    forest = ensemble.RandomForestRegressor(n_estimators=250)
    extra_forest = ensemble.ExtraTreesRegressor(n_estimators=250)
    boosting = ensemble.GradientBoostingRegressor(n_estimators=250)


    from sklearn import utils
    # X,y = utils.shuffle(X,y, random_state=0)
    X = X[:-380]
    y = y[:-380]
    print(X.shape)

    scorer = metrics.mean_squared_error
    print("SVR")
    print(np.mean(np.asarray(model_selection.cross_val_score(
        svr, X, y, cv = 5, scoring = metrics.make_scorer(scorer)))))

    print("Linear Regressor ==============")
    print(np.mean(np.asarray(model_selection.cross_val_score(
        linear_regressor, X, y, cv = 5, scoring = metrics.make_scorer(scorer)))))

    print("Random Forest ==================")
    print(np.mean(np.asarray(model_selection.cross_val_score(
        forest, X, y, cv = 5, scoring = metrics.make_scorer(scorer)))))

    print("Extra Forest ====================")
    print(np.mean(np.asarray(model_selection.cross_val_score(
        extra_forest, X, y, cv = 5, scoring = metrics.make_scorer(scorer)))))

    print("Boosting ====================")
    print(np.mean(np.asarray(model_selection.cross_val_score(
        boosting, X, y, cv = 5, scoring = metrics.make_scorer(scorer)))))
    #
    # X_train, X_test,y_train, y_test = model_selection.train_test_split(X,y,train_size=0.9)
    # print("SVR")
    # calculate(svr,X_train,y_train,X_test,y_test, verbose=False)
    # print("Linear Regressor ==============")
    # calculate(linear_regressor, X_train, y_train, X_test, y_test, verbose=False)
    # print("Random Forest ==================")
    # calculate(forest, X_train, y_train, X_test, y_test, verbose=False)
    # print("Extra Forest ====================")
    # calculate(extra_forest, X_train,y_train, X_test, y_test, verbose=False)
    # print("Boosting ====================")
    # calculate(boosting, X_train, y_train, X_test, y_test, verbose=False)





if __name__ == '__main__':
    main()







