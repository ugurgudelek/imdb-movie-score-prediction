import numpy as np
import pandas as pd
from sklearn import neighbors
import budget_fixer as bf

# feature preproccessing part

def name_to_score_avg(dataset):
    name_score_set = dataset[['director_name', 'actor_1_name','actor_2_name','actor_3_name','imdb_score']]

    unique_directors = name_score_set.director_name.unique()
    unique_actors = pd.Series(np.concatenate((name_score_set.actor_1_name.values,name_score_set.actor_2_name.values,name_score_set.actor_3_name.values))).unique()

    directors_dict = dict()
    actors_dict = dict()

    # calculate director's (imdb_sum,size)
    for i,director in enumerate(unique_directors):
        print(i,director)
        director_df = name_score_set.loc[name_score_set.director_name == director]
        size = director_df.imdb_score.size
        sum = director_df.imdb_score.sum()
        directors_dict[director] = (sum, size)

    # calculate actor's (imdb_sum,size)
    for i,actor in enumerate(unique_actors):
        print(i, actor)
        actor_df = name_score_set.loc[(name_score_set.actor_1_name == actor) | (name_score_set.actor_2_name == actor) | (name_score_set.actor_3_name == actor)]
        size = actor_df.imdb_score.size
        sum = actor_df.imdb_score.sum()
        actors_dict[actor] = (sum,size)


    for i,director in enumerate(name_score_set.director_name):
        print(i,director)
        d = directors_dict[director]        # d[0]: sum, d[1]: size

        # if director has only 1 movie. set previous average to nan
        name_score_set.director_name.iloc[i] = (d[0] - name_score_set.imdb_score.iloc[i]) / (d[1] - 1) if d[1] > 1 else np.nan

    for i,(actor_1,actor_2,actor_3) in enumerate(zip(name_score_set.actor_1_name,name_score_set.actor_2_name,name_score_set.actor_3_name)):
        print(i,actor_1,actor_2,actor_3)
        a_1 = actors_dict[actor_1]
        a_2 = actors_dict[actor_2]
        a_3 = actors_dict[actor_3]
        name_score_set.actor_1_name.iloc[i] = (a_1[0] - name_score_set.imdb_score.iloc[i]) / (a_1[1] - 1) if a_1[1] > 1 else np.nan
        name_score_set.actor_2_name.iloc[i] = (a_2[0] - name_score_set.imdb_score.iloc[i]) / (a_2[1] - 1) if a_2[1] > 1 else np.nan
        name_score_set.actor_3_name.iloc[i] = (a_3[0] - name_score_set.imdb_score.iloc[i]) / (a_3[1] - 1) if a_3[1] > 1 else np.nan

    dataset[['director_name', 'actor_1_name', 'actor_2_name', 'actor_3_name', 'imdb_score']] = name_to_score_avg(
        name_score_set)
    return dataset

def feature_processing(dataset, save=False, save_path = None):

    # drop unnecessary features
    dataset = dataset.drop('movie_imdb_link', axis=1)
    dataset = dataset.drop('aspect_ratio', axis=1)
    dataset = dataset.drop('movie_title', axis=1)

    # # plot_keywords
    # dataset.plot_keywords = dataset.plot_keywords.fillna("not_available")
    # unique_keywords = set()
    # for keyword in dataset.plot_keywords.str.split('|'):
    #     unique_keywords = unique_keywords.union(set(keyword))
    #
    # for keyword in unique_keywords:
    #     dataset['keyword_'+keyword] = dataset.plot_keywords.str.contains(keyword).astype(int)
    #
    # dataset = dataset.drop('keyword_not_available', axis=1)
    dataset = dataset.drop('plot_keywords', axis=1)

    # one hot encoding over color feature
    dataset.color = dataset.color.map({'Color': 1, ' BlackandWhite': 0})

    # one hot encoding language feature
    dataset.language = dataset.language.fillna("Other")
    dataset.language = pd.factorize(dataset.language)[0]

    # one hot encoding content_rating feature
    dataset.content_rating = dataset.content_rating.fillna("Not Rated")
    dataset.content_rating = pd.factorize(dataset.content_rating)[0]

    # split genres
    unique_genres = set()
    for genre in dataset.genres.str.split('|'):
        unique_genres = unique_genres.union(set(genre))

    for genre in unique_genres:
        dataset['genre_' + genre] = dataset.genres.str.contains(genre).astype(int)
    dataset = dataset.drop('genres', axis=1)


    # change wrong fb likes to np.NaN
    # actor_1_facebook_likes
    dataset.actor_1_facebook_likes.loc[dataset.actor_1_facebook_likes == 0] = np.nan

    # actor_2_facebook_likes
    dataset.actor_2_facebook_likes.loc[dataset.actor_2_facebook_likes == 0] = np.nan

    # actor_3_facebook_likes
    dataset.actor_3_facebook_likes.loc[dataset.actor_3_facebook_likes == 0] = np.nan

    # director_facebook_likes
    dataset.director_facebook_likes.loc[dataset.director_facebook_likes == 0] = np.nan


    # director,actor names
    # this part implemented as "name_to_score_avg" function
    dataset = name_to_score_avg(dataset)


    # prepare for budget fixing
    df = dataset.dropna(subset=['country', 'budget', 'title_year'])  # dropnan values for bf.handler
    df = df.reset_index(drop=True)  # reset indexes to not cause any problem
    usd_df = bf.handler(df.ID, df.country, df.budget, df.title_year, pd.read_pickle('hist_currency_dict.pickle'))
    dataset = dataset.merge(usd_df, on='ID', how='outer')  # merge outerly cuz we dont want to dropna
    dataset['fixed_budget'] = dataset['usd_today']
    dataset = dataset.drop(['usd_old', 'usd_today', 'budget'], axis=1)

    # one hot encoding country feature
    dataset.country = dataset.country.fillna("Other")
    dataset.country = pd.factorize(dataset.country)[0]

    # fill Nan values with ease of classifier
    scores = dataset.imdb_score
    train = dataset.drop('imdb_score', axis=1)
    dataset = fillna_with_clf(train)
    dataset['imdb_score'] = scores


    if save:
        pd.to_pickle(dataset, '{}'.format(save_path))
    return dataset

def fillna_classifier(estimator, feature_name, dataset, na_dropout_len=5):
    """:returns classified DataFrame['ID', feature_name]"""
    subset = dataset.columns.values.tolist()

    # maybe we want a subset to remove
    # remove_subset = None
    na_dropout_names = dataset.isnull().sum().sort_values(ascending=False).iloc[:na_dropout_len].index
    # drop first 5 most nan value column from dataset
    remove_subset = na_dropout_names

    for remove_ind in remove_subset:
        subset.remove(remove_ind)

    if feature_name in subset:
        subset.remove(feature_name)

    proper_df = dataset.dropna(subset=subset)



    train_df = proper_df[~dataset[feature_name].isnull()]
    train_X = np.asarray(train_df[subset])
    train_y = np.asarray(train_df[feature_name])

    test_df = proper_df[dataset[feature_name].isnull()]
    test_X = np.asarray(test_df[subset])

    # return if we have 0 nan value
    if len(test_X) == 0:
        return dataset.copy(deep=True)

    estimator.fit(train_X, train_y)

    prediction_df = pd.DataFrame(np.zeros(len(test_df[feature_name])), columns=[feature_name])
    prediction_df.index = test_df.index
    prediction_df['ID'] = test_df['ID']

    prediction_df[feature_name] = estimator.predict(test_X)

    dataset_copy = dataset.copy(deep=True)

    # dataset_copy = dataset_copy.set_index(dataset_copy.ID)
    for i,id in enumerate(prediction_df.ID):
        dataset_copy[feature_name].loc[dataset_copy.ID == id] = prediction_df[feature_name].iloc[i]

    print("{} {} item filled with classifier.".format(len(prediction_df.ID), feature_name))
    return dataset_copy

def fillna_with_clf(dataset):

    # extract features and match with models
    sorted_by_nan = dataset.isnull().sum().sort_values(ascending=False).index
    dataset = dataset.reindex(columns=sorted_by_nan)
    features = dataset.columns.values.tolist()
    estimators = list()
    for feature in features:
        print("feauture: {} -- {}".format(feature, dataset[feature].value_counts().size))
        if dataset[feature].value_counts().size > 50:      # assumed that we dont have any class which has 50 diff value
            estimators.append(neighbors.KNeighborsRegressor)
        else:
            estimators.append(neighbors.KNeighborsClassifier)


    # classify or regression for NaN values
    update_df = dataset.copy(deep=True)
    for feature,estimator in zip(features,estimators):      #foreach feature call classifier
        update_df[feature] = fillna_classifier(estimator(), feature, dataset)[feature]


    return update_df.copy(deep=True)