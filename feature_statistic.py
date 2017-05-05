from sklearn import ensemble
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def feature_importance(X_df,y_df,how_many=10, verbose=False,plot=False):

    X = X_df.values
    y = y_df.values

    # Build a forest and compute the feature importances
    forest = ensemble.ExtraTreesRegressor(n_estimators=250)

    forest.fit(X, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    sorted_names = [X_df.columns[indices[i]] for i in range(len(indices))]

    if verbose:
        # Print the feature ranking
        print("Feature ranking:")

        for f in range(X.shape[1]):
            print("%d. feature %s (%f)" % (f + 1, sorted_names[f], importances[indices[f]]))

    if plot:
        # Plot the feature importances of the forest
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(X.shape[1]), importances[indices],
                color="r", yerr=std[indices], align="center")
        plt.xticks(range(X.shape[1]), sorted_names, rotation=90)
        plt.xlim([-1, X.shape[1]])
        plt.show()

    return sorted_names[:how_many]

def draw_hexbins(dataset, draw_these):
    # fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10,10))
    # ax=axes[i//3,i%3]
    # plt.subplots_adjust(wspace=0.5, hspace=0.5)
    for i,feature in enumerate(dataset.columns.values):
        if feature in draw_these:
            dataset.plot( y='imdb_score', x=feature, kind='hexbin', gridsize=35, sharex=False, colormap='cubehelix',
                         title='Hexbin of Imdb_Score and {}'.format(feature))
    plt.show()

def test(dataset):




    # Set up the matplotlib figure
    # f, ax = plt.subplots(2,5, figsize=(8, 8))
    plt.title('Pearson Correlation of Movie Features')
    # Draw the heatmap using seaborn
    yticks = dataset.columns.values
    xticks = dataset.columns.values

    sns.heatmap(dataset.astype(float).corr(), linewidths=0.25,xticklabels=xticks, yticklabels=yticks, vmax=1.0, square=True, cmap="YlGnBu",
                linecolor='black')
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)

    from sklearn import decomposition
    pca = decomposition.PCA(n_components=9)
    x_9d = pca.fit_transform(dataset)

    plt.figure(figsize=(7, 7))
    plt.scatter(x_9d[:, 0], x_9d[:, 1], c='goldenrod', alpha=0.5)
    plt.ylim(-10, 30)

    # Set a 3 KMeans clustering
    from sklearn import cluster
    kmeans = cluster.KMeans(n_clusters=3)
    # Compute cluster centers and predict cluster indices
    X_clustered = kmeans.fit_predict(x_9d)

    # Define our own color map
    LABEL_COLOR_MAP = {0: 'r', 1: 'g', 2: 'b'}
    label_color = [LABEL_COLOR_MAP[l] for l in X_clustered]

    # Plot the scatter digram
    plt.figure(figsize=(7, 7))
    plt.scatter(x_9d[:, 0], x_9d[:, 2], c=label_color, alpha=0.5)

    # Create a temp dataframe from our PCA projection data "x_9d"
    df = pd.DataFrame(x_9d)
    df = df[[0, 1, 2]]  # only want to visualise relationships between first 3 projections
    df['X_cluster'] = X_clustered
    #
    # # Call Seaborn's pairplot to visualize our KMeans clustering on the PCA projected data
    # sns.pairplot(df, hue='X_cluster', palette='Dark2', diag_kind='kde', size=1.85)

    sns.set(style='darkgrid')
    f, ax = plt.subplots(figsize=(8, 8))
    # ax.set_aspect('equal')
    ax = sns.kdeplot(df[0], df[1], cmap="Greens",
                     shade=True, shade_lowest=False)
    ax = sns.kdeplot(df[0], df[2], cmap="Reds",
                     shade=True, shade_lowest=False)
    ax = sns.kdeplot(df[1], df[2], cmap="Blues",
                     shade=True, shade_lowest=False)
    red = sns.color_palette("Reds")[-2]
    blue = sns.color_palette("Blues")[-2]
    green = sns.color_palette("Greens")[-2]
    ax.text(0.5, 0.5, "2nd and 3rd Projection", size=12, color=blue)
    ax.text(-5.8, -0.6, "1st and 3rd Projection", size=12, color=red)
    ax.text(-5.8, 0.25, "1st and 2nd Projection", size=12, color=green)
    plt.xlim(-20, 20)
    plt.ylim(-20, 20)

    plt.show()