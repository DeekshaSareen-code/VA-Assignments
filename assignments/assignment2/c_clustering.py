from assignments.assignment2.imports import *
from sklearn.cluster import KMeans
from sklearn.cluster import OPTICS, DBSCAN, AffinityPropagation, SpectralClustering
from sklearn import metrics

"""
Clustering is an unsupervised form of machine learning. It uses unlabeled data and returns the similarity/dissimilarity between rows of the data.
See https://scikit-learn.org/stable/modules/clustering.html for an overview of methods in sklearn.
"""


#######################################################
# Read the comments carefully in the following method(s)
# Assignment questions are marked as "??subtask"
#######################################################


def simple_k_means(x: pd.DataFrame, n_clusters=3, score_metric='euclidean') -> Dict:
    model = KMeans(n_clusters=n_clusters)
    clusters = model.fit_transform(x)

    # There are many methods of deciding a score of a cluster model. Here is one example:
    score = metrics.silhouette_score(x, model.labels_, metric=score_metric)
    return dict(model=model, score=score, clusters=clusters)


def iris_clusters() -> Dict:
    """
    Let's use the iris dataset and clusterize it:
    """

    iris = process_iris_dataset_again()
    # print("iris ",iris)
    # iris.drop("large_sepal_length", axis=1, inplace=True)

    # Let's generate the clusters considering only the numeric columns first
    no_species_column = simple_k_means(iris.iloc[:, :4])

    ohe = generate_one_hot_encoder(iris['species'])
    df_ohe = replace_with_one_hot_encoder(iris, 'species', ohe, list(ohe.get_feature_names()))

    # Notice that here I have binary columns, but I am using euclidean distance to do the clustering AND score evaluation
    # This is pretty bad
    no_binary_distance_clusters = simple_k_means(df_ohe)

    # Finally, lets use just a label encoder for the species.
    # It is still bad to change the labels to numbers directly because the distances between them does not make sense
    le = generate_label_encoder(iris['species'])
    df_le = replace_with_label_encoder(iris, 'species', le)
    labeled_encoded_clusters = simple_k_means(df_le)

    # See the result for yourself:
    r1 = round(no_species_column['score'], 2)
    r2 = round(no_binary_distance_clusters['score'], 2)
    r3 = round(labeled_encoded_clusters['score'], 2)
    print(
        f"Clustering Scores:\nno_species_column:{r1}, no_binary_distance_clusters:{r2}, labeled_encoded_clusters:{r3}")

    return max(r1, r2, r3)


##############################################
# Implement all the below methods
# Don't install any other python package other than provided by python or in requirements.txt
##############################################
def custom_clustering(x: pd.DataFrame, epsilon: float, minimum_samples: int) -> Dict:
    """
    As you saw before, it is much harder to apply the right distance metrics. Take a look at:
    https://scikit-learn.org/stable/modules/clustering.html
    and check the metric used for each implementation. You will notice that suppositions were made,
    which makes harder to apply these clustering algorithms as-is due to the metrics used.
    Also go into each and notice that some of them there is a way to choose a distance/similarity/affinity metric.
    You don't need to check how each technique is implemented (code/math), but do use the information from the clustering
    lecture and check the parameters of the method (especially if there is any distance metric available among them).
    Chose one of them which is, in your opinion, generic enough, and justify your choice with a comment in the code (1 sentence).
    The return of this method should be the model, a score (e.g. silhouette
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html) and the result of clustering the
    input dataset.
    """

    # Since DBSCAN takes the data based on density, outliers are automatically reduced, making it better or more
    # generic algorithm
    model = DBSCAN(eps=epsilon, min_samples=minimum_samples, n_jobs=-1)
    clusters = model.fit_predict(x)

    # print(model.labels_)

    # Silhouette score
    score = metrics.silhouette_score(x, model.labels_, sample_size=1000)

    return dict(model=model, score=score, clusters=clusters)


def cluster_iris_dataset_again() -> Dict:
    """
    Run the df returned form process_iris_dataset_again() method of A1 e_experimentation file through the custom_clustering and discuss (3 sentences)
    the result (clusters and score) and also say any limitations (e.g. problems with metrics) that you find.
    We are not looking for an exact answer, we want to know if you really understand your choice and the results of custom_clustering.
    Once again, don't worry about the clustering technique implementation, but do analyse the data/result and check if the clusters makes sense.
    """
    df1 = process_iris_dataset_again()
    final_data = custom_clustering(df1, 0.5, 30)
    print("cluster_iris_dataset_again: ", final_data)

    # After looking at the output, it gave us 0.73 score which is better than simple k mean.
    # One limitation that I found was it is finding multiple clusters, based on density.
    return final_data


def cluster_amazon_video_game() -> Dict:
    """
    Run the df returned from process_amazon_video_game_dataset() methods of A1 e_experimentation file through the custom_clustering and discuss (max 3 sentences)
    the result (clusters and score) and also say any limitations (e.g. problems with metrics) that you find.
    We are not looking for an exact answer, we want to know if you really understand your choice and the results of custom_clustering.
    Once again, don't worry about the clustering technique implementation, but do analyse the data/result and check if the clusters makes sense.
    """

    df2 = process_amazon_video_game_dataset()
    df2.drop("asin", axis=1, inplace=True)

    # Since Count and reviews have different ranges, we normalize them.
    for col in list(df2.columns):
        df2[col] = normalize_column(df2[col])

    print(df2)

    df2 = df2.sample(frac=0.3)

    final_data = custom_clustering(df2, 0.1, 2000)
    print("cluster_amazon_video_game: ", final_data)

    # We got 0.619 score only with this dataset. One limitation that we found was DBSCAN is resource intensive while
    # handling large datasets
    return final_data


def cluster_amazon_video_game_again() -> Dict:
    """
    Run the df returned from process_amazon_video_game_dataset_again() methods of A1 e_experimentation file through the custom_clustering and discuss (max 3 sentences)
    the result (clusters and score) and also say any limitations (e.g. problems with metrics) that you find.
    We are not looking for an exact answer, we want to know if you really understand your choice and the results of custom_clustering.
    Once again, don't worry about the clustering technique implementation, but do analyse the data/result and check if the clusters makes sense.
    """
    df3 = process_amazon_video_game_dataset_again()

    df3.drop("user", axis=1, inplace=True)

    # Since Count and reviews have different ranges, we normalize them.
    for col in get_numeric_columns(df3):
        df3[col] = normalize_column(df3[col])

    df3 = df3.sample(frac=0.08)
    print(df3)

    final_data = custom_clustering(df3, 0.1, 2000)

    # Using this configuration, and normalizing the columns, we found that the score increased to 0.83

    print("cluster_amazon_video_game_again: ", final_data)
    return dict(model=None, score=None, clusters=None)


def cluster_life_expectancy() -> Dict:
    """
    Run the df returned from process_amazon_video_game_dataset_again() methods of A1 e_experimentation file through the custom_clustering and discuss (max 3 sentences)
    the result (clusters and score) and also say any limitations (e.g. problems with metrics) that you find.
    We are not looking for an exact answer, we want to know if you really understand your choice and the results of custom_clustering.
    Once again, don't worry about the clustering technique implementation, but do analyse the data/result and check if the clusters makes sense.
    """
    df4 = process_life_expectancy_dataset()
    df4.drop("country", axis=1, inplace=True)
    df4["year"] = pd.to_numeric(df4["year"])

    for col in get_numeric_columns(df4):
        df4[col] = normalize_column(df4[col])

    df4.drop("Latitude", axis=1, inplace=True)
    df4.drop("x0_africa", axis=1, inplace=True)

    final_data = custom_clustering(df4, 0.05, 10)

    print("cluster_amazon_video_game_again: ", final_data)

    # Here we got result only 0.42. It shows DBSCAN is not good because it only cluster based on density. In this case
    # actual distance between points would work better.

    return final_data


def run_clustering():
    start = time.time()
    print("Clustering in progress...")
    # assert iris_clusters() is not None
    # assert len(cluster_iris_dataset_again().keys()) == 3
    # assert len(cluster_amazon_video_game().keys()) == 3
    # assert len(cluster_amazon_video_game_again().keys()) == 3
    assert len(cluster_life_expectancy().keys()) == 3

    end = time.time()
    run_time = round(end - start, 4)
    print("Clustering ended...")
    print(f"{30 * '-'}\nClustering run_time:{run_time}s\n{30 * '-'}\n")


if __name__ == "__main__":
    #
    desired_width = 320

    pd.set_option('display.width', desired_width)

    np.set_printoptions(linewidth=desired_width)

    pd.set_option('display.max_columns', 10)
    #

    run_clustering()
