import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

import helper

# Import movies
movies = pd.read_csv('./datasets/movies.csv')

# Import Ratings
ratings = pd.read_csv('./datasets/ratings.csv')

ratings_twogenre = helper.get_genre_ratings(ratings, movies, ['Horror', 'Thriller'],
                                            ['avg_horror_rating', 'avg_thriller_rating'])
refined_dataset_twogenre = helper.bias_genre_rating_dataset(ratings_twogenre, 3.5, 2.5)

# helper.draw_scatterplot(refined_dataset_twogenre['avg_horror_rating'], 'Avg Horror rating',
#                         refined_dataset_twogenre['avg_thriller_rating'], 'Avg Thriller rating')

X = refined_dataset_twogenre[['avg_horror_rating', 'avg_thriller_rating']].values
from sklearn.cluster import KMeans

kmeans_two_genre = KMeans(n_clusters=2, random_state=0)
predictions1 = kmeans_two_genre.fit_predict(X)

# helper.draw_clusters(refined_dataset_twogenre, predictions1)

kmeans_two_genre1 = KMeans(n_clusters=3, random_state=1)
predictions2 = kmeans_two_genre1.fit_predict(X)

# helper.draw_clusters(refined_dataset_twogenre, predictions2)

kmeans_two_genre2 = KMeans(n_clusters=4, random_state=1)
predictions3 = kmeans_two_genre2.fit_predict(X)

# helper.draw_clusters(refined_dataset_twogenre, predictions3)

# list_of_k = range(2, len(X)+1, 5)
list_of_k = range(2, 264, 5)

print('Calculate error values for all above')
errors_list = [helper.clustering_errors(k, X) for k in list_of_k]

# # Plot the each value of K vs. the silhouette score at that value
# fig, ax = plt.subplots(figsize=(16, 6))
# ax.set_xlabel('Value of K')
# ax.set_ylabel('Score (higher is better)')
# ax.plot(list_of_k, errors_list)

# # Ticks and grid
# xticks = np.arange(min(list_of_k), max(list_of_k) + 1, 5.0)
# ax.set_xticks(xticks, minor=False)
# ax.set_xticks(xticks, minor=True)
# ax.xaxis.grid(True, which='both')
# yticks = np.arange(round(min(errors_list), 2), max(errors_list), .05)
# ax.set_yticks(yticks, minor=False)
# ax.set_yticks(yticks, minor=True)
# ax.yaxis.grid(True, which='both')
# plt.show()

kmeans_two_genre3 = KMeans(n_clusters=12, random_state=6)

predictions4 = kmeans_two_genre3.fit_predict(X)

# helper.draw_clusters(refined_dataset_twogenre, predictions4, cmap='Accent')

refined_dataset_3genre = helper.get_genre_ratings(ratings, movies,
                                                  ['Horror', 'Thriller', 'Fantasy'],
                                                  ['avg_horror_rating', 'avg_thriller_rating', 'avg_fantasy_rating'])
refined_dataset_3genre = helper.bias_genre_rating_dataset(refined_dataset_3genre, 3.5, 2.5).dropna()
refined_dataset_3genre.head()

X_fantasy = refined_dataset_3genre[['avg_horror_rating', 'avg_thriller_rating', 'avg_fantasy_rating']].values

kmeans_three_genre1 = KMeans(n_clusters=12)
predictions_1_1 = kmeans_three_genre1.fit_predict(X_fantasy)

helper.draw_clusters_3d(refined_dataset_3genre, predictions_1_1)

titles_df = pd.merge(ratings, movies[['movieId', 'title']], on='movieId')
ratings_users = pd.pivot_table(titles_df, index='userId', columns='title', values='rating')
# print(ratings_users.iloc[:5])

# shrinking the dataset for better visualization
num_movies = 30
num_users = 18
most_rated_sorted = helper.sort_by_rating_density(ratings_users, num_movies, num_users)
# print(most_rated_sorted.head())

helper.draw_movies_heatmap(most_rated_sorted)

ratings_df_subset = pd.pivot_table(titles_df, index='userId', columns='title', values='rating')
filtered_most_rated = helper.get_most_rated_movies(ratings_df_subset, 2000)

# Remove all nulls
tmpmovies = filtered_most_rated.copy()
tmpmovies = tmpmovies.fillna(0)
dtcols = filtered_most_rated.columns
tmpdict = {}
for v in dtcols:
    tmpdict[v] = pd.arrays.SparseArray(tmpmovies[v])
sparseFrame = pd.DataFrame(tmpdict)
sparse_ratings = csr_matrix(sparseFrame)

### Perform Predictions
new_k_values = range(2, 100 + 1, 5)
sparse_errors_k = [helper.clustering_errors(k, sparse_ratings) for k in new_k_values]

# fig, ax = plt.subplots(figsize=(16, 6))
# ax.set_xlabel('number of clusters')
# ax.set_ylabel('Score (higher is better)')
# ax.plot(new_k_values, sparse_errors_k)
# xticks = np.arange(min(new_k_values), max(new_k_values) + 1, 5.0)
# ax.set_xticks(xticks, minor=False)
# ax.set_xticks(xticks, minor=True)
# ax.xaxis.grid(True, which='both')
# yticks = np.arange(round(min(sparse_errors_k), 2), max(sparse_errors_k), .05)
# ax.set_yticks(yticks, minor=False)
# ax.set_yticks(yticks, minor=True)
# ax.yaxis.grid(True, which='both')
# plt.show()

predictions_sparse_1 = KMeans(n_clusters=12, algorithm='full').fit_predict(sparse_ratings)

predict_cluster = pd.concat([filtered_most_rated.reset_index(), pd.DataFrame({'group': predictions_sparse_1})], axis=1)
predict_cluster.head()

cluster_id = 5
newnum_users = 70
newnum_movies = 300
cluster_1 = predict_cluster[predict_cluster.group == cluster_id].drop(['index', 'group'], axis=1)
cluster_1 = helper.sort_by_rating_density(cluster_1, newnum_movies, newnum_users)
helper.draw_movies_heatmap(cluster_1, axis_labels=False)

cluster_1.fillna('').head()

picked_movie = 'Braveheart (1995)'
print(picked_movie)

cluster_1[picked_movie].mean()

cluster_1.mean().head(10)

cluster_1.fillna('').head()

user_id = 83
sel_user_ratings = cluster_1.loc[user_id, :]
all_unrated = sel_user_ratings[sel_user_ratings.isnull()]
mean_ratings = pd.concat([all_unrated, cluster_1.mean()], axis=1, join='inner').loc[:, 0]

print(mean_ratings.sort_values(ascending=False)[:5])
