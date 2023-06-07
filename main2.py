import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans

import helper

# Import movies
movies = pd.read_csv('./datasets/movies.csv')
# Import Ratings
ratings = pd.read_csv('./datasets/ratings.csv')
# Calculate the average rating of romance and scifi movies
genre_ratings = helper.get_genre_ratings(ratings, movies, ['Romance', 'Sci-Fi'],
                                         ['avg_romance_rating', 'avg_scifi_rating'])
biased_dataset = helper.bias_genre_rating_dataset(genre_ratings, 3.2, 2.5)

# Merge the two tables then pivot, so we have Users X Movies dataframe
ratings_title = pd.merge(ratings, movies[['movieId', 'title']], on='movieId')
user_movie_ratings = pd.pivot_table(ratings_title.iloc[0:150000], index='userId', columns='title', values='rating')

n_movies = 30
n_users = 18
most_rated_movies_users_selection = helper.sort_by_rating_density(user_movie_ratings, n_movies, n_users)
print(user_movie_ratings)
print("----------------------")
print(most_rated_movies_users_selection)
# user_movie_ratings = pd.pivot_table(ratings_title.iloc[0:150000], index='userId', columns='title', values='rating')
most_rated_movies_1k = helper.get_most_rated_movies(user_movie_ratings, 1000)

sparse_ratings = csr_matrix(pd.SparseDataFrame(most_rated_movies_1k).to_coo())

predictions = KMeans(n_clusters=20, algorithm='full').fit_predict(sparse_ratings)

max_users = 70
max_movies = 50

clustered = pd.concat([most_rated_movies_1k.reset_index(), pd.DataFrame({'group': predictions})], axis=1)

# TODO: Pick a cluster ID from the clusters above
cluster_number = 11

# Let's filter to only see the region of the dataset with the number of values
n_users = 75
n_movies = 300
cluster = clustered[clustered.group == cluster_number].drop(['index', 'group'], axis=1)

cluster = helper.sort_by_rating_density(cluster, n_movies, n_users)

# TODO: Fill in the name of the column/movie. e.g. 'Forrest Gump (1994)'
movie_name = "Matrix, The (1999)"

print(cluster[movie_name].mean())

print(cluster.mean().head(20))

# Look at the table above outputted by the command "cluster.fillna('').head()"
# and pick one of the user ids (the first column in the table)
user_id = 19

# Get all this user's ratings
user_2_ratings = cluster.loc[user_id, :]

# Which movies did they not rate? (We don't want to recommend movies they've already rated)
user_2_unrated_movies = user_2_ratings[user_2_ratings.isnull()]

# What are the ratings of these movies the user did not rate?
avg_ratings = pd.concat([user_2_unrated_movies, cluster.mean()], axis=1, join='inner').loc[:, 0]

# Let's sort by rating so the highest rated movies are presented first
print(avg_ratings.sort_values(ascending=False)[:20])
