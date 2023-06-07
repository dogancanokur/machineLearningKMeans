import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans

import helper


def train():
    print("train basladi")
    # Import movies
    movies = pd.read_csv('./datasets/movies.csv')
    # Import Ratings
    ratings = pd.read_csv('./datasets/ratings.csv')

    titles_df = pd.merge(ratings, movies[['movieId', 'title']], on='movieId')

    ratings_df_subset = pd.pivot_table(titles_df.iloc[0:1500000], index='userId', columns='title', values='rating')
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
    # Perform Predictions
    predictions_sparse_1 = KMeans(n_clusters=12, algorithm='full').fit_predict(sparse_ratings)

    predict_cluster = pd.concat([filtered_most_rated.reset_index(), pd.DataFrame({'group': predictions_sparse_1})],
                                axis=1)

    cluster_id = 5
    newnum_users = 70
    newnum_movies = 300
    cluster_1 = predict_cluster[predict_cluster.group == cluster_id].drop(['index', 'group'], axis=1)
    cluster_1 = helper.sort_by_rating_density(cluster_1, newnum_movies, newnum_users)
    # print(cluster_1.fillna('').head())
    print("train bitti")
    return cluster_1


train_data = train()
train_data.to_excel('./train.xlsx', sheet_name='sheet1', index=False)



# movie_name = "Matrix, The (1999)"
movie_name = "Cinderella (1950)"
print(train_data.mean().head(20))
train_data[movie_name].mean()

# TODO: Pick a user ID from the dataset
# Look at the table above outputted by the command "cluster.fillna('').head()"
# and pick one of the user ids (the first column in the table)
user_id = 83438

# train_data.to_excel('./test.xlsx', sheet_name='sheet1', index=False)
# Get all this user's ratings
user_2_ratings  = train_data.loc[user_id, :]

# Which movies did they not rate? (We don't want to recommend movies they've already rated)
user_2_unrated_movies =  user_2_ratings[user_2_ratings.isnull()]

# What are the ratings of these movies the user did not rate?
avg_ratings = pd.concat([user_2_unrated_movies, train_data.mean()], axis=1, join='inner').loc[:,0]

# Let's sort by rating so the highest rated movies are presented first
avg_ratings.sort_values(ascending=False)[:20]
















film = ""
# Cinderella (1950)
# Raising Arizona (1987)
# Silence of the Lambs, The (1991)
# Titanic (1997)
# Casablanca (1942)
# Ghost (1990)
print("Film Öneri Programı\n")

# print("'exit' yazarak cikabilirsiniz\n")
# print("1 - Cinderella (1950)\n")
# print("2 - Raising Arizona (1987)\n")
# print("3 - Silence of the Lambs, The (1991)\n")
# print("4 - Titanic (1997)\n")
# print("5 - Casablanca (1942)\n")
# print("6 - Ghost (1990)\n")
# print("-------------------")
# while film != "exit":
#     film = input(("Film seçiniz giriniz: "))
#     if film == "1": film = 'Cinderella (1950)'
#     if film == "2": film = 'Raising Arizona (1987)'
#     if film == "3": film = 'Silence of the Lambs, The (1991)'
#     if film == "4": film = 'Titanic (1997)'
#     if film == "5": film = 'Casablanca (1942)'
#     if film == "6": film = 'Ghost (1990)'
#     if film != "exit":
#         print(film)
#
#         train_data = train_data.fillna('-')
#         print(train_data[film].head(10))
#         print("--------------------------\n")
#         train_data[film].to_excel('./result.xlsx', sheet_name='sheet1', index=False)
#
#         print(train_data.head())
#         print(train_data[film].mean())
#
#         print(train_data.mean().head(10))
#         train_data.mean().to_excel('./train2.xlsx', sheet_name='sheet1', index=False)
#         train_data.mean().sort_values(by=train_data.columns[1]).to_excel('./train3.xlsx', sheet_name='sheet1',
#                                                                          index=False)
#
#         # user_id = 83
#         # sel_user_ratings = train_data.loc[user_id, :]
#         # all_unrated = sel_user_ratings[sel_user_ratings.isnull()]
#         # mean_ratings = pd.concat([all_unrated, train_data.mean()], axis=1, join='inner').loc[:, 0]
#         #
#         # print(mean_ratings.sort_values(ascending=False)[:5])
