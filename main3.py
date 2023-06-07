import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans

import helper

# Import movies
movies = pd.read_csv('./datasets/movies.csv')
# Import Ratings
ratings = pd.read_csv('./datasets/ratings.csv')
def getrecoms():
    inputdata=request.get_json()
    usrid='x'+str(random.randint(1,9000))
    tmpinput=[]
    for v in inputdata:
        tmpdict={}
        tmpdict['userId']=usrid
        tmpdict['movieId']=v['movieId']
        tmpdict['rating']=v['rating']
        tmpdict['timestamp']=964982703
        tmpinput.append(tmpdict)
    out=evaluateinput(tmpinput)
    resp=make_response(jsonify(results=out))
    return resp

def evaluateinput():
    for rw in inputuser:
        ratings=ratings.append({'userId':rw['userId'],'movieId':rw['movieId'],'rating':rw['rating'],'timestamp':964982703},ignore_index=True)
    # Merge the two tables then pivot so we have Users X Movies dataframe
    ratings_title = pd.merge(ratings, movies[['movieId', 'title']], on='movieId' )
    user_movie_ratings = pd.pivot_table(ratings_title, index='userId', columns= 'title', values='rating')

    user_movie_ratings =  pd.pivot_table(ratings_title, index='userId', columns= 'title', values='rating')
    most_rated_movies_1k=user_movie_ratings

    # Remove all nulls
    tmpmovies=most_rated_movies_1k.copy()
    tmpmovies=tmpmovies.fillna(0)
    dtcols=most_rated_movies_1k.columns
    tmpdict={}
    for v in dtcols:
        tmpdict[v]=pd.arrays.SparseArray(tmpmovies[v])

    sparseFrame=pd.DataFrame(tmpdict)
    sparse_ratings = csr_matrix(sparseFrame)

    # 20 clusters
    predictions = KMeans(n_clusters=20, algorithm='full').fit_predict(sparse_ratings)

    clustered = pd.concat([most_rated_movies_1k.reset_index(), pd.DataFrame({'group':predictions})], axis=1)

    cluster_number = clustered[clustered['userId']==inputuser[0]['userId']]['group'].values[0]

    cluster = clustered[clustered.group == cluster_number].drop(['group'], axis=1)
    user_id = cluster[cluster['userId']==inputuser[0]['userId']].index[0]

    # Get all this user's ratings
    user_2_ratings  = cluster.loc[user_id, :]
    user_2_unrated_movies =  user_2_ratings[user_2_ratings.isnull()]

    # What are the ratings of these movies the user did not rate?
    avg_ratings = pd.concat([user_2_unrated_movies, cluster.mean()], axis=1, join='inner').loc[:,0]

    # Let's sort by rating so the highest rated movies are presented first
    avg_ratings.sort_values(ascending=False)[:20]

    otput_names=avg_ratings.sort_values(ascending=False)[:5].index

    outputlist=[]
    for nme in otput_names:
        tmpdict={}
        tmpnme=nme[:-7]
        tmpdict['title']=tmpnme
        #get movie image
        apiurl='http://www.omdbapi.com/?t='+tmpnme+'&apikey=apikey'
        resp=requests.get(apiurl)
        if resp.json()['Response']=='False':
            continue
        tmpdict['imageurl']=resp.json()['Poster']
        outputlist.append(tmpdict)
    return outputlist
