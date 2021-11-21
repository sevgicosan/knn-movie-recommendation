import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

# read the files
print("reading files...")
ratings = pd.read_csv('/Users/sevgicosan/Desktop/Movie Recommendation System/ratings.csv_back')
movies = pd.read_csv('/Users/sevgicosan/Desktop/Movie Recommendation System/movies.csv_back')

# make copies
print("copying files...")
ratings_copy = ratings.copy()
movies_copy = movies.copy()

# 'timestamp' column is not necessary
print("dropping timestamp...")
ratings_copy.drop("timestamp", axis = 1, inplace = True)

# top_movies are the movies that has more than 100 ratings
# top_movies_ids are the IDs of top_movies
print("removing movies with less than 100 ratings...")
top_movie_ids = ratings_copy[ratings_copy['movieId'].isin(ratings_copy['movieId'].value_counts()[ratings_copy['movieId'].value_counts() >= 100].index)].movieId.unique()
top_movie_ratings = ratings.loc[ratings["movieId"].isin(top_movie_ids)]
top_movies = movies.loc[movies["movieId"].isin(top_movie_ids)]

# Merge movies and ratings dataframes
print("merging movies and ratings...")
movies_and_ratings = pd.merge(top_movies,top_movie_ratings, on = 'movieId')

# Create pivot table
print("creating pivot table...")
movies_and_features = movies_and_ratings.pivot(index = 'movieId', columns = 'userId', values = 'rating')

# Replace NAs with 0s
print("replacing NAs with 0s...")
movies_and_features.fillna(0, inplace = True)

# movies_and_features is a sparse matrix.
print("converting to csr_matrix...")
mat_movies_and_features = csr_matrix(movies_and_features.values)

# Fit KNN model with k = 6
print("fitting model...")
neigh = NearestNeighbors(n_neighbors = 6)
neigh.fit(mat_movies_and_features)

while True:
    # Get movie id as input
    movie_id = input('movie_id: ')
    movie_id = int(movie_id.strip())

    if movie_id in top_movie_ids:
        # get the feature vector (ratings) of the movie
        movie = movies_and_features.loc[movie_id, :]

        # Get k neighbors of the movie
        # Scores will be stored the nearest movies distances in ascending order 
        # And the indices will be the nearest movies indices founded by knn
        scores, indices = neigh.kneighbors([movie])

        print("recommended movie scores: %s " % (scores))

        indices = indices[0]

        # get closest movie ids from indices
        closest_movie_ids = movies_and_features.index.values[indices]
        
        # print closest movies
        print(movies[movies["movieId"].isin(closest_movie_ids)][["movieId", "title"]])
    else:
        print("movie_id", movie_id, "does not exist")