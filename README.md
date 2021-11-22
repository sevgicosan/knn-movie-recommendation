# Movie Recommendation

A movie recommendation approach based on user ratings of movies only. Movies are embedded in the rating space by creating vectors of ratings for each for movie by different users. KNN is applied to find closest movies (recommendations) to any given movie. You can find detailed descripton of the approach at [Movie Recommendation](http://sevgi.me/movie-recommendation).

[MovieLens database](https://grouplens.org/datasets/movielens/) is used during training in which there are >60k movies, and >25m user ratings.

## Using the Code
To use the code, download [ml-25m.zip](https://files.grouplens.org/datasets/movielens/ml-25m.zip) and replace ratings and movies paths in `knn-movie-recommendation.py`.