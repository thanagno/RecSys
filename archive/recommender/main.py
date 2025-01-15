import pandas as pd

from helper_functions import (
    load_user_ratings,
    get_user_neighbors,
    recommend_ub
)

def main():
    input_folder = '../../data/ml-latest-small'

    # load datasets
    movies_df = pd.read_csv(input_folder + '/movies.csv', index_col='movieId')
    tags_df = pd.read_csv(input_folder + '/tags.csv')
    ratings_df = pd.read_csv(input_folder + '/ratings.csv')

    # calculate the user ratings and neighbors
    user_ratings = load_user_ratings(ratings_df)

    neighbors_u = get_user_neighbors(user_ratings)

    # recommend a movie
    recommend_ub(100, movies_df, neighbors_u, user_ratings, 10, 10)


if __name__ == "__main__":
    main()
