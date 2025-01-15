from collections import defaultdict
from itertools import combinations
from pprint import pprint

import pandas as pd


def discretize_rating(rating: float):
    """
    Converts a given float rating to a string value
    """
    polarity = 'A'  # average

    if rating < 3:
        polarity = 'N'  # negative
    elif rating > 3:
        polarity = 'P'  # positive

    return polarity


def load_user_ratings(ratings_df: pd.DataFrame):
    """
    Loads all the ratings submitted by each user
    Returns a dictionary that maps each user to a second dict that maps movies to discretized ratings
    """

    distinct_users = set(ratings_df['userId'])  # get all distinct users

    user_ratings = {}  # store movie ratings per user

    for user in distinct_users:  # for each user

        # get the movie id and rating for every rating submitted by this user
        my_ratings = ratings_df[ratings_df.userId == user][['movieId', 'rating']]

        # discretize the ratings and attach them to the user
        user_ratings[user] = dict(zip(my_ratings.movieId, my_ratings.rating.apply(discretize_rating)))

    return user_ratings


def get_user_neighbors(user_ratings: dict,  # ratings submitted by each user
                       min_rating_num: int = 5  # at least this many ratings are required for a comparison
                       ):
    '''
    Compute rating-based similarity between every two pairs of users

    '''

    # get all possible pairs of usres
    pairs = list(combinations(list(user_ratings.keys()), 2))

    usim = defaultdict(dict)  # initialize the sim dictionary

    for u1, u2 in pairs:  # for every user pair

        # get a set with all the discretized ratings (movie id, polarity tuples) for u1 and u2
        s1 = set([(mid, pol) for mid, pol in user_ratings[u1].items()])
        s2 = set([(mid, pol) for mid, pol in user_ratings[u2].items()])

        # check if both users respect the lower bound
        if len(s1) < min_rating_num or len(s2) < min_rating_num: continue

        # get the union and intersection for these two users
        union = s1.union(s2)
        inter = s1.intersection(s2)

        # compute user sim via the jaccard coeff
        jacc = len(inter) / len(union)

        # remember the sim values
        if jacc > 0:
            usim[u1][u2] = jacc
            usim[u2][u1] = jacc

    # attach each user to its neighbors, sorted by sim in descending order
    return {user: sorted(usim[user].items(), key=lambda x: x[1], reverse=True) for user in usim}


def recommend_ub(user: int,
                 movies_df: pd.DataFrame,  # movie info
                 neighbors_u: dict,  # neighbors dict
                 user_ratings: dict,  # ratings submitted per user
                 neighbor_num: int,  # number of neighbors to consider
                 rec_num: int  # number of movies to recommend
                 ):
    """
    Delivers user-based recommendations. Given a specific user:
    - find the user's neighbor_num most similar users
    - Go over all the movies rated by all neighbors
    - Each movie gets +2 if a neighbor liked it, -2 if a neighbor didn't like it, -1 if  neighbor was neutral
    - +2,-1,and -2 are scaled based on user sim
    - Sort the movies by their scores in desc order
    - Go over the sorted movie list. If the user has already rated the movie, store its rating. Otherwise print.

    """

    top_k = neighbors_u[user][:neighbor_num]  # get the top k neighbors of this user

    votes = defaultdict(int)  # count the votes per movie

    for neighbor, sim_val in top_k:  # for each neighbor

        for mid, pol in user_ratings[neighbor].items():  # for each movie rated by this neighbor

            if pol == 'P':  # positive neighbor rating
                votes[mid] += 2 * sim_val
            elif pol == 'N':  # negative
                votes[mid] -= 2 * sim_val
            else:  # average
                votes[mid] -= 1 * sim_val

    # sort the movies in desc order
    srt = sorted(votes.items(), key=lambda x: x[1], reverse=True)

    print('\nI suggest the following movies because they have\
    received positive ratings from users who tend to\nlike what you like:\n')

    cnt = 0  # count number of recommendations made

    already_rated = {}

    for mov, score in srt:  # for each movie

        title = movies_df.loc[mov]['title']  # get the title

        rat = user_ratings[user].get(mov, None)  # check if the user has already rated the movie

        if rat:  # movie already rated
            already_rated[title] = rat  # store the rating
            continue

        cnt += 1  # one more recommendation
        print(mov, title)  # print

        if cnt == rec_num: break  # stop once you 've made enough recommendations

    print("\n"* 2, "Movies already rated:")
    pprint(already_rated)
