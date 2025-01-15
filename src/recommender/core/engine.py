import logging
from collections import defaultdict
from itertools import combinations
from typing import Callable, Dict, Set, List, Tuple, Optional

import pandas as pd
from tqdm import tqdm
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)


class MovieRecommendation(TypedDict):
    """TypedDict for movie ratings."""
    movieId: int
    title: str
    movie_genres: str
    recommendedScore: float


class PreProcessor:
    """A class used to preprocess movie ratings data."""

    def __init__(self,
                 min_rating_num: int = 5,
                 discretize_func: str = 'discretize_rating'):
        """
        Args:
            min_rating_num (int): Minimum number of ratings required for a comparison.
            discretize_func (str): The name of the function to be used for discetizing the ratings
        """
        self.min_rating_num: int = min_rating_num
        self.discretize_func = discretize_func

    @staticmethod
    def discretize_rating(rating: float) -> str:
        """
        Converts a given float rating to a string value representing sentiment using N, A and P polarities.
        """
        if rating < 3:
            return 'N'  # Negative
        elif rating > 3:
            return 'P'  # Positive
        return 'A'  # Average

    def _load_user_ratings(self,
                           ratings_df: pd.DataFrame,
                           discretize_func: str = 'discretize_rating'
                           ) -> Dict[int, Dict[int, str]]:
        """
        Loads all the ratings submitted by each user and discretizes them.

        Args:
            discretize_func (str): The name of the function used to discretize ratings.
            ratings_df: The DataFrame containing columns 'userId', 'movieId', 'rating'.

        Returns:
            Dict[int, MovieRatings]: A dictionary mapping each user to a dictionary of movie IDs and
                their discretized ratings.
        """
        logger.info("Processing User Ratings:")
        try:
            discretize_function: Callable[[float], str] = getattr(self, discretize_func)
        except AttributeError:
            error_msg = f"The provided `discretize_func` for the user ratings is not supported: {discretize_func}"
            logger.error(error_msg)
            raise AttributeError(error_msg)

        distinct_users: Set[int] = set(ratings_df['userId'])

        user_ratings: Dict[int, Dict[int, str]] = {}

        # load and discretize ratings
        for userId in tqdm(distinct_users, desc="Loading User Ratings..."):
            my_ratings = ratings_df[ratings_df['userId'] == userId][['movieId', 'rating']]
            user_ratings[userId] = dict(zip(my_ratings['movieId'], my_ratings['rating'].apply(discretize_function)))

        return user_ratings

    @staticmethod
    def _get_user_neighbors(user_ratings: Dict[int, Dict[int, str]],
                            min_rating_num: int = 5
                            ) -> Dict[int, List[Tuple[int, float]]]:
        """
        Compute rating-based similarity between every two pairs of users using Jaccard coefficient.

        Args:
            user_ratings (Dict[int, MovieRatings]): Ratings submitted by each user.
            min_rating_num (int): Minimum number of ratings required for a comparison.

        Returns:
            Dict[int, List[Tuple[int, float]]]: A dictionary mapping each user to a list of tuples containing neighbor
                user IDs and their similarity scores.
        """
        logger.info("Calculating user Neighbors")
        pairs = list(combinations(list(user_ratings.keys()), 2))
        usim = defaultdict(dict)

        for u1, u2 in tqdm(pairs, desc="Calculating User Neighbors"):
            s1 = set(user_ratings[u1].items())
            s2 = set(user_ratings[u2].items())

            if len(s1) < min_rating_num or len(s2) < min_rating_num:
                continue

            union = s1.union(s2)
            inter = s1.intersection(s2)

            jacc = len(inter) / len(union)

            if jacc > 0:
                usim[u1][u2] = jacc
                usim[u2][u1] = jacc

        return {user: sorted(usim[user].items(), key=lambda x: x[1], reverse=True) for user in usim}

    def preprocess(self,
                   ratings_df: pd.DataFrame
                   ) -> Tuple[Dict[int, Dict[int, str]], Dict[int, List[Tuple[int, float]]]]:
        """
        Calculate neighbors for each user based on their rating similarities.

        Args:
            ratings_df (pd.DataFrame): The DataFrame containing columns 'userId', 'movieId', 'rating'.

        Returns:
            A Tuple of:
            - Dict[int, MovieRatings]: A dictionary mapping each user to a dictionary of movie IDs and
                their discretized ratings.
            - Dict[int, List[Tuple[int, float]]]: A dictionary mapping each user to a list of tuples containing neighbor
                user IDs and their similarity scores.
        """
        # process user ratings and calculate neighbors
        user_ratings = self._load_user_ratings(ratings_df)
        neighbors_u = self._get_user_neighbors(user_ratings=user_ratings,
                                               min_rating_num=self.min_rating_num)
        return user_ratings, neighbors_u


def recommend_ub(user: int,
                 movies_df: pd.DataFrame,
                 neighbors_u: Dict[int, List[Tuple[int, float]]],
                 user_ratings: Dict[int, Dict[int, str]],
                 neighbor_num: int,
                 rec_num: int
                 ) -> List[MovieRecommendation]:
    """
    Delivers user-based recommendations. Given a specific user:
    - Find the user's `neighbor_num` most similar users.
    - Go over all the movies rated by all neighbors.
    - Each movie gets a weighted score based on neighbor ratings and similarity:
      +2 for positive, -2 for negative, -1 for neutral (scaled by similarity).
    - Sort movies by their scores in descending order.
    - Recommend movies not already rated by the user.
    - Print the recommendations and list movies already rated by the user.

    Args:
        user (int): User ID for whom recommendations are being generated.
        movies_df (pd.DataFrame): DataFrame containing movie details with movie IDs as the index.
        neighbors_u (Dict[int, List[Tuple[int, float]]]): Dictionary mapping user IDs
            to a list of (neighbor ID, similarity) tuples.
        user_ratings (dict): Dictionary mapping user IDs to their movie ratings {movie_id: 'P'/'N'/'A'}.
        neighbor_num (int): Number of most similar neighbors to consider.
        rec_num (int): Number of recommendations to make.

    Returns:
        None
    """
    logger.info(f"Calculating recommended movies for user: {user}")

    top_k: List[Tuple[int, float]] = neighbors_u.get(user, [])[:neighbor_num]  # get the top k neighbors of this user
    if not top_k:
        logger.warning(f"No similar neighbors found for user {user}")
        return []

    votes = defaultdict(int)  # count the votes per movie

    for neighbor, sim_val in top_k:  # for each neighbor
        logger.debug(f"Processing neighbor {neighbor} with similarity {sim_val:.2f}")
        for mid, pol in user_ratings[neighbor].items():  # for each movie rated by this neighbor

            if pol == 'P':  # positive neighbor rating
                votes[mid] += 2 * sim_val
            elif pol == 'N':  # negative
                votes[mid] -= 2 * sim_val
            else:  # average
                votes[mid] -= 1 * sim_val

    # sort the movies in desc order
    srt: List[Tuple[int, float]] = sorted(votes.items(), key=lambda x: x[1], reverse=True)

    cnt = 0  # count number of recommendations made

    already_rated = defaultdict(str)
    recommendations: List[MovieRecommendation] = []

    for mov, score in srt:  # for each movie

        movie: pd.Series = movies_df.loc[mov]  # get the movie
        title: str = movie['title']
        genres: str = movie['genres']

        rating: Optional[str] = user_ratings[user].get(mov, None)  # check if the user has already rated the movie

        if rating:  # movie already rated
            already_rated[title] = rating  # store the rating
            continue

        cnt += 1  # one more recommendation
        recommendations.append(
            MovieRecommendation(movieId=mov,
                                title=title,
                                movie_genres=genres,
                                recommendedScore=round(score, 3),

                                )
        )

        if cnt == rec_num:
            break  # stop once you 've made enough recommendations

    # Log the results
    logger.debug(dict(already_rated))
    logger.info(recommendations)

    return recommendations
