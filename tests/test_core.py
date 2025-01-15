"""
Generated using ChatGPT
"""
import pytest
import pandas as pd


from recommender.core import PreProcessor, recommend_ub


@pytest.fixture
def sample_ratings_df():
    """Fixture to create a sample ratings DataFrame."""
    data = {
        'userId': [1, 1, 2, 2, 3, 3],
        'movieId': [101, 102, 101, 103, 102, 104],
        'rating': [4.5, 2.0, 3.5, 5.0, 2.0, 1.5]
    }
    return pd.DataFrame(data)


@pytest.fixture
def preprocessor():
    """Fixture to create a PreProcessor instance."""
    return PreProcessor(min_rating_num=1)


def test_discretize_rating():
    """Test the discretize_rating method."""
    preprocessor = PreProcessor()

    assert preprocessor.discretize_rating(4.5) == 'P'  # Positive
    assert preprocessor.discretize_rating(3.0) == 'A'  # Average
    assert preprocessor.discretize_rating(2.5) == 'N'  # Negative


def test_load_user_ratings(sample_ratings_df, preprocessor):
    """Test the _load_user_ratings method."""
    user_ratings = preprocessor._load_user_ratings(sample_ratings_df)

    assert isinstance(user_ratings, dict)
    assert len(user_ratings) == 3  # Three distinct users

    # Check ratings for user 1
    assert user_ratings[1][101] == 'P'  # Rating 4.5 -> 'P'
    assert user_ratings[1][102] == 'N'  # Rating 2.0 -> 'N'


def test_get_user_neighbors(sample_ratings_df, preprocessor):
    """Test the _get_user_neighbors method."""
    user_ratings = preprocessor._load_user_ratings(sample_ratings_df)
    neighbors = preprocessor._get_user_neighbors(user_ratings, 1)
    print(neighbors)
    assert isinstance(neighbors, dict)
    assert 1 in neighbors  # User 1 should have neighbors

    # Check that neighbors list for user 1 is sorted by similarity
    assert isinstance(neighbors[1], list)
    assert len(neighbors[1]) > 0  # Should have neighbors based on Jaccard similarity


def test_preprocess(sample_ratings_df, preprocessor):
    """Test the full preprocess method."""
    user_ratings, neighbors_u = preprocessor.preprocess(sample_ratings_df)

    assert isinstance(user_ratings, dict)
    assert isinstance(neighbors_u, dict)
    assert 1 in neighbors_u  # User 1 should have neighbors


def test_recommend_ub(sample_ratings_df):
    """Test the recommend_ub function."""
    # Sample movie DataFrame
    movies_data = {
        'movieId': [101, 102, 103, 104],
        'title': ['Movie A', 'Movie B', 'Movie C', 'Movie D'],
        'genres': ['Action', 'Comedy', 'Drama', 'Action']
    }
    movies_df = pd.DataFrame(movies_data).set_index('movieId')

    # Sample user ratings and neighbors
    user_ratings = {
        1: {101: 'P', 102: 'N'},
        2: {101: 'A', 103: 'P'},
        3: {102: 'P', 104: 'N'}
    }

    neighbors_u = {
        1: [(2, 0.5), (3, 0.8)],
        2: [(1, 0.5), (3, 0.9)],
        3: [(1, 0.8), (2, 0.9)]
    }

    # Test recommendation for user 1
    recommendations = recommend_ub(1, movies_df, neighbors_u, user_ratings, neighbor_num=2, rec_num=2)

    assert len(recommendations) == 2  # Should recommend 2 movies
    assert recommendations[0]['movieId'] == 103  # Movie D
    assert recommendations[0]['recommendedScore'] == 1.0  # Example score
    assert recommendations[1]['movieId'] == 104  # Movie C


# You can also include tests for edge cases such as:
# - No similar neighbors found
# - User has rated all movies
# - Test with empty DataFrame

@pytest.mark.parametrize(
    "rating, expected",
    [
        (5.0, 'P'),
        (3.0, 'A'),
        (1.0, 'N')
    ]
)
def test_discretize_rating_parametrize(rating, expected):
    """Test discretize_rating with parameterized values."""
    preprocessor = PreProcessor()
    assert preprocessor.discretize_rating(rating) == expected
