"""
Main function of the module to run the pipeline needed for recommendations.
"""
import logging
from argparse import Namespace
from pathlib import Path

from recommender.core import (
    PreProcessor,
    recommend_ub
)
from recommender.file_operations import (
    DataHandler,
    unpickle_object,
    pickle_object
)
from recommender.utils import (
    RecArgumentParser,
    InputArguments,
    setup_logging,
)

# Setup logging
logging_path: Path = Path("loggers")
setup_logging(logger_root_dir=logging_path,
              log_level="DEBUG")

logger = logging.getLogger(__name__)
logger.info("Main App starting...")


def main(input_args: InputArguments):
    """Main function to call all the function of the module needed."""
    logger.info(f"Input arguments: {input_args}")
    data_handler: DataHandler = DataHandler(
        input_folder=input_args.data_path,
    )

    # --- Calculate user neighbors ---
    # Try loading the pickle files if force read is not provided
    try:
        if input_args.force_calculate:
            raise FileNotFoundError()
        user_ratings = unpickle_object(Path("./data/models/user_ratings.pkl"))
        neighbors_u = unpickle_object(Path("./data/models/user_neighbors.pkl"))
    except FileNotFoundError:

        preprocessor = PreProcessor()
        user_ratings, neighbors_u = preprocessor.preprocess(data_handler.ratings_df)
        # Pickle the objects for future use
        pickle_object(user_ratings, Path("./data/models/user_ratings.pkl"))
        pickle_object(neighbors_u, Path("./data/models/user_neighbors.pkl"))

    # find recommended movies
    recommended_movies = recommend_ub(user=100,
                                      movies_df=data_handler.movies_df,
                                      neighbors_u=neighbors_u,
                                      user_ratings=user_ratings,
                                      neighbor_num=input_args.neighbors_num,
                                      rec_num=input_args.recommendations_num)

    for movie in recommended_movies:
        print(movie['title'])


if __name__ == "__main__":
    _args: Namespace = RecArgumentParser().parse()

    # Parse command-line arguments
    arguments: InputArguments = InputArguments.from_dict(vars(_args))

    main(arguments)
