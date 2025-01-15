"""
File containing read operations.
"""
import logging
from functools import cached_property
from pathlib import Path
from typing import Optional
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
logger = logging.getLogger(__name__)


class DataHandler:
    """Class to read the required data for the recommender."""

    def __init__(self, input_folder: Path,
                 movies_file_name: str = 'movies.csv',
                 movies_index_col: Optional[str] = 'movieId',
                 ratings_file_name: str = 'ratings.csv',
                 ratings_index_col: Optional[str] = None):
        # Validate that the file is indeed a
        self.input_folder: Path = input_folder
        assert isinstance(self.input_folder,
                          Path), f"Expected 'input_folder' to be of type 'Path', but got {type(self.input_folder)}"
        self.movies_file_path: Path = self.input_folder / movies_file_name
        self.movies_index_col: str = movies_index_col
        self.ratings_file_path: Path = self.input_folder / ratings_file_name
        self.ratings_index_col: str = ratings_index_col

        # verify that the files exist
        for file in [self.movies_file_path, self.ratings_file_path]:
            logger.info(f"Validating file `{file}`")
            self._validate_file(file)

    @staticmethod
    def _validate_file(input_file: Path) -> None:
        """Validates the existence of a file. """
        if not input_file.is_file():
            logger.exception(f"The required file was not found: {str(input_file)}.")
            raise FileNotFoundError(f"The required file was not found: {str(input_file)}.")

    @cached_property
    def movies_df(self) -> pd.DataFrame:
        """Read and returns the `movie.csv` and `ratings.csv` in a pandas dataframe"""
        return pd.read_csv(self.movies_file_path, index_col=self.movies_index_col)

    @cached_property
    def ratings_df(self) -> pd.DataFrame:
        """Read and returns the `movie.csv` and `ratings.csv` in a pandas dataframe"""
        return pd.read_csv(self.ratings_file_path, index_col=self.ratings_index_col)


def pickle_object(obj: Any, filename: Path) -> None:
    """
    Pickle (serialize) an object and save it to a file.

    Args:
        obj (Any): The object to pickle (any Python object).
        filename (Path): The path to the file where the object will be saved.

    Returns:
        None: This function does not return a value.

    Raises:
        Exception: If there is an error during pickling.
    """
    try:
        filename = Path(filename)

        # create the parent directory
        filename.parent.mkdir(exist_ok=True)

        with open(filename, 'wb') as f:
            pickle.dump(obj, f)
        logger.info(f"Object has been pickled and saved to {str(filename)}")
    except Exception as e:
        logger.error(f"Error while pickling object: {e}")


def unpickle_object(filename: Path) -> Any:
    """
    Unpickle (deserialize) an object from a file.

    Args:
        filename (Path): The path to the file from which to load the object.

    Returns:
        Any: The deserialized Python object.

    Raises:
        Exception: If there is an error during unpickling.
    """
    try:
        filename = Path(filename)
        with open(filename, 'rb') as f:
            obj = pickle.load(f)
        logger.info(f"Object has been unpickled from {str(filename)}")
        return obj
    except FileNotFoundError:
        logger.error(f"Error while unpickling object: {e}")
        raise FileNotFoundError
