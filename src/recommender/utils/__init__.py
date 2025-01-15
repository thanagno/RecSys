"""
Utils folder for the recommender system app.
"""
from recommender.utils.input_args import (
    RecArgumentParser,
    InputArguments,
)

from recommender.utils.logger_config import (
    setup_logging
)

__all__ = [
    "RecArgumentParser",
    "InputArguments",
    "setup_logging",
]
