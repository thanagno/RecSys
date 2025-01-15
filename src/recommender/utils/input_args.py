import inspect
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Union


class RecArgumentParser:
    """Parse the input arguments."""
    parser: ArgumentParser = ArgumentParser()

    def parse(self) -> Namespace:
        self.parser.add_argument('--data-path', '-d',
                                 type=str,
                                 help='Path where the `.csv` files are located. Can be relative or absolute.',
                                 required=False,
                                 default='./data/ml-latest-small')
        self.parser.add_argument('--neighbors-num', '-n',
                                 type=int,
                                 help='Number of neighbors to consider for the recommendation.',
                                 required=False,
                                 default=10)
        self.parser.add_argument('--recommendations-num', '-r',
                                 type=int,
                                 help='Number of neighbors to consider for the recommendation.',
                                 required=False,
                                 default=10)
        self.parser.add_argument('--force-calculate', '-f',
                                 action='store_true',
                                 help='Flag to force calculate the ratings and neighbors even if the pre-calculated '
                                      'files exist',
                                 required=False,
                                 default=False)
        return self.parser.parse_args()


@dataclass
class InputArguments:
    data_path: Union[str | Path]
    neighbors_num: int
    recommendations_num: int
    force_calculate: bool

    def __post_init__(self):
        """Convert data_path to a Path object."""
        if isinstance(self.data_path, str):
            self.data_path: Path = Path(self.data_path)
            if not self.data_path.is_dir():
                raise ValueError(f"The specified input path is not a valid path: `{self.data_path}`")

    @classmethod
    def from_dict(cls, env):
        """Allows the definition of the class even if extra arguments are provided in the input"""
        return cls(**{
            k: v for k, v in env.items()
            if k in inspect.signature(cls).parameters
        })
