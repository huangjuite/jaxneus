
from typing import NamedTuple
import pickle
from jaxtyping import PyTree


class ModelState(NamedTuple):
    """Model parameters and optimizer state.

    Attributes
    ----------
    params: main model parameters
    opt_state: optimizer state for the main model
    """

    params: PyTree
    opt_state: PyTree

    def save(self, path: str) -> None:
        """Pickle checkpoint; avoid including `ModelState`."""
        with open(path, "wb") as f:
            pickle.dump((self.params, self.opt_state), f)

    @classmethod
    def load(cls, path: str) -> "ModelState":
        """Load checkpoint."""
        with open(path, "rb") as f:
            params, opt_state = pickle.load(f)
        return cls(params, opt_state)
