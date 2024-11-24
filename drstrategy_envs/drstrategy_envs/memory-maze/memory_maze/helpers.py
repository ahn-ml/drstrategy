from dm_env.specs import BoundedArray, DiscreteArray
import numpy as np

def sample_spec(space: BoundedArray) -> np.ndarray:
    if isinstance(space, DiscreteArray):
        return np.random.randint(space.num_values, size=space.shape)

    if isinstance(space, BoundedArray):
        return np.random.uniform(space.minimum, space.maximum, size=space.shape)

    raise NotImplementedError

# https://github.com/jurgisp/pydreamer/blob/8a4ea85a925b57ddf24a221b41dfb3257f89d63a/pydreamer/preprocessing.py#L10
def to_onehot(x: np.ndarray, n_categories: int) -> np.ndarray:
    e = np.eye(n_categories, dtype=np.float32)
    return e[x]  # Nice trick: https://stackoverflow.com/a/37323404
