import random
import numpy as np
try:  # torch may be unavailable or broken in some environments
    import torch  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    torch = None


def set_seed(seed: int) -> None:
    """Seed random number generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if hasattr(torch, "cuda"):
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
