import logging


def setup_logging(level: int = logging.INFO) -> None:
    """Configure basic logging format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
