"""Shared logging setup so every entry point (Streamlit, FastAPI, scripts, eval) shows
the same readable trace of what the agent is actually doing — retrieval, grading, tool
calls — instead of running silently.
"""

import logging

_NOISY_LIBRARIES = ("httpx", "httpcore", "urllib3", "qdrant_client", "google_genai")


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for name in _NOISY_LIBRARIES:
        logging.getLogger(name).setLevel(logging.WARNING)
