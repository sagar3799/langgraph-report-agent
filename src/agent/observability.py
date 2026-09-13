"""Optional Langfuse tracing for graph invocations.

Langfuse's LangGraph integration needs a callback handler passed into each invoke() call
(unlike LangSmith's pure-env-var auto-instrumentation). Graceful no-op when LANGFUSE_PUBLIC_KEY
isn't set, so local dev/tests never need a Langfuse account.
"""

import logging
import os

logger = logging.getLogger(__name__)

_handler = None
_handler_checked = False


def get_invoke_config(**extra_metadata) -> dict:
    """Build the `config=` kwarg for graph.invoke(). Empty dict if Langfuse isn't configured.

    extra_metadata is merged into the trace's metadata (e.g. loop_count, confidence) so you
    can filter runs in the Langfuse dashboard, not just see that tracing exists.
    """
    handler = _get_handler()
    if handler is None:
        return {}
    config: dict = {"callbacks": [handler]}
    if extra_metadata:
        config["metadata"] = extra_metadata
    return config


def _get_handler():
    global _handler, _handler_checked
    if _handler_checked:
        return _handler
    _handler_checked = True

    if not os.environ.get("LANGFUSE_PUBLIC_KEY"):
        logger.info("LANGFUSE_PUBLIC_KEY not set — Langfuse tracing disabled")
        return None

    try:
        from langfuse.langchain import CallbackHandler

        _handler = CallbackHandler()
        logger.info("Langfuse tracing enabled")
    except Exception as exc:  # missing package, bad keys, network issue -- never break the app
        logger.warning("Langfuse tracing unavailable, continuing without it: %s", exc)
        _handler = None

    return _handler
