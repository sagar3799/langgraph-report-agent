"""Gemini chat model + structured-output helper with an explicit retry-once fallback.

Gemini's JSON mode is reliable but not perfect (truncated output, an extra field,
a string where a number was expected). Rather than trust it blindly, we validate
against the Pydantic schema and, on failure, retry exactly once with the validation
error fed back to the model. If that also fails we raise so the caller can decide
what a safe degraded response looks like.
"""

import json
import os
from typing import TypeVar

from google.api_core.exceptions import GoogleAPIError
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, ValidationError

GEMINI_MODEL = "gemini-flash-lite-latest"  # far higher free-tier daily quota than -latest

T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(RuntimeError):
    """Raised when the model still fails to produce valid, schema-matching JSON after a retry."""


def get_chat_model(temperature: float = 0.0) -> ChatGoogleGenerativeAI:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set in the environment (.env)")
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL, google_api_key=api_key, temperature=temperature
    )


def _content_to_text(content: str | list) -> str:
    """Newer langchain message content can be a list of content blocks instead of a plain string."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


def _extract_json(raw: str | list) -> dict:
    text = _content_to_text(raw).strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def generate_structured(
    llm: ChatGoogleGenerativeAI,
    system_prompt: str,
    user_prompt: str,
    schema: type[T],
) -> T:
    """Call the LLM and parse+validate its JSON reply against `schema`, retrying once on failure."""
    schema_hint = json.dumps(schema.model_json_schema(), indent=2)
    messages: list[BaseMessage] = [
        SystemMessage(
            content=(
                f"{system_prompt}\n\n"
                "Respond with ONLY valid JSON matching this schema, no markdown fences, "
                f"no commentary:\n{schema_hint}"
            )
        ),
        HumanMessage(content=user_prompt),
    ]

    last_error: Exception | None = None
    for _attempt in range(2):
        try:
            response = llm.invoke(messages)
            data = _extract_json(response.content)
            return schema.model_validate(data)
        except (json.JSONDecodeError, ValidationError, GoogleAPIError) as exc:
            last_error = exc
            messages.append(HumanMessage(
                content=(
                    "Your previous response was invalid: "
                    f"{exc}\nReturn ONLY corrected JSON matching the schema, nothing else."
                )
            ))

    raise StructuredOutputError(
        f"Model failed to produce valid {schema.__name__} JSON after retry: {last_error}"
    )
