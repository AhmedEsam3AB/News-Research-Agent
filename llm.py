# Lets the `str | None` annotations below work on Python 3.9 as well.
from __future__ import annotations

import json
import os
import re

from langchain_core.messages import HumanMessage, convert_to_messages
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import ChatOpenAI
from openai import APIError
from pydantic import ValidationError

# OpenRouter exposes an OpenAI-compatible API, so ChatOpenAI works against it
# as long as we point base_url at OpenRouter and use an OpenRouter key.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Any OpenRouter model slug, e.g. "anthropic/claude-sonnet-4.5",
# "google/gemini-2.5-pro", "meta-llama/llama-3.3-70b-instruct".
DEFAULT_MODEL = "openai/gpt-4o"


def get_llm(temperature: float = 0.7, model: str | None = None) -> ChatOpenAI:
    """Build a ChatOpenAI client pointed at OpenRouter."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Add it to your .env file "
            "(get a key at https://openrouter.ai/keys)."
        )

    # Optional attribution headers; OpenRouter uses them for its app rankings.
    default_headers = {}
    if os.getenv("OPENROUTER_SITE_URL"):
        default_headers["HTTP-Referer"] = os.getenv("OPENROUTER_SITE_URL")
    if os.getenv("OPENROUTER_APP_NAME"):
        default_headers["X-Title"] = os.getenv("OPENROUTER_APP_NAME")

    # Free OpenRouter accounts cannot afford a model's default max_tokens, so
    # allow capping it from the environment.
    max_tokens = os.getenv("OPENROUTER_MAX_TOKENS")

    return ChatOpenAI(
        model=model or os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL),
        temperature=temperature,
        max_tokens=int(max_tokens) if max_tokens else None,
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        default_headers=default_headers or None,
    )


# Models found to lack native structured output, so we stop paying for the
# doomed first attempt on every later call.
_NO_NATIVE_STRUCTURED: set[str] = set()


def _json_schema_of(schema) -> dict:
    """JSON schema for a Pydantic model or a TypedDict."""
    return convert_to_openai_tool(schema)["function"]["parameters"]


def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model response.

    Models without structured-output support wrap JSON in prose or code fences,
    or ignore the instruction entirely.
    """
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)

    decoder = json.JSONDecoder()
    for start in (m.start() for m in re.finditer(r"[\[{]", text)):
        try:
            value, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError(f"No JSON object found in model response: {text[:500]!r}")


def with_schema(llm: ChatOpenAI, schema) -> Runnable:
    """`with_structured_output`, but tolerant of models that can't do it natively.

    Many OpenRouter models have no tool-calling support and ignore
    `response_format`. For those, fall back to asking for JSON in the prompt and
    parsing it out of the reply, with one corrective retry.
    """
    native = llm.with_structured_output(schema)
    schema_json = json.dumps(_json_schema_of(schema), indent=2)
    is_pydantic = hasattr(schema, "model_validate")

    instruction = (
        "Reply with a single JSON object matching this JSON schema. Output raw "
        "JSON only: no prose, no explanation, no headings, no markdown code "
        f"fences. Start your reply with the opening brace.\n\n{schema_json}"
    )

    def retry_message(error: Exception) -> str:
        return (
            f"That reply could not be parsed. The error was:\n{error}\n\n"
            "Output ONLY the JSON object matching the schema - nothing before "
            "it and nothing after it. Escape any double quotes that appear "
            'inside string values as \\" so the JSON stays valid.'
        )

    def _coerce(text: str):
        data = _extract_json(text)
        return schema.model_validate(data) if is_pydantic else data

    def _run(prompt_value, config=None):
        # 1. Native structured output, unless this model already proved it can't.
        if llm.model_name not in _NO_NATIVE_STRUCTURED:
            try:
                result = native.invoke(prompt_value, config=config)
                if result is not None:
                    return result
            except APIError:
                # Billing, rate-limit or transport failure - says nothing about
                # whether the model supports schemas, and the prompt fallback
                # would just burn more doomed requests.
                raise
            except Exception:
                pass
            # Silent None or an outright error both mean: don't try this again.
            _NO_NATIVE_STRUCTURED.add(llm.model_name)

        # 2. Ask for JSON in the prompt and parse it back out.
        if hasattr(prompt_value, "to_messages"):
            messages = list(prompt_value.to_messages())
        else:
            messages = list(convert_to_messages(prompt_value))
        messages.append(HumanMessage(content=instruction))

        last_error: Exception | None = None
        for _ in range(3):
            reply = llm.invoke(messages, config=config)
            try:
                return _coerce(reply.text)
            except (ValueError, ValidationError) as e:
                last_error = e
                messages = [*messages, reply, HumanMessage(content=retry_message(e))]

        raise ValueError(
            f"{llm.model_name} could not produce JSON matching the schema. "
            f"Last error: {last_error}"
        )

    return RunnableLambda(_run)
