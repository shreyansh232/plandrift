"""LLM API client wrapper with structured output support.

Uses OpenRouter as the provider, which is OpenAI-compatible.
"""

import json
import os
import time
from typing import Any, Callable, Optional, Type, TypeVar

from openai import OpenAI, RateLimitError
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# Default: Gemini 3 Flash for the hackathon. Tool calling is disabled via fallback.
DEFAULT_MODEL = os.environ.get(
    "OPENROUTER_MODEL", "google/gemini-3-flash-preview"
)
FAST_MODEL = os.environ.get("OPENROUTER_MODEL_FAST")


class AIClient:
    """Wrapper for OpenRouter (OpenAI-compatible) API with structured output parsing."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
    ):
        """Initialize the client.

        Args:
            api_key: OpenRouter API key. If not provided, uses OPENROUTER_API_KEY env var.
            model: Model to use for completions.
        """
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key is required. Set OPENROUTER_API_KEY environment "
                "variable or pass api_key parameter."
            )
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=OPENROUTER_BASE_URL,
        )
        self.model = model

    def _create_completion_with_retry(self, **kwargs):
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                return self.client.chat.completions.create(**kwargs)
            except RateLimitError as err:
                last_error = err
                # Exponential backoff for upstream rate limits.
                time.sleep(1.5 * (2**attempt))
        if last_error:
            raise last_error
        raise RateLimitError("Upstream rate limit")

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Send a chat completion request and return the response text.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            temperature: Sampling temperature (0-2).
            max_tokens: Maximum tokens in response.

        Returns:
            The assistant's response text.
        """
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        response = self._create_completion_with_retry(**kwargs)
        if not response.choices:
            raise ValueError("Empty response from API — model returned no choices.")
        return response.choices[0].message.content or ""

    def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        tool_executor: Callable[[str, dict[str, Any]], str],
        temperature: float = 0.7,
        max_tool_calls: int = 5,
        on_tool_call: Optional[Callable[[str, dict], None]] = None,
    ) -> str:
        """Send a chat request with tool support, automatically executing tools.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            tools: List of tool definitions for OpenAI.
            tool_executor: Function that executes tools and returns results.
            temperature: Sampling temperature (0-2).
            max_tool_calls: Maximum number of tool calls to allow.
            on_tool_call: Optional callback when a tool is called (for UI updates).

        Returns:
            The final assistant response text.
        """
        # Gemini 3 tool-calls via OpenRouter can error ("Thought signature not valid").
        # Fallback to manual search query generation when tools are unsupported.
        if "gemini-3" in self.model.lower():
            return self._chat_with_tools_fallback(
                messages=messages,
                tool_executor=tool_executor,
                temperature=temperature,
                max_tool_calls=max_tool_calls,
                on_tool_call=on_tool_call,
            )

        messages = messages.copy()
        tool_calls_made = 0

        while tool_calls_made < max_tool_calls:
            response = self._create_completion_with_retry(
                model=self.model,
                messages=messages,
                tools=tools,
                temperature=temperature,
            )

            if not response.choices:
                raise ValueError("Empty response from API — model returned no choices.")

            message = response.choices[0].message

            # If no tool calls, return the content
            if not message.tool_calls:
                return message.content or ""

            # Add the assistant message with tool calls
            messages.append(message.model_dump())

            # Execute each tool call
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                try:
                    arguments = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    arguments = {}

                # Notify UI if callback provided
                if on_tool_call:
                    on_tool_call(tool_name, arguments)

                # Execute the tool
                result = tool_executor(tool_name, arguments)

                # Add tool result to messages
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    }
                )
                tool_calls_made += 1

        # If we hit max tool calls, get final response without tools
        response = self._create_completion_with_retry(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        if not response.choices:
            raise ValueError("Empty response from API — model returned no choices.")
        return response.choices[0].message.content or ""

    def _chat_with_tools_fallback(
        self,
        messages: list[dict],
        tool_executor: Callable[[str, dict[str, Any]], str],
        temperature: float = 0.7,
        max_tool_calls: int = 5,
        on_tool_call: Optional[Callable[[str, dict], None]] = None,
    ) -> str:
        """Fallback path for models that do not support tool calls.

        Asks the model for search queries, runs web_search directly, then
        appends results and continues the chat without tools.
        """
        convo = "\n".join(
            f"{m.get('role','user')}: {m.get('content','')}" for m in messages
        )
        query_prompt = f"""Given the conversation below, list up to {max_tool_calls} web search queries needed to answer accurately.
One query per line. If no search is needed, respond with "NONE".

Conversation:
{convo}
"""
        query_text = self.chat(
            [
                {"role": "system", "content": "You generate search queries only."},
                {"role": "user", "content": query_prompt},
            ],
            temperature=0.2,
        )
        raw_lines = [line.strip("•- \t") for line in query_text.splitlines()]
        queries = [line for line in raw_lines if line and line.upper() != "NONE"]
        queries = queries[:max_tool_calls]

        results: list[str] = []
        for query in queries:
            if on_tool_call:
                on_tool_call("web_search", {"query": query})
            result = tool_executor("web_search", {"query": query})
            results.append(result)

        messages = messages.copy()
        if results:
            messages.append(
                {
                    "role": "system",
                    "content": "Research results:\n" + "\n\n".join(results),
                }
            )

        return self.chat(messages, temperature=temperature)

    def _build_example(self, schema: dict) -> dict:
        """Build a minimal example object from a JSON schema."""
        props = schema.get("properties", {})
        defs = schema.get("$defs", {})
        example: dict = {}

        for name, prop in props.items():
            example[name] = self._example_value(name, prop, defs)

        return example

    def _example_value(self, name: str, prop: dict, defs: dict) -> Any:
        """Generate a placeholder value for a schema property."""
        # Handle $ref
        if "$ref" in prop:
            ref_name = prop["$ref"].split("/")[-1]
            if ref_name in defs:
                ref_schema = defs[ref_name]
                # Enum
                if "enum" in ref_schema:
                    return ref_schema["enum"][0]
                return self._build_example(ref_schema)
            return "..."

        # Handle anyOf (Optional fields)
        if "anyOf" in prop:
            for option in prop["anyOf"]:
                if option.get("type") != "null":
                    return self._example_value(name, option, defs)
            return None

        t = prop.get("type", "string")
        desc = prop.get("description", "")
        if t == "string":
            # Use description as hint if available
            if desc:
                return f"<{desc}>"
            return f"<{name}>"
        elif t == "integer":
            return 0
        elif t == "number":
            return 0.0
        elif t == "boolean":
            return True
        elif t == "array":
            item_schema = prop.get("items", {"type": "string"})
            return [self._example_value("item", item_schema, defs)]
        elif t == "object":
            return self._build_example(prop)
        return "..."

    def chat_structured(
        self,
        messages: list[dict],
        response_format: Type[T],
        temperature: float = 0.7,
        max_retries: int = 1,
        include_schema: bool = True,
        include_example: bool = True,
    ) -> T:
        """Send a chat request and parse response into a Pydantic model.

        Shows the model an example of the expected JSON shape and asks it
        to fill in real values. Works with any OpenAI-compatible provider.
        Retries with error feedback if validation fails.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            response_format: Pydantic model class for the response.
            temperature: Sampling temperature (0-2).
            max_retries: Number of retries on validation failure.

        Returns:
            Parsed response as the specified Pydantic model.
        """
        schema = response_format.model_json_schema()
        example = self._build_example(schema) if include_example else None

        example_block = (
            "Expected structure:\n" + json.dumps(example, indent=2) + "\n\n"
            if include_example
            else ""
        )
        schema_block = (
            "Full JSON schema for reference:\n" + json.dumps(schema, indent=2) + "\n\n"
            if include_schema
            else ""
        )
        schema_instruction = (
            "Respond with a JSON object using EXACTLY the structure below. "
            "Fill in real values instead of placeholders.\n\n"
            "CRITICAL: Every nested object MUST remain an object with its own keys. "
            "Do NOT flatten objects into strings. For example, if the schema shows "
            'an array of objects like [{"activity": "...", "cost_estimate": "..."}], '
            "each element MUST be an object with those keys, NOT a plain string.\n\n"
            f"{example_block}{schema_block}"
            "Return ONLY the JSON object. No markdown, no explanation."
        )

        augmented_messages = messages.copy()
        augmented_messages.append({
            "role": "user",
            "content": schema_instruction,
        })

        last_error: Optional[Exception] = None
        for attempt in range(1 + max_retries):
            response = self._create_completion_with_retry(
                model=self.model,
                messages=augmented_messages,
                response_format={"type": "json_object"},
                temperature=temperature,
            )

            if not response.choices:
                error_detail = (
                    getattr(response, "error", None) or "No response from model"
                )
                raise ValueError(f"Empty response from API: {error_detail}")

            content = response.choices[0].message.content or ""

            try:
                data = json.loads(content)
                return response_format.model_validate(data)
            except (json.JSONDecodeError, Exception) as e:
                last_error = e
                if attempt < max_retries:
                    # Feed the error back to the model for a retry
                    error_feedback = (
                        f"Your previous JSON response had validation errors:\n"
                        f"{str(e)[:1000]}\n\n"
                        "Please fix these errors. Remember:\n"
                        "- Array items that should be objects MUST be objects "
                        "(with their own keys), NOT plain strings.\n"
                        "- Follow the exact structure from the schema.\n\n"
                        "Return ONLY the corrected JSON object."
                    )
                    augmented_messages.append({
                        "role": "assistant",
                        "content": content,
                    })
                    augmented_messages.append({
                        "role": "user",
                        "content": error_feedback,
                    })

        raise ValueError(
            f"Failed to parse structured response after {1 + max_retries} "
            f"attempt(s): {last_error}\nRaw content: {content[:500]}"
        )
