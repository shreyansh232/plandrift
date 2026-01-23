"""OpenAI API client wrapper with structured output support."""

import json
import os
from typing import Any, Callable, Optional, Type, TypeVar

from openai import OpenAI
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class OpenAIClient:
    """Wrapper for OpenAI API with structured output parsing."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        """Initialize the OpenAI client.

        Args:
            api_key: OpenAI API key. If not provided, uses OPENAI_API_KEY env var.
            model: Model to use for completions.
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        self.client = OpenAI(api_key=self.api_key)
        self.model = model

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

        response = self.client.chat.completions.create(**kwargs)
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
        messages = messages.copy()
        tool_calls_made = 0

        while tool_calls_made < max_tool_calls:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                temperature=temperature,
            )

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
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""

    def chat_structured(
        self,
        messages: list[dict],
        response_format: Type[T],
        temperature: float = 0.7,
    ) -> T:
        """Send a chat request and parse response into a Pydantic model.

        Uses OpenAI's structured output feature for reliable parsing.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            response_format: Pydantic model class for the response.
            temperature: Sampling temperature (0-2).

        Returns:
            Parsed response as the specified Pydantic model.
        """
        response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=messages,
            response_format=response_format,
            temperature=temperature,
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise ValueError("Failed to parse structured response from OpenAI")
        return parsed
