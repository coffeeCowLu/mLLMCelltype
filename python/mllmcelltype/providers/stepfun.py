"""StepFun provider module for LLMCellType."""

from __future__ import annotations

import requests

from ..logger import write_log
from .common import (
    build_chat_completions_body,
    call_openai_compatible_api,
    ensure_api_key,
    resolve_endpoint_url,
)


def process_stepfun(
    prompt: str, model: str, api_key: str, base_url: str | None = None
) -> list[str]:
    """Process request using StepFun models.

    Args:
        prompt: The prompt to send to the API
        model: The model name (e.g., 'step-3.5-flash', 'step-3.5-flash-2603', 'step-3')
        api_key: StepFun API key
        base_url: Optional custom base URL

    Returns:
        List[str]: Processed responses, one per cluster

    """
    write_log(f"Starting StepFun API request with model: {model}")

    api_key = ensure_api_key(api_key, "StepFun")
    url = resolve_endpoint_url("stepfun", "StepFun", base_url)

    write_log(f"Using model: {model}")

    body = build_chat_completions_body(
        model=model,
        prompt=prompt,
        temperature=0.7,
        max_tokens=4096,
    )

    return call_openai_compatible_api(
        provider_name="StepFun",
        api_key=api_key,
        url=url,
        body=body,
        post_func=requests.post,
    )
