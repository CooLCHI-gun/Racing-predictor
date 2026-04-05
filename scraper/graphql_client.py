"""
Shared HKJC GraphQL client helpers.

This module centralizes:
  - endpoint URL
  - browser-like headers
  - retries / timeout
  - whitelist-error handling
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

import requests

from config import config

logger = logging.getLogger(__name__)

GRAPHQL_URL = config.HKJC_GRAPHQL_URL

GRAPHQL_HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Origin": "https://bet.hkjc.com",
    "Referer": "https://bet.hkjc.com/",
    "Accept": "application/json",
}


def execute_graphql(
    query: str,
    variables: Dict[str, Any],
    operation_name: str = "racing",
    max_retries: int = 3,
    timeout: int = 15,
) -> Optional[Dict[str, Any]]:
    """
    Execute a GraphQL request against HKJC API.

    Returns:
        data dictionary on success, or None on failure/whitelist block.
    """
    payload = {
        "operationName": operation_name,
        "query": query,
        "variables": variables,
    }

    session = requests.Session()
    session.headers.update(GRAPHQL_HEADERS)

    for attempt in range(1, max_retries + 1):
        try:
            resp = session.post(GRAPHQL_URL, json=payload, timeout=timeout)
            resp.raise_for_status()
            body = resp.json()

            errors = body.get("errors") or []
            if errors:
                err_codes = [e.get("extensions", {}).get("code", "") for e in errors]
                if "WHITELIST_ERROR" in err_codes:
                    logger.info("GraphQL query blocked by HKJC whitelist")
                    return None
                logger.warning("GraphQL errors: %s", errors)
                return None

            return body.get("data")

        except Exception as exc:
            logger.warning("GraphQL attempt %s/%s failed: %s", attempt, max_retries, exc)
            if attempt < max_retries:
                time.sleep(1.5 ** attempt)

    return None
