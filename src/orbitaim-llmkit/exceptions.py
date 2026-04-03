# src/my_llm_router/exceptions.py


# ─────────────────────────────────────────────────────────────
# BASE
# ─────────────────────────────────────────────────────────────

class LLMRouterError(Exception):
    """
    Base exception for all my_llm_router errors.

    Catch this to handle every error the package can raise:

        try:
            response = await router.get_insight(...)
        except LLMRouterError as e:
            print(f"Something went wrong: {e}")
    """
    pass


# ─────────────────────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────────────────────

class RouterValidationError(LLMRouterError):
    """
    Raised when the input to Router.get_insight() fails validation.

    Causes
    ------
    - Any required field is missing or empty
    - temperature is outside 0.0–1.0
    - max_tokens is <= 0 or > 128,000
    - pydantic_model is not a valid Pydantic BaseModel class
    - LLM response does not match the provided pydantic_model schema

    Example
    -------
        try:
            await router.get_insight(temperature=5.0, ...)
        except RouterValidationError as e:
            print(f"Bad input: {e}")
    """
    pass


# ─────────────────────────────────────────────────────────────
# REGISTRY
# ─────────────────────────────────────────────────────────────

class ProviderNotFoundError(LLMRouterError):
    """
    Raised when llm_name is not registered in the registry.

    Causes
    ------
    - Typo in llm_name e.g. "anthropic" vs "antropic"
    - Provider module was never imported so @register() never ran
    - Provider was unregistered (e.g. in tests)

    Example
    -------
        try:
            await router.get_insight(llm_name="unknown_llm", ...)
        except ProviderNotFoundError as e:
            print(f"No such provider: {e}")
    """
    pass


# ─────────────────────────────────────────────────────────────
# LLM CALL
# ─────────────────────────────────────────────────────────────

class LLMCallError(LLMRouterError):
    """
    Raised when the LLM provider returns an error or times out.

    Causes
    ------
    - API key invalid or expired
    - Rate limit hit
    - Network timeout
    - Provider returned an unexpected response format
    - Context window exceeded

    Example
    -------
        try:
            await router.get_insight(llm_name="openai", ...)
        except LLMCallError as e:
            print(f"LLM failed: {e}")
    """
    pass