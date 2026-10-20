"""
config.py - Secure API key loading + LLM setup.

The starter code hard-coded an Azure OpenAI key. That is replaced here with the
OpenAI API or the Google Gemini API, and the key is NEVER written in the code.
It is read from (in this order):
  1. Environment variables (GitHub Codespaces secrets, `export ...`, VS Code envFile)
  2. A local `.env` file loaded with python-dotenv (the .env file is git-ignored)
  3. Google Colab secrets (the key icon in Colab's left sidebar)

Supported variables:
  LLM_PROVIDER   = "google" or "openai"   (optional - auto-detected from the key)
  GOOGLE_API_KEY = your Gemini key         (GEMINI_API_KEY also accepted)
  OPENAI_API_KEY = your OpenAI key
  API_KEY        = generic key name used in the course "secret configuration" doc
  GEMINI_MODEL / OPENAI_MODEL = optional model overrides
"""

import os

from dotenv import load_dotenv

# Load variables from a .env file if one exists (does nothing otherwise)
load_dotenv()

# Default models (override with GEMINI_MODEL / OPENAI_MODEL if these get renamed)
DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"
DEFAULT_OPENAI_MODEL = "gpt-5-mini"


def _get_secret(name: str):
    """Look up a secret in the environment, then in Colab secrets if running in Colab."""
    value = os.getenv(name)
    if value:
        return value
    try:  # Only works inside Google Colab
        from google.colab import userdata  # type: ignore

        return userdata.get(name)
    except Exception:
        return None


def get_provider_and_key():
    """Figure out which provider to use and return (provider, api_key)."""
    provider = (os.getenv("LLM_PROVIDER") or "").strip().lower()
    google_key = _get_secret("GOOGLE_API_KEY") or _get_secret("GEMINI_API_KEY")
    openai_key = _get_secret("OPENAI_API_KEY")
    generic_key = _get_secret("API_KEY")

    if provider in ("google", "gemini"):
        return "google", google_key or generic_key
    if provider == "openai":
        return "openai", openai_key or generic_key

    # No provider set: auto-detect from whichever key exists
    if google_key:
        return "google", google_key
    if openai_key:
        return "openai", openai_key
    if generic_key:
        # OpenAI keys start with "sk-"; anything else is treated as a Google key
        return ("openai" if generic_key.startswith("sk-") else "google"), generic_key
    return None, None


def get_model():
    """Create the chat model used by the router and every agent."""
    provider, api_key = get_provider_and_key()
    if not api_key:
        raise RuntimeError(
            "Missing API key. Set GOOGLE_API_KEY (Gemini) or OPENAI_API_KEY in your "
            "environment, a .env file, Codespaces secrets or Colab secrets."
        )

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        model_name = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        model = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0)
    else:
        from langchain_openai import ChatOpenAI

        model_name = os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
        model = ChatOpenAI(model=model_name, api_key=api_key)

    # Never print the key itself - only confirm it was loaded
    print(f"API key loaded successfully. Provider: {provider} | Model: {model_name}")
    return model
