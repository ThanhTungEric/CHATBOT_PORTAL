import os
import json
import logging
import requests
from typing import Optional
from googletrans import Translator
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MistralHelper:
    def __init__(self):
        self.api_key = os.getenv("MISTRAL_KEY")
        if not self.api_key:
            raise ValueError("MISTRAL_KEY environment variable is not set")

        self.base_url = "https://api.mistral.ai/v1/chat/completions"
        self.translator = Translator()

        try:
            with open("qa_data_fewshot.json", encoding="utf-8") as f:
                self.few_shot_examples = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("qa_data_fewshot.json not found")

    def detect_language(self, text: str) -> str:
        try:
            lang = self.translator.detect(text).lang
            return "vi" if lang == "vi" else "en"
        except Exception as e:
            logger.warning(f"Language detection failed, defaulting to 'vi': {e}")
            return "vi"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def call_model(self, user_input: str, lang: Optional[str] = None) -> Optional[str]:
        if not lang:
            lang = self.detect_language(user_input)

        system_prompt = self.few_shot_examples.get("system_prompt", {}).get(lang, "")
        examples = self.few_shot_examples.get("examples", {}).get(lang, [])

        messages = [{"role": "system", "content": system_prompt}] + examples
        messages.append({"role": "user", "content": user_input})

        try:
            response = requests.post(
                self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "mistral-small",  # or "mistral-medium" if you have access
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 1000,
                    "stream": False
                }
            )
            response.raise_for_status()
            data = response.json()

            usage = data.get("usage", {})
            logger.info(f"Mistral usage: {usage.get('total_tokens', 0)} tokens "
                        f"(prompt: {usage.get('prompt_tokens', 0)}, "
                        f"completion: {usage.get('completion_tokens', 0)})")

            return data["choices"][0]["message"]["content"]

        except requests.exceptions.HTTPError as e:
            self.handle_http_error(e, response)
            return None
        except Exception as e:
            logger.error(f"Mistral API error: {e}")
            return None

    def handle_http_error(self, error, response):
        try:
            status = response.status_code
            error_data = response.json()

            if status == 429:
                logger.error("Rate limit exceeded. You're hitting the Mistral free tier cap.")
            elif status == 401:
                logger.error("401 Unauthorized. Check your Mistral API key.")
            elif status == 503:
                logger.error("503 Service Unavailable. Try again later.")
            elif status == 404:
                logger.error(f"404 Not Found. Check your endpoint: {self.base_url}")
            else:
                logger.error(f"HTTP {status} Error: {error_data}")
        except Exception:
            logger.error(f"Unhandled HTTP error: {error}")
