import os
import json
import logging
import requests
from typing import Optional
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiHelper:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_KEY environment variable is not set")

        self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={self.api_key}"

        try:
            with open("Data/qa_data_fewshot.json", encoding="utf-8") as f:
                self.few_shot_examples = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(".json not found")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def call_model(self, user_input: str) -> Optional[str]:
        system_prompt = self.few_shot_examples.get("system_prompt", "")
        examples = self.few_shot_examples.get("examples", [])

        # Gemini expects "parts" for each message inside "contents"
        contents = []

        if system_prompt:
            contents.append({"role": "user", "parts": [{"text": system_prompt}]})

        for example in examples:
            if example["role"] == "user":
                contents.append({"role": "user", "parts": [{"text": example["content"]}]})
            elif example["role"] == "assistant":
                contents.append({"role": "model", "parts": [{"text": example["content"]}]})

        contents.append({"role": "user", "parts": [{"text": user_input}]})

        try:
            response = requests.post(
                self.base_url,
                headers={"Content-Type": "application/json"},
                json={"contents": contents}
            )

            logger.info("Sending request to Gemini API...")
            logger.info(f"Status Code: {response.status_code}")
            response.raise_for_status()

            data = response.json()
            logger.debug(f"Gemini response: {json.dumps(data, ensure_ascii=False)[:500]}")

            usage = data.get("usage", {})
            logger.info(f"Gemini usage: {usage.get('total_tokens', 0)} tokens "
                        f"(prompt: {usage.get('prompt_tokens', 0)}, "
                        f"completion: {usage.get('completion_tokens', 0)})")

            candidates = data.get("candidates", [])
            if candidates and "content" in candidates[0] and "parts" in candidates[0]["content"]:
                return candidates[0]["content"]["parts"][0]["text"]
            else:
                logger.error(f"Unexpected response format: {data}")
                return None

        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error: {e} | Response: {response.text}")
            return None
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return None
