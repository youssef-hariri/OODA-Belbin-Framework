# src/llm_client.py

import os
import openai
from typing import Optional, List, Dict

class LLMClient:
    """
    Handles interaction with the DeepSeek Coder API.
    This class is now a lightweight client for making network requests.
    """
    def __init__(self):
        """
        Initializes the client by configuring the API key and base URL.
        """
        print("--- [LLMClient] Initializing for DeepSeek API... ---")
        
        # --- IMPORTANT ---
        # The API key will be read from an environment variable for security.
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable not set. Please set it before running.")
            
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com/v1"
        )
        self.model_name = "deepseek-coder"

        print("--- [LLMClient] API client configured successfully. ---")


    def generate_response(self, user_prompt: str, persona_prompt: Optional[str] = None) -> str:
        """
        Generates a response by making an API call to the DeepSeek model.
        """
        messages: List[Dict[str, str]] = []
        if persona_prompt:
            messages.append({"role": "system", "content": persona_prompt})
        messages.append({"role": "user", "content": user_prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=4096,
                temperature=0,  # Set to 0 for deterministic output
            )
            
            response_text = response.choices[0].message.content
            return response_text

        except Exception as e:
            print(f"--- [LLMClient] Error during API call: {e} ---")
            # Return a clear error message to be logged
            return f"Error generating response from API: {e}"
