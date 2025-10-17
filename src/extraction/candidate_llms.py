import os
from google import genai
from google.genai.errors import ClientError

def ask_gemini_model(query: str, model: str = "gemini-2.5-flash") -> str:
    """Sends a concise query to a Gemini model, handling common API errors."""
    try:
        # --- MANUAL API KEY DEFINITION ---
        # WARNING: Hardcoding the API key is a security risk. Replace 
        # "YOUR_API_KEY_HERE" with your actual Gemini API key.
        client = genai.Client(api_key="AIzaSyB2ZiVqtv45yjudCnB7aJ46t7r8L2MFK9s")
        
        # Define response instructions
        instructions = (
            "Just answer in one paragraph using around 40-50 words. "
            "Please do not include any apologies or anything extra."
        )
        prompt = f"{query} {instructions}"

        response = client.models.generate_content(
            model=model,
            contents=prompt
        )
        
        return response.text
        
    except ClientError as e:
        # Catch API-specific errors (e.g., 404 NOT_FOUND, rate limits)
        print(f"API Error: Failed to generate content for model {model}. Details: {e.status_code}")
    except Exception as e:
        # Catch configuration errors (e.g., missing API key, network issues)
        print(f"Configuration Error: Could not initialize or connect. Details: {e}")
        
def is_model_gemini(model_name):
    return "gemini" in model_name