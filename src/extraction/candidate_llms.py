def ask_gemini_model(query: str, model: str = "gemini-2.5-flash") -> str:
    from google import genai
    
    client = genai.Client(api_key="AIzaSyB2ZiVqtv45yjudCnB7aJ46t7r8L2MFK9s")#os.getenv("GEMINI_API_KEY"))
    
    response = client.models.generate_content(
        model=model,
        contents=query)
    return response.text

def is_model_gemini(model_name):
    return "gemini" in model_name