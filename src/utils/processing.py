import re
import emoji

def clean_text(text):
    """
    Normalizes text by:
    1. Replacing mentions with @USER
    2. Replacing URLs with HTTPURL
    3. Demojizing (converting emojis to text)
    """
    if not isinstance(text, str):
        return ""
    
    # Replace URL
    text = re.sub(r'http\S+', 'HTTPURL', text)
    text = re.sub(r'www\S+', 'HTTPURL', text)
    
    # Replace User mentions
    text = re.sub(r'@\w+', '@USER', text)
    
    # Demojize
    text = emoji.demojize(text)
    
    # Standardize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
