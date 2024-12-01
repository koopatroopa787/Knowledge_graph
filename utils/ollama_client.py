# knowledge_graph/utils/ollama_client.py
import os
import json
import requests

BASE_URL = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')

def generate(model_name, prompt, system=None, template=None, context=None, options=None, callback=None):
    """Generate a response using Ollama model."""
    try:
        url = f"{BASE_URL}/api/generate"
        payload = {
            "model": model_name, 
            "prompt": prompt, 
            "system": system, 
            "template": template, 
            "context": context, 
            "options": options
        }
        
        payload = {k: v for k, v in payload.items() if v is not None}
        
        with requests.post(url, json=payload, stream=True) as response:
            response.raise_for_status()
            final_context = None
            full_response = ""

            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    if callback:
                        callback(chunk)
                    else:
                        if not chunk.get("done"):
                            response_piece = chunk.get("response", "")
                            full_response += response_piece
                    
                    if chunk.get("done"):
                        final_context = chunk.get("context")
            
            return full_response, final_context
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None, None

def list_models():
    """List available models."""
    try:
        response = requests.get(f"{BASE_URL}/api/tags")
        response.raise_for_status()
        data = response.json()
        return data.get('models', [])
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None

def show_model(model_name):
    """Show info about a model."""
    try:
        url = f"{BASE_URL}/api/show"
        payload = {"name": model_name}
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None

def check_ollama():
    """Check if Ollama is running."""
    try:
        url = f"{BASE_URL}/"
        response = requests.head(url)
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException:
        return False