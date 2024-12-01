# knowledge_graph/utils/prompts.py
import json
import time
from typing import List, Dict, Any
from tqdm import tqdm
from .ollama_client import generate, show_model, check_ollama

class ModelHandler:
    def __init__(self, model_name="mistral-openorca:latest"):
        self.model_name = model_name
        self.max_retries = 3
        self.retry_delay = 2  # seconds
        self.timeout = 30  # seconds
        
        if not check_ollama():
            raise RuntimeError(
                "Ollama service is not running! Please start it with 'ollama serve'"
            )
    
    def _retry_generate(self, system_prompt: str, user_prompt: str, metadata: Dict = {}) -> List[Dict[str, Any]]:
        """Helper method to retry generation with exponential backoff and timeout."""
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                
                # Add timeout to the request
                response, _ = generate(
                    model_name=self.model_name,
                    system=system_prompt,
                    prompt=user_prompt,
                    options={"timeout": self.timeout}
                )
                
                if response:
                    # Try to clean up the response if needed
                    response = response.strip()
                    if not response.startswith('['):
                        response = '[' + response
                    if not response.endswith(']'):
                        response = response + ']'
                    
                    result = json.loads(response)
                    return [dict(item, **metadata) for item in result]
                
            except json.JSONDecodeError as e:
                print(f"\nERROR ### Invalid JSON response (attempt {attempt + 1}/{self.max_retries})")
                print(f"Response: {response}")
                print(f"JSON Error: {str(e)}")
                
            except Exception as e:
                print(f"\nERROR ### Generation failed (attempt {attempt + 1}/{self.max_retries})")
                print(f"Error: {str(e)}")
            
            if attempt < self.max_retries - 1:
                wait_time = self.retry_delay * (2 ** attempt)
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
        
        print("\nERROR ### All retry attempts failed!")
        return []

    def process_batch(self, texts: List[str], extraction_type: str = "concepts", batch_size: int = 5) -> List[Dict[str, Any]]:
        """Process a batch of texts with progress bar."""
        results = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc=f"Processing {extraction_type}"):
            batch = texts[i:i+batch_size]
            for text in batch:
                if extraction_type == "concepts":
                    result = self.extract_concepts(text)
                else:
                    result = self.extract_graph_relations(text)
                if result:
                    results.extend(result)
            # Small delay between batches to avoid overwhelming the model
            time.sleep(0.5)
            
        return results

    def extract_concepts(self, prompt: str, metadata: Dict = {}) -> List[Dict[str, Any]]:
        """Extract key concepts from text using Ollama model."""
        sys_prompt = (
            "Your task is extract the key concepts (and non personal entities) mentioned in the given context. "
            "Extract only the most important and atomistic concepts, if needed break the concepts down to the simpler concepts."
            "Categorize the concepts in one of the following categories: "
            "[event, concept, place, object, document, organisation, condition, misc]\n"
            "Format your output as a list of json with the following format:\n"
            "[\n"
            "   {\n"
            '       "entity": The Concept,\n'
            '       "importance": The contextual importance of the concept on a scale of 1 to 5 (5 being the highest),\n'
            '       "category": The Type of Concept\n'
            "   }\n"
            "]\n"
        )
        
        return self._retry_generate(sys_prompt, prompt, metadata)

    def extract_graph_relations(self, input_text: str, metadata: Dict = {}) -> List[Dict[str, Any]]:
        """Generate graph relationships from text using Ollama model."""
        sys_prompt = (
            "Extract ontological relationships from the given context. Format as JSON list of relationships.\n"
            "Each relationship should have:\n"
            '- "node_1": First concept/entity\n'
            '- "node_2": Related concept/entity\n'
            '- "edge": Brief description of relationship\n'
            "Keep relationships atomic and focused.\n"
        )

        user_prompt = f"Context: ```{input_text}```"
        return self._retry_generate(sys_prompt, user_prompt, metadata)

    def get_model_info(self):
        """Get information about the current model."""
        return show_model(self.model_name)