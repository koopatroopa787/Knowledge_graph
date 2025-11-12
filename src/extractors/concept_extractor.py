"""Concept extraction from documents."""

import json
import logging
import time
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from tqdm import tqdm

from ..models import BaseLLMClient

logger = logging.getLogger(__name__)


class ConceptExtractor:
    """Extract concepts and entities from text using LLM."""

    def __init__(self, llm_client: BaseLLMClient, batch_size: int = 5,
                 delay_between_batches: int = 2):
        """
        Initialize concept extractor.

        Args:
            llm_client: LLM client to use for extraction
            batch_size: Number of documents to process in each batch
            delay_between_batches: Delay in seconds between batches
        """
        self.llm_client = llm_client
        self.batch_size = batch_size
        self.delay_between_batches = delay_between_batches
        self.max_retries = 3

    def _get_system_prompt(self) -> str:
        """Get the system prompt for concept extraction."""
        return """You are an expert at extracting key concepts and entities from text.

Your task is to identify and extract the most important concepts, entities, and ideas from the given text.

Follow these guidelines:
1. Extract only significant concepts (not trivial or overly common terms)
2. Break down complex concepts into simpler, atomic concepts when appropriate
3. Include both explicit entities (people, places, organizations) and abstract concepts (ideas, events, conditions)
4. Assign each concept to one of these categories: event, concept, place, object, document, organization, condition, misc
5. Rate the importance of each concept on a scale of 1-5 (5 being most important)

Output Format:
Return a valid JSON array with this structure:
[
  {
    "entity": "concept name",
    "importance": 3,
    "category": "concept"
  }
]

IMPORTANT: Return ONLY valid JSON. Do not include any explanatory text."""

    def _parse_response(self, response: str, chunk_id: str) -> List[Dict[str, Any]]:
        """Parse LLM response and extract concepts."""
        if not response:
            return []

        try:
            # Clean up response
            response = response.strip()

            # Try to extract JSON if wrapped in other text
            if not response.startswith('['):
                start = response.find('[')
                if start != -1:
                    response = response[start:]

            if not response.endswith(']'):
                end = response.rfind(']')
                if end != -1:
                    response = response[:end + 1]

            # Parse JSON
            concepts = json.loads(response)

            # Add chunk_id to each concept
            for concept in concepts:
                concept['chunk_id'] = chunk_id
                concept['type'] = 'concept'

            return concepts

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            logger.debug(f"Response was: {response}")
            return []
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return []

    def extract_from_text(self, text: str, chunk_id: str) -> List[Dict[str, Any]]:
        """
        Extract concepts from a single text.

        Args:
            text: Text to extract concepts from
            chunk_id: Unique identifier for this text chunk

        Returns:
            List of extracted concepts with metadata
        """
        system_prompt = self._get_system_prompt()
        user_prompt = f"Extract key concepts from the following text:\n\n{text}"

        for attempt in range(self.max_retries):
            response = self.llm_client.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                json_mode=True
            )

            if response:
                concepts = self._parse_response(response, chunk_id)
                if concepts:
                    return concepts
                logger.warning(f"Empty concepts on attempt {attempt + 1}/{self.max_retries}")

            if attempt < self.max_retries - 1:
                wait_time = 2 ** attempt
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)

        logger.warning(f"Failed to extract concepts after {self.max_retries} attempts")
        return []

    def extract_from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract concepts from a dataframe of documents.

        Args:
            df: DataFrame with 'text' and 'chunk_id' columns

        Returns:
            DataFrame with extracted concepts
        """
        all_concepts = []
        total_batches = (len(df) + self.batch_size - 1) // self.batch_size

        logger.info(f"Processing {len(df)} documents in {total_batches} batches")

        for i in tqdm(range(0, len(df), self.batch_size), desc="Extracting concepts"):
            batch_df = df.iloc[i:i + self.batch_size]
            current_batch = i // self.batch_size + 1

            for _, row in batch_df.iterrows():
                concepts = self.extract_from_text(row['text'], row['chunk_id'])
                if concepts:
                    all_concepts.extend(concepts)

            # Delay between batches
            if i + self.batch_size < len(df):
                time.sleep(self.delay_between_batches)

        logger.info(f"Extracted {len(all_concepts)} total concepts")

        if not all_concepts:
            logger.warning("No concepts were extracted!")
            return pd.DataFrame()

        # Convert to DataFrame and clean
        concepts_df = pd.DataFrame(all_concepts)
        concepts_df = concepts_df.replace(" ", np.nan)
        concepts_df = concepts_df.dropna(subset=["entity"])
        concepts_df["entity"] = concepts_df["entity"].str.lower().str.strip()

        # Remove duplicates within same chunk
        concepts_df = concepts_df.drop_duplicates(subset=['entity', 'chunk_id'])

        return concepts_df
