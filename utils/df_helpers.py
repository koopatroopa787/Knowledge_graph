# knowledge_graph/utils/df_helpers.py
import uuid
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from .prompts import ModelHandler

class DataFrameProcessor:
    def __init__(self, model_name="mistral-openorca:latest"):
        self.model_handler = ModelHandler(model_name)
        
    def documents_to_dataframe(self, documents) -> pd.DataFrame:
        """Convert documents to dataframe with unique chunk IDs."""
        rows = []
        for chunk in documents:
            row = {
                "text": chunk.page_content,
                **chunk.metadata,
                "chunk_id": uuid.uuid4().hex,
            }
            rows.append(row)

        return pd.DataFrame(rows)

    def extract_concepts(self, dataframe: pd.DataFrame) -> List[Dict[str, Any]]:
        """Extract concepts from each row in dataframe."""
        results = dataframe.apply(
            lambda row: self.model_handler.extract_concepts(
                row.text, {"chunk_id": row.chunk_id, "type": "concept"}
            ),
            axis=1,
        )
        
        # Filter out None results and reset index
        results = results.dropna()
        results = results.reset_index(drop=True)

        # Flatten the list of lists to one single list of entities
        try:
            concept_list = np.concatenate(results).ravel().tolist()
            return concept_list
        except ValueError as e:
            print(f"Error flattening results: {e}")
            return []

    def concepts_to_dataframe(self, concepts_list: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert concepts list to dataframe."""
        if not concepts_list:
            return pd.DataFrame()
            
        concepts_df = pd.DataFrame(concepts_list).replace(" ", np.nan)
        concepts_df = concepts_df.dropna(subset=["entity"])
        concepts_df["entity"] = concepts_df["entity"].apply(lambda x: x.lower())
        return concepts_df

    def extract_graph_relations(self, dataframe: pd.DataFrame) -> List[Dict[str, Any]]:
        """Extract graph relationships from dataframe."""
        results = dataframe.apply(
            lambda row: self.model_handler.extract_graph_relations(
                row.text, {"chunk_id": row.chunk_id}
            ),
            axis=1
        )
        
        results = results.dropna()
        results = results.reset_index(drop=True)

        try:
            relations_list = np.concatenate(results).ravel().tolist()
            return relations_list
        except ValueError as e:
            print(f"Error flattening results: {e}")
            return []

    def relations_to_dataframe(self, relations_list: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert graph relationships to dataframe."""
        if not relations_list:
            return pd.DataFrame()
            
        graph_df = pd.DataFrame(relations_list).replace(" ", np.nan)
        graph_df = graph_df.dropna(subset=["node_1", "node_2"])
        graph_df["node_1"] = graph_df["node_1"].apply(lambda x: x.lower())
        graph_df["node_2"] = graph_df["node_2"].apply(lambda x: x.lower())
        return graph_df