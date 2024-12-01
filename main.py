import os
import time
import logging
from pathlib import Path
import pandas as pd
import networkx as nx
import seaborn as sns
from pyvis.network import Network
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from utils.df_helpers import DataFrameProcessor
from utils.ollama_client import check_ollama

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class KnowledgeGraphBuilder:
    def __init__(self, input_path: str, output_dir: str, model_name: str = "llama3", pages=None):
        """
        Initialize the Knowledge Graph Builder.
        
        Args:
            input_path: Path to PDF file or directory
            output_dir: Directory to save outputs
            model_name: Name of the Ollama model to use
            pages: Pre-split document pages (optional)
        """
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.df_processor = DataFrameProcessor(model_name)
        self.pages = pages
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"Initialized builder with model: {model_name}")
        if self.pages:
            logger.info(f"Using {len(self.pages)} pre-split pages")
    
    def process_documents(self):
        """Process documents and extract concepts."""
        try:
            # Convert documents to dataframe
            self.df = self.df_processor.documents_to_dataframe(self.pages)
            logger.info(f"Created document dataframe with shape {self.df.shape}")
            
            # Process documents in smaller batches
            batch_size = 5
            all_concepts = []
            total_batches = (len(self.df) + batch_size - 1) // batch_size
            
            for i in range(0, len(self.df), batch_size):
                batch_df = self.df.iloc[i:i+batch_size]
                current_batch = i//batch_size + 1
                logger.info(f"Processing batch {current_batch} of {total_batches}")
                
                try:
                    concepts = self.df_processor.extract_concepts(batch_df)
                    if concepts:
                        all_concepts.extend(concepts)
                        logger.info(f"Extracted {len(concepts)} concepts from batch {current_batch}")
                except Exception as batch_error:
                    logger.error(f"Error processing batch {current_batch}: {str(batch_error)}")
                    continue  # Continue with next batch even if this one fails
                
                # Small delay between batches
                time.sleep(2)
            
            if not all_concepts:
                logger.error("No concepts were extracted from any batch")
                return False
            
            self.df_concepts = self.df_processor.concepts_to_dataframe(all_concepts)
            logger.info(f"Total concepts extracted: {len(all_concepts)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            raise
    
    def create_graph(self):
        """Create graph from concepts."""
        try:
            if not hasattr(self, 'df_concepts') or self.df_concepts.empty:
                logger.error("No concepts available to create graph")
                return False
                
            # Join concepts within same chunks
            dfne_join = pd.merge(
                self.df_concepts, self.df_concepts,
                how="inner",
                on="chunk_id",
                suffixes=("_L", "_R")
            )
            
            # Remove self loops
            self_loops_drop = dfne_join[dfne_join["entity_L"] == dfne_join["entity_R"]].index
            dfg = dfne_join.drop(index=self_loops_drop).reset_index()
            
            # Clean graph - remove less important nodes and edges
            less_important_nodes = dfg[(dfg["importance_L"] < 2)].index
            less_important_edges = dfg[
                (dfg["importance_L"] < 2) & 
                (dfg["importance_R"] < 2)
            ].index
            drops = less_important_nodes.union(less_important_edges)
            
            self.dfg_vis = dfg.drop(index=drops).reset_index()
            
            # Combine similar edges
            self.dfg_vis = (
                self.dfg_vis.groupby(["entity_L", "entity_R"])
                .agg({
                    "importance_L": "mean",
                    "importance_R": "mean",
                    "chunk_id": [",".join, "count"],
                })
                .reset_index()
            )
            
            self.dfg_vis.columns = [
                "entity_L",
                "entity_R",
                "importance_L",
                "importance_R",
                "chunks",
                "count",
            ]
            
            logger.info(f"Created graph with {len(self.dfg_vis)} edges")
            return True
            
        except Exception as e:
            logger.error(f"Error creating graph: {str(e)}")
            raise
    
    def visualize_graph(self):
        """Create and save interactive graph visualization."""
        try:
            if not hasattr(self, 'dfg_vis') or self.dfg_vis.empty:
                logger.error("No graph data available to visualize")
                return False
                
            # Create NetworkX graph
            G = nx.Graph()
            node_size_multiple = 6
            
            # Calculate nodes
            nodes = self.dfg_vis.groupby(["entity_L"]).agg({
                "importance_L": "mean"
            }).reset_index()
            
            # Try to detect communities
            for _, row in nodes.iterrows():
                G.add_node(row["entity_L"])
                
            for _, row in self.dfg_vis.iterrows():
                G.add_edge(str(row["entity_L"]), str(row["entity_R"]))
            
            try:
                communities_generator = nx.community.girvan_newman(G)
                top_level_communities = next(communities_generator)
                next_level_communities = next(communities_generator)
                communities = sorted(map(sorted, next_level_communities))
                
                # Generate colors for communities
                palette = "hls"
                colors = sns.color_palette(palette, len(communities)).as_hex()
                
                color_map = {}
                for idx, community in enumerate(communities):
                    for node in community:
                        color_map[node] = colors[idx]
                        
                logger.info(f"Detected {len(communities)} communities")
                
            except StopIteration:
                logger.warning("Could not detect communities, using default colors")
                color_map = {}
            
            # Create visualization network
            net = Network(
                bgcolor="#1a1a1a",
                height="900px",
                width="100%",
                select_menu=True,
                font_color="#cccccc",
            )
            
            # Reset graph for visualization
            G = nx.Graph()
            
            # Add nodes and edges to network
            for _, row in nodes.iterrows():
                G.add_node(
                    row["entity_L"],
                    size=row["importance_L"] * node_size_multiple,
                    title=row["entity_L"],
                    color=color_map.get(row["entity_L"], "#ffffff")
                )
            
            for _, row in self.dfg_vis.iterrows():
                G.add_edge(
                    str(row["entity_L"]),
                    str(row["entity_R"]),
                    weight=row["count"],
                    title=row["chunks"]
                )
            
            # Convert to Pyvis and set options
            net.from_nx(G)
            net.repulsion(node_distance=150, spring_length=400)
            net.show_buttons(filter_=["physics"])
            
            # Save visualization and data
            graph_file = self.output_dir / "knowledge_graph.html"
            net.save_graph(str(graph_file))
            
            logger.info(f"Saved graph visualization to {graph_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error visualizing graph: {str(e)}")
            raise
    
    def save_data(self):
        """Save intermediate dataframes."""
        try:
            if hasattr(self, 'df_concepts') and not self.df_concepts.empty:
                self.df_concepts.to_csv(
                    self.output_dir / "concepts.csv", 
                    sep="|", 
                    index=False
                )
            
            if hasattr(self, 'df') and not self.df.empty:
                self.df.to_csv(
                    self.output_dir / "chunks.csv", 
                    sep="|", 
                    index=False
                )
            
            if hasattr(self, 'dfg_vis') and not self.dfg_vis.empty:
                self.dfg_vis.to_csv(
                    self.output_dir / "graph.csv",
                    sep="|",
                    index=False
                )
                
            logger.info("Saved all data files")
            return True
            
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            raise

def main():
    # Get the current project directory
    current_dir = Path.cwd().parent
    
    # Configure paths relative to project root
    input_path = current_dir / "data" / "input" / "cureus-0015-00000040274.pdf"
    output_dir = current_dir / "data" / "output"
    
    # Show configured paths
    logger.info(f"Input path: {input_path}")
    logger.info(f"Output directory: {output_dir}")
    
    # Verify if file exists
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return
    
    # Use llama3 model
    model_name = "llama3"
    
    # Check if Ollama is running
    if not check_ollama():
        logger.error("Ollama service is not running!")
        return
        
    logger.info(f"Using model: {model_name}")
    
    # Initialize builder with smaller chunk size
    try:
        # First, load the document
        loader = PyPDFLoader(str(input_path))
        documents = loader.load()
        
        # Use smaller chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Reduced from 1000
            chunk_overlap=20,  # Reduced from 50
            length_function=len,
            is_separator_regex=False,
        )
        
        pages = splitter.split_documents(documents)
        logger.info(f"Split document into {len(pages)} chunks")
        
        builder = KnowledgeGraphBuilder(
            input_path=str(input_path),
            output_dir=str(output_dir),
            model_name=model_name,
            pages=pages  # Pass pre-split pages
        )
    except Exception as e:
        logger.error(f"Failed to initialize builder: {str(e)}")
        raise
    
    # Run pipeline with better error reporting
    steps = [
        ("Processing documents", builder.process_documents),
        ("Creating graph", builder.create_graph),
        ("Visualizing graph", builder.visualize_graph),
        ("Saving data", builder.save_data)
    ]
    
    for step_name, step_func in steps:
        logger.info(f"Starting: {step_name}")
        try:
            if not step_func():
                logger.error(f"Pipeline failed at: {step_name}")
                return
            logger.info(f"Completed: {step_name}")
        except Exception as e:
            logger.error(f"Error in {step_name}: {str(e)}")
            raise
    
    logger.info("Knowledge graph generation completed successfully!")

if __name__ == "__main__":
    main()