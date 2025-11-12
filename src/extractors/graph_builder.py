"""Build and visualize knowledge graphs."""

import logging
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
import networkx as nx
import seaborn as sns
from pyvis.network import Network

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Build and visualize knowledge graphs from concepts."""

    def __init__(self, min_importance: int = 2,
                 node_size_multiple: int = 6):
        """
        Initialize graph builder.

        Args:
            min_importance: Minimum importance threshold for including concepts
            node_size_multiple: Multiplier for node sizes in visualization
        """
        self.min_importance = min_importance
        self.node_size_multiple = node_size_multiple

    def build_graph_dataframe(self, concepts_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build graph edges by joining concepts within same chunks.

        Args:
            concepts_df: DataFrame with extracted concepts

        Returns:
            DataFrame with graph edges
        """
        if concepts_df.empty:
            logger.error("Cannot build graph from empty concepts dataframe")
            return pd.DataFrame()

        logger.info(f"Building graph from {len(concepts_df)} concepts")

        # Self-join on chunk_id to create edges
        dfne_join = pd.merge(
            concepts_df, concepts_df,
            how="inner",
            on="chunk_id",
            suffixes=("_L", "_R")
        )

        logger.info(f"Created {len(dfne_join)} potential edges")

        # Remove self-loops
        self_loops = dfne_join[dfne_join["entity_L"] == dfne_join["entity_R"]].index
        dfg = dfne_join.drop(index=self_loops).reset_index(drop=True)

        logger.info(f"Removed {len(self_loops)} self-loops, {len(dfg)} edges remain")

        # Filter by importance
        important_edges = dfg[
            (dfg["importance_L"] >= self.min_importance) |
            (dfg["importance_R"] >= self.min_importance)
        ]

        logger.info(f"Kept {len(important_edges)} edges with importance >= {self.min_importance}")

        if important_edges.empty:
            logger.warning("No important edges found! Try lowering min_importance threshold")
            return pd.DataFrame()

        # Aggregate duplicate edges
        dfg_agg = (
            important_edges.groupby(["entity_L", "entity_R"])
            .agg({
                "importance_L": "mean",
                "importance_R": "mean",
                "chunk_id": [lambda x: ",".join(sorted(set(x))), "count"],
            })
            .reset_index()
        )

        dfg_agg.columns = [
            "entity_L",
            "entity_R",
            "importance_L",
            "importance_R",
            "chunks",
            "count",
        ]

        logger.info(f"Aggregated to {len(dfg_agg)} unique edges")

        return dfg_agg

    def _detect_communities(self, G: nx.Graph, num_levels: int = 2):
        """
        Detect communities in the graph.

        Args:
            G: NetworkX graph
            num_levels: Number of community levels to detect

        Returns:
            Dictionary mapping nodes to colors, or empty dict if detection fails
        """
        try:
            communities_generator = nx.community.girvan_newman(G)

            # Get the specified level of communities
            communities = None
            for i in range(num_levels):
                communities = next(communities_generator)

            communities = sorted(map(sorted, communities))

            # Generate colors
            colors = sns.color_palette("hls", len(communities)).as_hex()

            color_map = {}
            for idx, community in enumerate(communities):
                for node in community:
                    color_map[node] = colors[idx]

            logger.info(f"Detected {len(communities)} communities")
            return color_map

        except (StopIteration, nx.NetworkXError) as e:
            logger.warning(f"Could not detect communities: {e}")
            return {}

    def visualize(self, graph_df: pd.DataFrame, output_path: Path,
                 enable_communities: bool = True,
                 community_levels: int = 2,
                 viz_config: dict = None) -> bool:
        """
        Create interactive visualization of the knowledge graph.

        Args:
            graph_df: DataFrame with graph edges
            output_path: Path to save HTML visualization
            enable_communities: Whether to detect and color communities
            community_levels: Number of community detection levels
            viz_config: Visualization configuration dict

        Returns:
            True if successful, False otherwise
        """
        if graph_df.empty:
            logger.error("Cannot visualize empty graph")
            return False

        # Default visualization config
        if viz_config is None:
            viz_config = {}

        height = viz_config.get('height', '900px')
        width = viz_config.get('width', '100%')
        bgcolor = viz_config.get('background_color', '#1a1a1a')
        font_color = viz_config.get('font_color', '#cccccc')
        node_distance = viz_config.get('node_distance', 150)
        spring_length = viz_config.get('spring_length', 400)

        try:
            # Build NetworkX graph for community detection
            G = nx.Graph()

            # Add nodes with importance
            nodes = graph_df.groupby(["entity_L"]).agg({
                "importance_L": "mean"
            }).reset_index()

            for _, row in nodes.iterrows():
                G.add_node(row["entity_L"])

            # Add edges
            for _, row in graph_df.iterrows():
                G.add_edge(row["entity_L"], row["entity_R"], weight=row["count"])

            logger.info(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

            # Detect communities
            color_map = {}
            if enable_communities and G.number_of_nodes() > 2:
                color_map = self._detect_communities(G, community_levels)

            # Create Pyvis network
            net = Network(
                bgcolor=bgcolor,
                height=height,
                width=width,
                select_menu=True,
                font_color=font_color,
            )

            # Rebuild graph with visual properties
            G_vis = nx.Graph()

            # Add nodes with visual properties
            for _, row in nodes.iterrows():
                node_id = row["entity_L"]
                G_vis.add_node(
                    node_id,
                    size=row["importance_L"] * self.node_size_multiple,
                    title=f"{node_id} (importance: {row['importance_L']:.1f})",
                    color=color_map.get(node_id, "#97c2fc"),
                    label=node_id
                )

            # Add edges with properties
            for _, row in graph_df.iterrows():
                G_vis.add_edge(
                    row["entity_L"],
                    row["entity_R"],
                    weight=row["count"],
                    title=f"Co-occurs {row['count']} times\nChunks: {row['chunks'][:100]}...",
                    value=row["count"]
                )

            # Convert to Pyvis
            net.from_nx(G_vis)

            # Set physics options
            net.repulsion(
                node_distance=node_distance,
                spring_length=spring_length
            )
            net.show_buttons(filter_=["physics"])

            # Save
            output_path.parent.mkdir(parents=True, exist_ok=True)
            net.save_graph(str(output_path))

            logger.info(f"Saved interactive graph to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
            return False

    def save_graph_data(self, graph_df: pd.DataFrame, output_dir: Path):
        """
        Save graph data in multiple formats.

        Args:
            graph_df: DataFrame with graph edges
            output_dir: Directory to save files
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Save as CSV
            csv_path = output_dir / "graph.csv"
            graph_df.to_csv(csv_path, sep="|", index=False)
            logger.info(f"Saved graph CSV to {csv_path}")

            # Save as JSON
            json_path = output_dir / "graph.json"
            graph_df.to_json(json_path, orient="records", indent=2)
            logger.info(f"Saved graph JSON to {json_path}")

            # Save graph statistics
            stats = {
                "num_edges": len(graph_df),
                "num_nodes": len(set(graph_df["entity_L"]) | set(graph_df["entity_R"])),
                "avg_importance": (graph_df["importance_L"].mean() + graph_df["importance_R"].mean()) / 2,
                "total_connections": graph_df["count"].sum(),
            }

            stats_path = output_dir / "graph_stats.json"
            import json
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
            logger.info(f"Saved graph statistics to {stats_path}")

        except Exception as e:
            logger.error(f"Error saving graph data: {e}")
