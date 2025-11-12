#!/usr/bin/env python3
"""
Knowledge Graph Generator - Main CLI Application

Generate knowledge graphs from documents using various LLM providers.
"""

import sys
import logging
import argparse
from pathlib import Path

from src.config import Config
from src.models import LLMFactory
from src.utils import DocumentProcessor
from src.extractors import ConceptExtractor, GraphBuilder


def setup_logging(config: Config):
    """Setup logging configuration."""
    log_level = getattr(logging, config.get('logging.level', 'INFO'))
    log_format = config.get('logging.format', '%(asctime)s - %(levelname)s - %(message)s')
    log_file = config.get('logging.file')

    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers
    )

    return logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate knowledge graphs from documents using LLMs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate knowledge graph from a PDF using default config
  python knowledge_graph.py document.pdf

  # Use a specific LLM provider
  python knowledge_graph.py document.pdf --provider openai --model gpt-4

  # Specify output directory
  python knowledge_graph.py document.pdf --output ./my_output

  # Use custom configuration file
  python knowledge_graph.py document.pdf --config my_config.yaml

For more information, see README.md
        """
    )

    parser.add_argument(
        'input_file',
        type=str,
        help='Path to input document (PDF or TXT)'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output directory (default: from config.yaml)'
    )

    parser.add_argument(
        '-c', '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )

    parser.add_argument(
        '--provider',
        type=str,
        choices=['openai', 'anthropic', 'ollama'],
        help='LLM provider (overrides config)'
    )

    parser.add_argument(
        '--model',
        type=str,
        help='Model name (overrides config)'
    )

    parser.add_argument(
        '--chunk-size',
        type=int,
        help='Document chunk size (overrides config)'
    )

    parser.add_argument(
        '--min-importance',
        type=int,
        help='Minimum importance threshold for graph (1-5, overrides config)'
    )

    parser.add_argument(
        '--no-communities',
        action='store_true',
        help='Disable community detection in visualization'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging (DEBUG level)'
    )

    return parser.parse_args()


def main():
    """Main application entry point."""
    args = parse_arguments()

    try:
        # Load configuration
        config = Config(args.config)

        # Override config with CLI arguments
        if args.verbose:
            config.config['logging']['level'] = 'DEBUG'

        # Setup logging
        logger = setup_logging(config)
        logger.info("=" * 80)
        logger.info("Knowledge Graph Generator v2.0")
        logger.info("=" * 80)

        # Validate input file
        input_path = Path(args.input_file)
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            return 1

        # Setup output directory
        output_dir = Path(args.output) if args.output else config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")

        # Override config with CLI arguments
        if args.provider:
            config.config['llm']['provider'] = args.provider
        if args.model:
            provider = config.llm_provider
            config.config['llm']['models'][provider] = args.model
        if args.chunk_size:
            config.config['document_processing']['chunk_size'] = args.chunk_size
        if args.min_importance:
            config.config['graph']['min_importance'] = args.min_importance

        # Initialize LLM client
        logger.info(f"Initializing LLM: {config.llm_provider} - {config.model_name}")
        llm_client = LLMFactory.from_config(config)

        # Check LLM availability
        logger.info("Checking LLM availability...")
        if not llm_client.check_availability():
            logger.error(
                f"LLM service not available. "
                f"Please check your {config.llm_provider} configuration."
            )
            return 1

        logger.info("LLM service is available ✓")

        # Step 1: Process document
        logger.info("\n" + "=" * 80)
        logger.info("Step 1: Processing Document")
        logger.info("=" * 80)

        doc_processor = DocumentProcessor(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )

        chunks_df = doc_processor.process_file(input_path)

        if config.save_intermediate:
            doc_processor.save_chunks(chunks_df, output_dir / "chunks.csv")

        # Step 2: Extract concepts
        logger.info("\n" + "=" * 80)
        logger.info("Step 2: Extracting Concepts")
        logger.info("=" * 80)

        concept_extractor = ConceptExtractor(
            llm_client=llm_client,
            batch_size=config.batch_size,
            delay_between_batches=config.delay_between_batches
        )

        concepts_df = concept_extractor.extract_from_dataframe(chunks_df)

        if concepts_df.empty:
            logger.error("No concepts extracted. Cannot build graph.")
            return 1

        if config.save_intermediate:
            concepts_df.to_csv(output_dir / "concepts.csv", sep="|", index=False)

        # Step 3: Build graph
        logger.info("\n" + "=" * 80)
        logger.info("Step 3: Building Knowledge Graph")
        logger.info("=" * 80)

        graph_builder = GraphBuilder(
            min_importance=config.min_importance,
            node_size_multiple=config.get('graph.node_size_multiple', 6)
        )

        graph_df = graph_builder.build_graph_dataframe(concepts_df)

        if graph_df.empty:
            logger.error("No graph edges created. Try lowering min_importance threshold.")
            return 1

        # Save graph data
        graph_builder.save_graph_data(graph_df, output_dir)

        # Step 4: Visualize graph
        logger.info("\n" + "=" * 80)
        logger.info("Step 4: Creating Visualization")
        logger.info("=" * 80)

        viz_config = config.get('graph.visualization', {})
        enable_communities = not args.no_communities and config.get('graph.enable_communities', True)
        community_levels = config.get('graph.community_levels', 2)

        success = graph_builder.visualize(
            graph_df=graph_df,
            output_path=output_dir / "knowledge_graph.html",
            enable_communities=enable_communities,
            community_levels=community_levels,
            viz_config=viz_config
        )

        if not success:
            logger.error("Failed to create visualization")
            return 1

        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("✓ Knowledge Graph Generated Successfully!")
        logger.info("=" * 80)
        logger.info(f"Input file: {input_path}")
        logger.info(f"Chunks processed: {len(chunks_df)}")
        logger.info(f"Concepts extracted: {len(concepts_df)}")
        logger.info(f"Graph edges: {len(graph_df)}")
        logger.info(f"\nOutputs saved to: {output_dir.absolute()}")
        logger.info(f"  - knowledge_graph.html (interactive visualization)")
        logger.info(f"  - graph.csv (graph data)")
        logger.info(f"  - graph.json (graph data in JSON)")
        logger.info(f"  - graph_stats.json (graph statistics)")

        if config.save_intermediate:
            logger.info(f"  - concepts.csv (extracted concepts)")
            logger.info(f"  - chunks.csv (document chunks)")

        logger.info("\n👉 Open knowledge_graph.html in your browser to explore the graph!")

        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n\nInterrupted by user", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
