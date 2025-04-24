import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the model")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples to evaluate")
    parser.add_argument(
        "--entry_point",
        type=str,
        default="all",
        choices=["flat-embeddings", "flat-rag", "hierarchical-embeddings", "hierarchical-rag", "all"],
        help="Entry point for the API",
    )
    parser.add_argument("--experiment_name", type=str, default="graph-rag-evaluation", help="Experiment name")
    return parser.parse_args()
