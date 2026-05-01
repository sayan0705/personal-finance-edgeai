"""Data pipeline package — ingestion, preprocessing, synthetic generation, and HF extraction."""

from .hf_datasets.pipeline import HFDatasetPipeline

__all__ = ["HFDatasetPipeline"]
