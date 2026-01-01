"""
Data Ingestion Package.
Provides unified interface for loading and validating data.
"""

from ingestion.data_loader import DataLoader
from ingestion.data_validator import DataValidator
from ingestion.error_handler import MalformedRecordHandler, DataIngestionLogger

__all__ = [
    'DataLoader',
    'DataValidator',
    'MalformedRecordHandler',
    'DataIngestionLogger',
]
