"""
Error handling, logging, and malformed record tracking.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from ingestion.config import LOGGING_CONFIG


class DataIngestionLogger:
    """Structured logging for data ingestion."""
    
    def __init__(self, name: str = 'DataIngestion'):
        """Initialize logger."""
        self.log_dir = Path(LOGGING_CONFIG['log_dir'])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(name)
        self.logger.setLevel(LOGGING_CONFIG['log_level'])
        
        # File handler
        error_log = self.log_dir / LOGGING_CONFIG['error_log_file'].name
        fh = logging.FileHandler(error_log)
        fh.setLevel(logging.DEBUG)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(LOGGING_CONFIG['format'])
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)


class MalformedRecordHandler:
    """Track and manage malformed records."""
    
    def __init__(self, dataset_name: str):
        """Initialize handler."""
        self.dataset_name = dataset_name
        self.malformed_records = []
        self.output_file = LOGGING_CONFIG['malformed_records_file']
        self.logger = DataIngestionLogger(f'MalformedHandler-{dataset_name}')
    
    def record_malformed(
        self,
        record: Dict[str, Any],
        error_type: str,
        error_message: str,
        timestamp: Optional[str] = None
    ):
        """
        Record a malformed record.
        
        Args:
            record: The malformed record data
            error_type: Type of error (e.g., 'TYPE_ERROR', 'MISSING_VALUE')
            error_message: Description of the error
            timestamp: Timestamp (auto-generated if None)
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        malformed_entry = {
            'timestamp': timestamp,
            'dataset': self.dataset_name,
            'error_type': error_type,
            'error_message': error_message,
            'record': record,
        }
        
        self.malformed_records.append(malformed_entry)
        self.logger.warning(f'Malformed record: {error_type} - {error_message}')
    
    def save_malformed_records(self):
        """Save malformed records to JSON file."""
        if not self.malformed_records:
            self.logger.info('No malformed records to save')
            return
        
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.output_file, 'w') as f:
            json.dump(self.malformed_records, f, indent=2, default=str)
        
        self.logger.info(f'Saved {len(self.malformed_records)} malformed records')
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of malformed records."""
        error_types = {}
        for record in self.malformed_records:
            error_type = record['error_type']
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            'dataset': self.dataset_name,
            'total_malformed': len(self.malformed_records),
            'error_types': error_types,
        }


# Custom Exceptions

class DataIngestionError(Exception):
    """Base exception for data ingestion errors."""
    pass


class FileNotFoundError(DataIngestionError):
    """Raised when data file is not found."""
    pass


class ParseError(DataIngestionError):
    """Raised when parsing fails."""
    pass


class ValidationError(DataIngestionError):
    """Raised when data validation fails."""
    pass


class SchemaError(DataIngestionError):
    """Raised when schema inference fails."""
    pass
