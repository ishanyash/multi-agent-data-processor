#!/usr/bin/env python3
"""
Enhanced File Handler for Multi-Agent Data Processor
Supports CSV, JSON, XLSX, and other formats
"""
import pandas as pd
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class EnhancedFileHandler:
    """Handle multiple file formats and convert to standardized DataFrame"""
    
    SUPPORTED_FORMATS = {
        '.csv': 'CSV (Comma Separated Values)',
        '.json': 'JSON (JavaScript Object Notation)',
        '.xlsx': 'Excel Spreadsheet',
        '.xls': 'Excel Spreadsheet (Legacy)',
        '.tsv': 'TSV (Tab Separated Values)',
        '.parquet': 'Parquet File',
        '.pkl': 'Pickle File',
        '.feather': 'Feather File'
    }
    
    def __init__(self):
        self.original_format = None
        self.original_filename = None
        self.conversion_log = []
    
    def detect_file_format(self, filepath: str) -> str:
        """Detect file format from extension"""
        suffix = Path(filepath).suffix.lower()
        if suffix in self.SUPPORTED_FORMATS:
            return suffix
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    
    def load_file(self, filepath: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Load file and convert to DataFrame with metadata"""
        self.original_filename = Path(filepath).name
        self.original_format = self.detect_file_format(filepath)
        
        metadata = {
            'original_format': self.original_format,
            'original_filename': self.original_filename,
            'file_size_mb': Path(filepath).stat().st_size / (1024 * 1024),
            'conversion_log': []
        }
        
        try:
            if self.original_format == '.csv':
                df = self._load_csv(filepath, metadata)
            elif self.original_format == '.json':
                df = self._load_json(filepath, metadata)
            elif self.original_format in ['.xlsx', '.xls']:
                df = self._load_excel(filepath, metadata)
            elif self.original_format == '.tsv':
                df = self._load_tsv(filepath, metadata)
            elif self.original_format == '.parquet':
                df = self._load_parquet(filepath, metadata)
            elif self.original_format == '.pkl':
                df = self._load_pickle(filepath, metadata)
            elif self.original_format == '.feather':
                df = self._load_feather(filepath, metadata)
            else:
                raise ValueError(f"Handler not implemented for {self.original_format}")
            
            # Add basic DataFrame info to metadata
            metadata.update({
                'loaded_shape': df.shape,
                'loaded_columns': df.columns.tolist(),
                'loaded_dtypes': df.dtypes.astype(str).to_dict(),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024)
            })
            
            logger.info(f"Successfully loaded {self.original_format} file: {df.shape}")
            return df, metadata
            
        except Exception as e:
            logger.error(f"Failed to load {self.original_format} file: {str(e)}")
            raise
    
    def _load_csv(self, filepath: str, metadata: Dict) -> pd.DataFrame:
        """Load CSV file with intelligent encoding detection"""
        encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                df = pd.read_csv(filepath, encoding=encoding)
                metadata['conversion_log'].append(f"CSV loaded with {encoding} encoding")
                return df
            except UnicodeDecodeError:
                continue
        
        # Last resort: try with default settings
        try:
            df = pd.read_csv(filepath)
            metadata['conversion_log'].append("CSV loaded with default encoding")
            return df
        except Exception:
            raise ValueError(f"Could not load CSV file with any encoding")
    
    def _load_json(self, filepath: str, metadata: Dict) -> pd.DataFrame:
        """Load JSON file and convert to DataFrame"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            # Array of objects
            df = pd.DataFrame(data)
            metadata['conversion_log'].append("JSON array converted to DataFrame")
        elif isinstance(data, dict):
            if 'data' in data:
                # Nested structure with 'data' key
                df = pd.DataFrame(data['data'])
                metadata['conversion_log'].append("JSON object with 'data' key converted")
            else:
                # Single object - create single row DataFrame
                df = pd.DataFrame([data])
                metadata['conversion_log'].append("Single JSON object converted to single-row DataFrame")
        else:
            raise ValueError("JSON structure not supported for DataFrame conversion")
        
        return df
    
    def _load_excel(self, filepath: str, metadata: Dict) -> pd.DataFrame:
        """Load Excel file, handling multiple sheets"""
        excel_file = pd.ExcelFile(filepath)
        
        if len(excel_file.sheet_names) == 1:
            # Single sheet
            df = pd.read_excel(filepath)
            metadata['conversion_log'].append(f"Excel file loaded (single sheet: {excel_file.sheet_names[0]})")
        else:
            # Multiple sheets - use first sheet and log others
            df = pd.read_excel(filepath, sheet_name=0)
            metadata['conversion_log'].append(
                f"Excel file with multiple sheets loaded. Used: {excel_file.sheet_names[0]}. "
                f"Available sheets: {excel_file.sheet_names}"
            )
            metadata['available_sheets'] = excel_file.sheet_names
        
        return df
    
    def _load_tsv(self, filepath: str, metadata: Dict) -> pd.DataFrame:
        """Load TSV file"""
        df = pd.read_csv(filepath, sep='\t')
        metadata['conversion_log'].append("TSV file loaded")
        return df
    
    def _load_parquet(self, filepath: str, metadata: Dict) -> pd.DataFrame:
        """Load Parquet file"""
        df = pd.read_parquet(filepath)
        metadata['conversion_log'].append("Parquet file loaded")
        return df
    
    def _load_pickle(self, filepath: str, metadata: Dict) -> pd.DataFrame:
        """Load Pickle file"""
        df = pd.read_pickle(filepath)
        metadata['conversion_log'].append("Pickle file loaded")
        return df
    
    def _load_feather(self, filepath: str, metadata: Dict) -> pd.DataFrame:
        """Load Feather file"""
        df = pd.read_feather(filepath)
        metadata['conversion_log'].append("Feather file loaded")
        return df
    
    def save_file(self, df: pd.DataFrame, output_path: str, format_type: str) -> bool:
        """Save DataFrame to specified format"""
        try:
            if format_type == 'csv':
                df.to_csv(output_path, index=False)
            elif format_type == 'json':
                df.to_json(output_path, orient='records', indent=2)
            elif format_type == 'xlsx':
                df.to_excel(output_path, index=False)
            elif format_type == 'parquet':
                df.to_parquet(output_path, index=False)
            elif format_type == 'pickle':
                df.to_pickle(output_path)
            elif format_type == 'feather':
                df.to_feather(output_path)
            else:
                raise ValueError(f"Unsupported output format: {format_type}")
            
            logger.info(f"File saved as {format_type}: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save as {format_type}: {str(e)}")
            return False
    
    @classmethod
    def get_supported_extensions(cls) -> list:
        """Get list of supported file extensions"""
        return list(cls.SUPPORTED_FORMATS.keys())
    
    @classmethod
    def get_format_description(cls, extension: str) -> str:
        """Get human-readable description of format"""
        return cls.SUPPORTED_FORMATS.get(extension, "Unknown format") 