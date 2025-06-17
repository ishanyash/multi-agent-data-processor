#!/usr/bin/env python3
"""
Autonomous Text Processing Agent
Specializes in parsing, splitting, and normalizing text data
"""

import pandas as pd
import numpy as np
import json
import re
from typing import Dict, List, Any, Tuple
from agents.base_agent_simple import BaseAgent

class AutonomousTextProcessor(BaseAgent):
    """Autonomous agent that processes and normalizes text data"""
    
    def __init__(self):
        super().__init__("AutonomousTextProcessor", "gpt-4")
        self.processing_rules = {}
        self.transformations_applied = []
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Main processing method - autonomously processes text data"""
        
        if 'dataframe' not in data:
            return {'status': 'error', 'message': 'No dataframe provided'}
        
        df = data['dataframe'].copy()
        original_shape = df.shape
        
        # Autonomous text processing
        df, processing_report = self._autonomous_text_processing(df)
        
        return {
            'status': 'success',
            'dataframe': df,
            'original_shape': original_shape,
            'final_shape': df.shape,
            'processing_report': processing_report,
            'transformations_applied': self.transformations_applied,
            'agent': 'AutonomousTextProcessor'
        }
    
    def _autonomous_text_processing(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Perform autonomous text processing"""
        
        self.transformations_applied = []
        processing_report = {
            'columns_processed': [],
            'transformations': [],
            'new_columns_created': []
        }
        
        # 1. Identify and split composite text fields
        df = self._split_composite_fields(df, processing_report)
        
        # 2. Normalize and standardize text formats
        df = self._normalize_text_formats(df, processing_report)
        
        # 3. Extract structured data from text
        df = self._extract_structured_data(df, processing_report)
        
        # 4. Clean and standardize categorical text
        df = self._standardize_categorical_text(df, processing_report)
        
        return df, processing_report
    
    def _split_composite_fields(self, df: pd.DataFrame, report: Dict) -> pd.DataFrame:
        """Automatically identify and split composite text fields"""
        
        for col in df.columns:
            if df[col].dtype == 'object':
                # Analyze if column contains composite data
                composite_analysis = self._analyze_composite_field(df, col)
                
                if composite_analysis['should_split']:
                    df = self._perform_field_split(df, col, composite_analysis, report)
        
        return df
    
    def _analyze_composite_field(self, df: pd.DataFrame, col: str) -> Dict[str, Any]:
        """Use AI to analyze if a text field should be split"""
        
        # Sample values for analysis
        sample_values = df[col].dropna().head(20).tolist()
        
        # Look for common separators
        separators = ['•', '|', ',', ';', '/', '-', '::', ' - ', ' | ']
        separator_counts = {}
        
        for sep in separators:
            count = sum(1 for val in sample_values if sep in str(val))
            if count > 0:
                separator_counts[sep] = count
        
        messages = [
            {"role": "system", "content": """You are a text processing expert. Analyze text values to determine if they contain composite data that should be split into separate fields.

Look for patterns like:
- "• Turkish • Bristol" (cuisine and location)
- "Category1 | Category2" (multiple categories)
- "Name - Description" (name and description)
- "Type: Value" (key-value pairs)

Return JSON with analysis:
{
    "should_split": true/false,
    "separator": "best separator to use",
    "split_type": "categories/location/key_value/custom",
    "suggested_columns": ["col1", "col2", ...],
    "reasoning": "why it should/shouldn't be split"
}"""},
            {"role": "user", "content": f"""
Analyze this column for composite data:

Column name: {col}
Sample values: {sample_values}
Separator frequency: {separator_counts}

Should this be split into multiple columns?
"""}
        ]
        
        try:
            response = self.call_llm(messages)
            analysis = json.loads(response.choices[0].message.content)
            return analysis
        except Exception as e:
            self.logger.error(f"AI analysis failed: {str(e)}")
            # Fallback analysis
            return self._fallback_composite_analysis(sample_values, separator_counts)
    
    def _fallback_composite_analysis(self, sample_values: List, separator_counts: Dict) -> Dict[str, Any]:
        """Fallback analysis for composite fields"""
        
        if not separator_counts:
            return {"should_split": False, "reasoning": "No common separators found"}
        
        # Find most common separator
        best_separator = max(separator_counts, key=separator_counts.get)
        frequency = separator_counts[best_separator] / len(sample_values)
        
        if frequency > 0.5:  # More than 50% of values have this separator
            return {
                "should_split": True,
                "separator": best_separator,
                "split_type": "categories",
                "suggested_columns": ["part1", "part2"],
                "reasoning": f"Separator '{best_separator}' found in {frequency:.1%} of values"
            }
        
        return {"should_split": False, "reasoning": "No consistent separator pattern"}
    
    def _perform_field_split(self, df: pd.DataFrame, col: str, analysis: Dict, report: Dict) -> pd.DataFrame:
        """Perform the actual field splitting"""
        
        separator = analysis['separator']
        split_type = analysis.get('split_type', 'categories')
        
        # Split the column
        split_data = df[col].astype(str).str.split(separator, expand=True)
        
        # Clean and name the new columns
        if split_type == 'categories':
            # For categories like "• Turkish • Bristol"
            new_columns = []
            for i in range(split_data.shape[1]):
                if split_data[i].notna().any():
                    col_name = f"{col}_category_{i+1}"
                    df[col_name] = split_data[i].str.strip().str.replace('•', '').str.strip()
                    new_columns.append(col_name)
        
        elif split_type == 'location':
            # For location data
            if split_data.shape[1] >= 2:
                df[f"{col}_primary"] = split_data[0].str.strip()
                df[f"{col}_location"] = split_data[1].str.strip()
                new_columns = [f"{col}_primary", f"{col}_location"]
        
        else:
            # Generic split
            new_columns = []
            for i in range(min(split_data.shape[1], 5)):  # Limit to 5 parts
                if split_data[i].notna().any():
                    col_name = f"{col}_part_{i+1}"
                    df[col_name] = split_data[i].str.strip()
                    new_columns.append(col_name)
        
        # Log the transformation
        transform_msg = f"Split '{col}' into {len(new_columns)} columns using separator '{separator}'"
        self.transformations_applied.append(transform_msg)
        report['transformations'].append(transform_msg)
        report['new_columns_created'].extend(new_columns)
        report['columns_processed'].append(col)
        
        return df
    
    def _normalize_text_formats(self, df: pd.DataFrame, report: Dict) -> pd.DataFrame:
        """Normalize text formats automatically"""
        
        for col in df.columns:
            if df[col].dtype == 'object':
                # Get format normalization strategy
                normalization_strategy = self._get_normalization_strategy(df, col)
                
                if normalization_strategy['apply']:
                    df = self._apply_normalization(df, col, normalization_strategy, report)
        
        return df
    
    def _get_normalization_strategy(self, df: pd.DataFrame, col: str) -> Dict[str, Any]:
        """Determine text normalization strategy using AI"""
        
        sample_values = df[col].dropna().head(15).tolist()
        
        # Basic pattern analysis
        patterns = {
            'has_extra_spaces': any('  ' in str(val) for val in sample_values),
            'mixed_case': len(set(str(val).isupper() for val in sample_values if str(val))) > 1,
            'has_special_chars': any(re.search(r'[^\w\s\-\.]', str(val)) for val in sample_values),
            'has_numbers': any(re.search(r'\d', str(val)) for val in sample_values)
        }
        
        messages = [
            {"role": "system", "content": """You are a text normalization expert. Analyze text values and determine the best normalization strategy.

Consider:
- Remove extra whitespace
- Standardize case (title, upper, lower, or keep mixed)
- Clean special characters
- Standardize formats

Return JSON:
{
    "apply": true/false,
    "remove_extra_spaces": true/false,
    "case_strategy": "title/upper/lower/keep",
    "clean_special_chars": true/false,
    "custom_rules": ["rule1", "rule2"],
    "reasoning": "explanation"
}"""},
            {"role": "user", "content": f"""
Column: {col}
Sample values: {sample_values}
Patterns detected: {patterns}

What normalization should be applied?
"""}
        ]
        
        try:
            response = self.call_llm(messages)
            strategy = json.loads(response.choices[0].message.content)
            return strategy
        except:
            # Fallback strategy
            return {
                "apply": patterns['has_extra_spaces'] or patterns['mixed_case'],
                "remove_extra_spaces": patterns['has_extra_spaces'],
                "case_strategy": "title" if patterns['mixed_case'] else "keep",
                "clean_special_chars": False,
                "reasoning": "Basic cleanup needed"
            }
    
    def _apply_normalization(self, df: pd.DataFrame, col: str, strategy: Dict, report: Dict) -> pd.DataFrame:
        """Apply normalization strategy to column"""
        
        transformations = []
        
        # Remove extra spaces
        if strategy.get('remove_extra_spaces'):
            df[col] = df[col].astype(str).str.replace(r'\s+', ' ', regex=True).str.strip()
            transformations.append("removed extra spaces")
        
        # Apply case strategy
        case_strategy = strategy.get('case_strategy', 'keep')
        if case_strategy == 'title':
            df[col] = df[col].str.title()
            transformations.append("applied title case")
        elif case_strategy == 'upper':
            df[col] = df[col].str.upper()
            transformations.append("applied upper case")
        elif case_strategy == 'lower':
            df[col] = df[col].str.lower()
            transformations.append("applied lower case")
        
        # Clean special characters if needed
        if strategy.get('clean_special_chars'):
            df[col] = df[col].str.replace(r'[^\w\s\-\.]', '', regex=True)
            transformations.append("cleaned special characters")
        
        if transformations:
            transform_msg = f"Normalized '{col}': {', '.join(transformations)}"
            self.transformations_applied.append(transform_msg)
            report['transformations'].append(transform_msg)
            report['columns_processed'].append(col)
        
        return df
    
    def _extract_structured_data(self, df: pd.DataFrame, report: Dict) -> pd.DataFrame:
        """Extract structured data from text fields"""
        
        for col in df.columns:
            if df[col].dtype == 'object':
                # Look for extractable patterns
                extraction_opportunities = self._identify_extraction_opportunities(df, col)
                
                for opportunity in extraction_opportunities:
                    df = self._perform_extraction(df, col, opportunity, report)
        
        return df
    
    def _identify_extraction_opportunities(self, df: pd.DataFrame, col: str) -> List[Dict[str, Any]]:
        """Identify data that can be extracted from text"""
        
        sample_values = df[col].dropna().head(10).tolist()
        
        opportunities = []
        
        # Check for common patterns
        patterns = {
            'price': r'\$[\d,]+(?:\.\d{2})?',
            'percentage': r'\d+%',
            'phone': r'\(\d{3}\)\s*\d{3}-\d{4}|\d{3}-\d{3}-\d{4}',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'date': r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
            'rating': r'\d+(?:\.\d+)?/\d+|\d+(?:\.\d+)?\s*(?:stars?|★)',
        }
        
        for pattern_name, pattern in patterns.items():
            matches = 0
            for val in sample_values:
                if re.search(pattern, str(val)):
                    matches += 1
            
            if matches > len(sample_values) * 0.3:  # 30% of values have this pattern
                opportunities.append({
                    'type': pattern_name,
                    'pattern': pattern,
                    'frequency': matches / len(sample_values)
                })
        
        return opportunities
    
    def _perform_extraction(self, df: pd.DataFrame, col: str, opportunity: Dict, report: Dict) -> pd.DataFrame:
        """Extract structured data based on opportunity"""
        
        pattern_type = opportunity['type']
        pattern = opportunity['pattern']
        
        # Extract the data
        extracted_col = f"{col}_{pattern_type}_extracted"
        df[extracted_col] = df[col].astype(str).str.extract(f'({pattern})', expand=False)
        
        # Clean the extracted data
        if pattern_type == 'price':
            df[extracted_col] = df[extracted_col].str.replace('$', '').str.replace(',', '')
            df[extracted_col] = pd.to_numeric(df[extracted_col], errors='coerce')
        elif pattern_type == 'percentage':
            df[extracted_col] = df[extracted_col].str.replace('%', '')
            df[extracted_col] = pd.to_numeric(df[extracted_col], errors='coerce')
        
        # Log the extraction
        transform_msg = f"Extracted {pattern_type} data from '{col}' into '{extracted_col}'"
        self.transformations_applied.append(transform_msg)
        report['transformations'].append(transform_msg)
        report['new_columns_created'].append(extracted_col)
        
        return df
    
    def _standardize_categorical_text(self, df: pd.DataFrame, report: Dict) -> pd.DataFrame:
        """Standardize categorical text data"""
        
        for col in df.columns:
            if df[col].dtype == 'object':
                unique_count = df[col].nunique()
                total_count = len(df[col].dropna())
                
                # If it looks categorical (low unique count relative to total)
                if unique_count < total_count * 0.1 and unique_count > 1:
                    standardization = self._get_categorical_standardization(df, col)
                    
                    if standardization['apply']:
                        df = self._apply_categorical_standardization(df, col, standardization, report)
        
        return df
    
    def _get_categorical_standardization(self, df: pd.DataFrame, col: str) -> Dict[str, Any]:
        """Determine categorical standardization strategy"""
        
        value_counts = df[col].value_counts().head(20)
        
        messages = [
            {"role": "system", "content": """Analyze categorical text values and determine standardization needs.

Look for:
- Similar values with different formatting ("New York", "new york", "NEW YORK")
- Abbreviations vs full forms ("CA" vs "California")
- Typos or variations ("Restaurant" vs "Resturant")

Return JSON:
{
    "apply": true/false,
    "standardizations": [
        {"from": "old_value", "to": "new_value", "reason": "explanation"}
    ],
    "reasoning": "overall explanation"
}"""},
            {"role": "user", "content": f"""
Column: {col}
Value counts: {value_counts.to_dict()}

What standardizations should be applied?
"""}
        ]
        
        try:
            response = self.call_llm(messages)
            standardization = json.loads(response.choices[0].message.content)
            return standardization
        except:
            return {"apply": False, "reasoning": "No standardization needed"}
    
    def _apply_categorical_standardization(self, df: pd.DataFrame, col: str, standardization: Dict, report: Dict) -> pd.DataFrame:
        """Apply categorical standardization"""
        
        changes_made = []
        
        for std in standardization.get('standardizations', []):
            from_val = std['from']
            to_val = std['to']
            
            # Apply the standardization
            mask = df[col] == from_val
            if mask.any():
                df.loc[mask, col] = to_val
                changes_made.append(f"'{from_val}' → '{to_val}'")
        
        if changes_made:
            transform_msg = f"Standardized categories in '{col}': {', '.join(changes_made[:3])}"
            if len(changes_made) > 3:
                transform_msg += f" and {len(changes_made)-3} more"
            
            self.transformations_applied.append(transform_msg)
            report['transformations'].append(transform_msg)
            report['columns_processed'].append(col)
        
        return df 