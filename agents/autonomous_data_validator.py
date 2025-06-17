#!/usr/bin/env python3
"""
Autonomous Data Validation Agent
Specializes in detecting and fixing data quality issues automatically
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Any, Tuple
from agents.base_agent_simple import BaseAgent

class AutonomousDataValidator(BaseAgent):
    """Autonomous agent that validates and fixes data quality issues"""
    
    def __init__(self):
        super().__init__("AutonomousDataValidator", "gpt-4")
        self.validation_rules = {}
        self.fixes_applied = []
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Main processing method - autonomously validates and fixes data"""
        
        if 'dataframe' not in data:
            return {'status': 'error', 'message': 'No dataframe provided'}
        
        df = data['dataframe'].copy()
        original_shape = df.shape
        
        # Autonomous validation and fixing
        df, validation_report = self._autonomous_validation(df)
        
        return {
            'status': 'success',
            'dataframe': df,
            'original_shape': original_shape,
            'final_shape': df.shape,
            'validation_report': validation_report,
            'fixes_applied': self.fixes_applied,
            'agent': 'AutonomousDataValidator'
        }
    
    def _autonomous_validation(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Perform autonomous validation and fixing"""
        
        self.fixes_applied = []
        validation_report = {
            'issues_detected': [],
            'fixes_applied': [],
            'recommendations': []
        }
        
        # 1. Fix negative values where they shouldn't exist
        df = self._fix_negative_values(df, validation_report)
        
        # 2. Fix impossible ranges and outliers
        df = self._fix_impossible_ranges(df, validation_report)
        
        # 3. Standardize and clean text data
        df = self._standardize_text_data(df, validation_report)
        
        # 4. Fix data type inconsistencies
        df = self._fix_data_types(df, validation_report)
        
        # 5. Remove obviously invalid rows
        df = self._remove_invalid_rows(df, validation_report)
        
        return df, validation_report
    
    def _fix_negative_values(self, df: pd.DataFrame, report: Dict) -> pd.DataFrame:
        """Autonomously fix negative values where they shouldn't exist"""
        
        # Use AI to determine which columns shouldn't have negative values
        columns_analysis = self._analyze_columns_for_negatives(df)
        
        for col, should_be_positive in columns_analysis.items():
            if should_be_positive and col in df.columns:
                negative_count = (df[col] < 0).sum()
                
                if negative_count > 0:
                    # Decide fix strategy based on column type and context
                    fix_strategy = self._decide_negative_fix_strategy(df, col)
                    
                    if fix_strategy == 'absolute':
                        df[col] = df[col].abs()
                        fix_msg = f"Converted {negative_count} negative values to positive in '{col}'"
                    elif fix_strategy == 'remove_rows':
                        initial_rows = len(df)
                        df = df[df[col] >= 0]
                        rows_removed = initial_rows - len(df)
                        fix_msg = f"Removed {rows_removed} rows with negative '{col}' values"
                    elif fix_strategy == 'set_zero':
                        df.loc[df[col] < 0, col] = 0
                        fix_msg = f"Set {negative_count} negative '{col}' values to zero"
                    else:
                        fix_msg = f"Kept {negative_count} negative values in '{col}' (valid for this context)"
                    
                    self.fixes_applied.append(fix_msg)
                    report['fixes_applied'].append(fix_msg)
        
        return df
    
    def _analyze_columns_for_negatives(self, df: pd.DataFrame) -> Dict[str, bool]:
        """Use AI to determine which columns should not have negative values"""
        
        # Prepare column analysis for AI
        column_info = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                sample_values = df[col].dropna().head(10).tolist()
                negative_count = (df[col] < 0).sum()
                
                column_info[col] = {
                    'sample_values': sample_values,
                    'negative_count': int(negative_count),
                    'total_count': len(df[col].dropna()),
                    'column_name': col
                }
        
        if not column_info:
            return {}
        
        messages = [
            {"role": "system", "content": """You are a data validation expert. Analyze column names and sample values to determine which columns should logically never have negative values.

For example:
- review_count, age, price, quantity, distance should be positive
- temperature, profit_loss, coordinates can be negative
- ratings typically should be positive

Return a JSON object with column names as keys and boolean values indicating if they should be positive only."""},
            {"role": "user", "content": f"""
Analyze these columns and determine which should only have positive values:

{json.dumps(column_info, indent=2)}

Return JSON format:
{{
    "column_name": true/false,
    ...
}}

True = should be positive only, False = can be negative
"""}
        ]
        
        try:
            response = self.call_llm(messages)
            analysis = json.loads(response.choices[0].message.content)
            return analysis
        except Exception as e:
            self.logger.error(f"AI analysis failed: {str(e)}")
            # Fallback: basic heuristics
            return self._fallback_negative_analysis(df)
    
    def _fallback_negative_analysis(self, df: pd.DataFrame) -> Dict[str, bool]:
        """Fallback heuristic analysis for negative values"""
        
        positive_keywords = ['count', 'quantity', 'amount', 'price', 'cost', 'age', 'rating', 'score', 'distance', 'size', 'length', 'width', 'height', 'weight']
        negative_ok_keywords = ['temperature', 'profit', 'loss', 'change', 'difference', 'coordinate', 'latitude', 'longitude', 'balance']
        
        analysis = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                col_lower = col.lower()
                
                should_be_positive = any(keyword in col_lower for keyword in positive_keywords)
                can_be_negative = any(keyword in col_lower for keyword in negative_ok_keywords)
                
                analysis[col] = should_be_positive and not can_be_negative
        
        return analysis
    
    def _decide_negative_fix_strategy(self, df: pd.DataFrame, col: str) -> str:
        """Decide how to fix negative values in a column"""
        
        negative_count = (df[col] < 0).sum()
        total_count = len(df[col].dropna())
        negative_percentage = (negative_count / total_count) * 100 if total_count > 0 else 0
        
        # Use AI to decide strategy
        messages = [
            {"role": "system", "content": """You are a data cleaning expert. Given a column with negative values that should be positive, decide the best fix strategy.

Options:
- "absolute": Convert negative to positive (|-5| = 5)
- "remove_rows": Remove rows with negative values
- "set_zero": Set negative values to zero
- "keep": Keep as is (if there's a valid reason)

Consider the percentage of negative values and column context."""},
            {"role": "user", "content": f"""
Column: {col}
Negative values: {negative_count} out of {total_count} ({negative_percentage:.1f}%)
Sample negative values: {df[df[col] < 0][col].head(5).tolist()}

What's the best fix strategy? Return just the strategy name.
"""}
        ]
        
        try:
            response = self.call_llm(messages)
            strategy = response.choices[0].message.content.strip().lower()
            
            if strategy in ['absolute', 'remove_rows', 'set_zero', 'keep']:
                return strategy
            else:
                return 'absolute'  # Default
        except:
            # Fallback logic
            if negative_percentage > 20:
                return 'keep'  # Too many negatives, might be valid
            elif negative_percentage > 5:
                return 'set_zero'
            else:
                return 'absolute'
    
    def _fix_impossible_ranges(self, df: pd.DataFrame, report: Dict) -> pd.DataFrame:
        """Fix values that are outside possible ranges"""
        
        # Analyze each numeric column for impossible ranges
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                # Use AI to determine valid ranges
                valid_range = self._determine_valid_range(df, col)
                
                if valid_range:
                    min_val, max_val = valid_range
                    
                    # Fix values outside range
                    outliers = df[(df[col] < min_val) | (df[col] > max_val)]
                    
                    if len(outliers) > 0:
                        # Cap values to valid range
                        df.loc[df[col] < min_val, col] = min_val
                        df.loc[df[col] > max_val, col] = max_val
                        
                        fix_msg = f"Capped {len(outliers)} values in '{col}' to valid range [{min_val}, {max_val}]"
                        self.fixes_applied.append(fix_msg)
                        report['fixes_applied'].append(fix_msg)
        
        return df
    
    def _determine_valid_range(self, df: pd.DataFrame, col: str) -> Tuple[float, float] or None:
        """Use AI to determine valid range for a column"""
        
        sample_values = df[col].dropna().describe()
        
        messages = [
            {"role": "system", "content": """You are a data validation expert. Given column statistics, determine if there's a logical valid range for this type of data.

For example:
- Ratings: typically 0-5 or 1-10
- Percentages: 0-100
- Ages: 0-150
- Prices: > 0, no upper limit usually

Return JSON with min and max values, or null if no logical range exists."""},
            {"role": "user", "content": f"""
Column: {col}
Statistics: {sample_values.to_dict()}

Determine valid range in JSON format:
{{"min": value, "max": value}} or null if no logical range
"""}
        ]
        
        try:
            response = self.call_llm(messages)
            range_data = json.loads(response.choices[0].message.content)
            
            if range_data and 'min' in range_data and 'max' in range_data:
                return (range_data['min'], range_data['max'])
        except:
            pass
        
        return None
    
    def _standardize_text_data(self, df: pd.DataFrame, report: Dict) -> pd.DataFrame:
        """Standardize text data automatically"""
        
        for col in df.columns:
            if df[col].dtype == 'object':
                # Remove extra whitespace
                df[col] = df[col].astype(str).str.strip()
                
                # Standardize case for categorical-looking data
                unique_count = df[col].nunique()
                total_count = len(df[col].dropna())
                
                if unique_count < total_count * 0.1:  # Likely categorical
                    # Use AI to determine best case standardization
                    case_strategy = self._determine_case_strategy(df, col)
                    
                    if case_strategy == 'title':
                        df[col] = df[col].str.title()
                    elif case_strategy == 'upper':
                        df[col] = df[col].str.upper()
                    elif case_strategy == 'lower':
                        df[col] = df[col].str.lower()
                    
                    if case_strategy != 'keep':
                        fix_msg = f"Standardized case in '{col}' to {case_strategy}"
                        self.fixes_applied.append(fix_msg)
                        report['fixes_applied'].append(fix_msg)
        
        return df
    
    def _determine_case_strategy(self, df: pd.DataFrame, col: str) -> str:
        """Determine best case standardization strategy"""
        
        sample_values = df[col].dropna().head(20).tolist()
        
        messages = [
            {"role": "system", "content": """Analyze text values and determine the best case standardization:
- "title": Title Case (proper names, categories)
- "upper": UPPER CASE (codes, abbreviations)
- "lower": lower case (tags, simple categories)
- "keep": Keep as is (mixed case has meaning)

Return just the strategy name."""},
            {"role": "user", "content": f"""
Column: {col}
Sample values: {sample_values}

Best case strategy?
"""}
        ]
        
        try:
            response = self.call_llm(messages)
            strategy = response.choices[0].message.content.strip().lower()
            return strategy if strategy in ['title', 'upper', 'lower', 'keep'] else 'keep'
        except:
            return 'keep'
    
    def _fix_data_types(self, df: pd.DataFrame, report: Dict) -> pd.DataFrame:
        """Automatically fix obvious data type issues"""
        
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try to convert to numeric if it looks numeric
                try:
                    # Check if values look numeric
                    non_null = df[col].dropna()
                    if len(non_null) > 0:
                        # Try converting a sample
                        sample = non_null.head(10)
                        numeric_count = 0
                        
                        for val in sample:
                            try:
                                float(str(val).replace(',', '').replace('$', '').replace('%', ''))
                                numeric_count += 1
                            except:
                                pass
                        
                        if numeric_count / len(sample) > 0.8:  # 80% look numeric
                            # Convert to numeric
                            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '').str.replace('$', '').str.replace('%', ''), errors='coerce')
                            
                            fix_msg = f"Converted '{col}' from text to numeric"
                            self.fixes_applied.append(fix_msg)
                            report['fixes_applied'].append(fix_msg)
                
                except Exception as e:
                    self.logger.debug(f"Could not convert {col} to numeric: {str(e)}")
        
        return df
    
    def _remove_invalid_rows(self, df: pd.DataFrame, report: Dict) -> pd.DataFrame:
        """Remove obviously invalid rows"""
        
        initial_rows = len(df)
        
        # Remove rows that are completely empty or have only null values
        df = df.dropna(how='all')
        
        # Use AI to identify other invalid row patterns
        invalid_patterns = self._identify_invalid_patterns(df)
        
        for pattern in invalid_patterns:
            if pattern['type'] == 'duplicate_content':
                # Remove rows where all values are the same
                duplicates = df.duplicated(keep='first')
                df = df[~duplicates]
            elif pattern['type'] == 'placeholder_values':
                # Remove rows with placeholder values like 'N/A', 'NULL', etc.
                placeholder_values = ['N/A', 'NULL', 'TBD', '---', 'UNKNOWN', '', 'nan', 'none']
                mask = df.isin(placeholder_values).all(axis=1)
                df = df[~mask]
        
        rows_removed = initial_rows - len(df)
        if rows_removed > 0:
            fix_msg = f"Removed {rows_removed} invalid rows"
            self.fixes_applied.append(fix_msg)
            report['fixes_applied'].append(fix_msg)
        
        return df
    
    def _identify_invalid_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Use AI to identify invalid row patterns"""
        
        # Sample some potentially problematic rows
        sample_data = {
            'total_rows': len(df),
            'columns': df.columns.tolist(),
            'sample_rows': df.head(10).to_dict('records'),
            'duplicate_count': df.duplicated().sum()
        }
        
        messages = [
            {"role": "system", "content": """Analyze data sample and identify patterns that indicate invalid rows.

Common invalid patterns:
- All values are placeholders (N/A, NULL, etc.)
- Duplicate content
- Test data patterns
- Impossible combinations

Return JSON list of patterns found:
[{"type": "pattern_name", "description": "what makes it invalid"}]
"""},
            {"role": "user", "content": f"""
Analyze this data sample for invalid row patterns:

{json.dumps(sample_data, indent=2, default=str)}

Return JSON list of invalid patterns found.
"""}
        ]
        
        try:
            response = self.call_llm(messages)
            patterns = json.loads(response.choices[0].message.content)
            return patterns if isinstance(patterns, list) else []
        except:
            # Fallback: basic patterns
            return [
                {"type": "duplicate_content", "description": "Exact duplicate rows"},
                {"type": "placeholder_values", "description": "Rows with only placeholder values"}
            ] 