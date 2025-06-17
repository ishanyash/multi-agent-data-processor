#!/usr/bin/env python3
"""
Autonomous Price Normalization Agent
Specializes in standardizing price formats and currency handling
"""

import pandas as pd
import numpy as np
import json
import re
from typing import Dict, List, Any, Tuple
from agents.base_agent_simple import BaseAgent

class AutonomousPriceNormalizer(BaseAgent):
    """Autonomous agent that normalizes price and currency data"""
    
    def __init__(self):
        super().__init__("AutonomousPriceNormalizer", "gpt-4")
        self.price_mappings = {}
        self.normalizations_applied = []
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Main processing method - autonomously normalizes price data"""
        
        if 'dataframe' not in data:
            return {'status': 'error', 'message': 'No dataframe provided'}
        
        df = data['dataframe'].copy()
        original_shape = df.shape
        
        # Autonomous price normalization
        df, normalization_report = self._autonomous_price_normalization(df)
        
        return {
            'status': 'success',
            'dataframe': df,
            'original_shape': original_shape,
            'final_shape': df.shape,
            'normalization_report': normalization_report,
            'normalizations_applied': self.normalizations_applied,
            'agent': 'AutonomousPriceNormalizer'
        }
    
    def _autonomous_price_normalization(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Perform autonomous price normalization"""
        
        self.normalizations_applied = []
        normalization_report = {
            'price_columns_found': [],
            'normalizations': [],
            'price_ranges_created': [],
            'currency_standardizations': []
        }
        
        # 1. Identify price columns
        price_columns = self._identify_price_columns(df)
        normalization_report['price_columns_found'] = price_columns
        
        # 2. Normalize price formats
        for col in price_columns:
            df = self._normalize_price_column(df, col, normalization_report)
        
        # 3. Create price range categories
        df = self._create_price_ranges(df, price_columns, normalization_report)
        
        # 4. Standardize currency representations
        df = self._standardize_currency_formats(df, normalization_report)
        
        return df, normalization_report
    
    def _identify_price_columns(self, df: pd.DataFrame) -> List[str]:
        """Identify columns that contain price data"""
        
        price_columns = []
        
        for col in df.columns:
            # Check column name for price indicators
            col_lower = col.lower()
            price_keywords = ['price', 'cost', 'amount', 'fee', 'charge', 'rate', 'fare', 'salary', 'wage', 'income']
            
            name_suggests_price = any(keyword in col_lower for keyword in price_keywords)
            
            # Check data content for price patterns
            content_suggests_price = False
            if df[col].dtype == 'object':
                sample_values = df[col].dropna().head(20).astype(str).tolist()
                
                # Look for currency symbols and price patterns
                price_patterns = [
                    r'\$+',  # Dollar signs like $$$$
                    r'\$\d+',  # $123
                    r'\d+\$',  # 123$
                    r'£\d+',   # £123
                    r'€\d+',   # €123
                    r'\d+\.\d{2}',  # 123.45
                ]
                
                price_matches = 0
                for val in sample_values:
                    if any(re.search(pattern, val) for pattern in price_patterns):
                        price_matches += 1
                
                content_suggests_price = price_matches > len(sample_values) * 0.3
            
            if name_suggests_price or content_suggests_price:
                price_columns.append(col)
        
        return price_columns
    
    def _normalize_price_column(self, df: pd.DataFrame, col: str, report: Dict) -> pd.DataFrame:
        """Normalize a specific price column"""
        
        # Analyze the price format
        price_format_analysis = self._analyze_price_format(df, col)
        
        if price_format_analysis['needs_normalization']:
            df = self._apply_price_normalization(df, col, price_format_analysis, report)
        
        return df
    
    def _analyze_price_format(self, df: pd.DataFrame, col: str) -> Dict[str, Any]:
        """Analyze the format of price data in a column"""
        
        sample_values = df[col].dropna().head(30).astype(str).tolist()
        
        # Analyze patterns
        patterns = {
            'dollar_signs': sum(1 for val in sample_values if '$' in val and val.count('$') > 1),
            'currency_symbols': sum(1 for val in sample_values if any(sym in val for sym in ['$', '£', '€', '¥'])),
            'numeric_only': sum(1 for val in sample_values if val.replace('.', '').replace(',', '').isdigit()),
            'ranges': sum(1 for val in sample_values if '-' in val or 'to' in val.lower()),
            'text_prices': sum(1 for val in sample_values if any(word in val.lower() for word in ['cheap', 'expensive', 'free', 'low', 'high', 'moderate']))
        }
        
        messages = [
            {"role": "system", "content": """You are a price data expert. Analyze price values and determine the best normalization strategy.

Common price formats:
- $$$$ (price level indicators)
- $25.99 (actual prices)
- 25-50 (price ranges)
- "Expensive", "Moderate" (text descriptions)

Return JSON:
{
    "needs_normalization": true/false,
    "current_format": "description of current format",
    "normalization_strategy": "specific strategy to apply",
    "target_format": "what it should become",
    "create_numeric": true/false,
    "create_categories": true/false,
    "reasoning": "explanation"
}"""},
            {"role": "user", "content": f"""
Column: {col}
Sample values: {sample_values}
Pattern analysis: {patterns}

What normalization strategy should be applied?
"""}
        ]
        
        try:
            response = self.call_llm(messages)
            analysis = json.loads(response.choices[0].message.content)
            return analysis
        except Exception as e:
            self.logger.error(f"AI analysis failed: {str(e)}")
            return self._fallback_price_analysis(sample_values, patterns)
    
    def _fallback_price_analysis(self, sample_values: List[str], patterns: Dict) -> Dict[str, Any]:
        """Fallback analysis for price formats"""
        
        total_samples = len(sample_values)
        
        # If mostly dollar signs (like $$$$)
        if patterns['dollar_signs'] > total_samples * 0.5:
            return {
                "needs_normalization": True,
                "current_format": "Price level indicators ($$$$)",
                "normalization_strategy": "convert_to_numeric_scale",
                "target_format": "Numeric scale (1-4)",
                "create_numeric": True,
                "create_categories": True,
                "reasoning": "Convert $ symbols to numeric price levels"
            }
        
        # If text descriptions
        elif patterns['text_prices'] > total_samples * 0.3:
            return {
                "needs_normalization": True,
                "current_format": "Text price descriptions",
                "normalization_strategy": "standardize_text_categories",
                "target_format": "Standardized categories",
                "create_numeric": False,
                "create_categories": True,
                "reasoning": "Standardize text price descriptions"
            }
        
        return {"needs_normalization": False, "reasoning": "No clear normalization needed"}
    
    def _apply_price_normalization(self, df: pd.DataFrame, col: str, analysis: Dict, report: Dict) -> pd.DataFrame:
        """Apply price normalization based on analysis"""
        
        strategy = analysis['normalization_strategy']
        
        if strategy == 'convert_to_numeric_scale':
            df = self._convert_dollar_signs_to_numeric(df, col, analysis, report)
        elif strategy == 'standardize_text_categories':
            df = self._standardize_price_text(df, col, analysis, report)
        elif strategy == 'extract_numeric_prices':
            # Extract numeric values from mixed price text
            df[f"{col}_numeric"] = df[col].astype(str).str.extract(r'(\d+(?:\.\d{2})?)', expand=False)
            df[f"{col}_numeric"] = pd.to_numeric(df[f"{col}_numeric"], errors='coerce')
            report['normalizations'].append(f"Extracted numeric values from '{col}'")
        elif strategy == 'normalize_ranges':
            # Handle price ranges like "25-50"
            df[f"{col}_min"] = df[col].astype(str).str.extract(r'(\d+)(?:-|\s*to\s*)', expand=False)
            df[f"{col}_max"] = df[col].astype(str).str.extract(r'-\s*(\d+)|to\s*(\d+)', expand=False).fillna(method='bfill', axis=1).iloc[:, 0]
            report['normalizations'].append(f"Split price ranges in '{col}'")
        
        return df
    
    def _convert_dollar_signs_to_numeric(self, df: pd.DataFrame, col: str, analysis: Dict, report: Dict) -> pd.DataFrame:
        """Convert $$$$ format to numeric scale"""
        
        # Create mapping for dollar signs
        dollar_mapping = self._create_dollar_sign_mapping(df, col)
        
        # Create new numeric column
        numeric_col = f"{col}_numeric"
        df[numeric_col] = df[col].astype(str).map(dollar_mapping)
        
        # Create categorical column
        if analysis.get('create_categories'):
            category_col = f"{col}_category"
            category_mapping = {
                1: 'Budget',
                2: 'Moderate', 
                3: 'Expensive',
                4: 'Luxury'
            }
            df[category_col] = df[numeric_col].map(category_mapping)
            
            normalization_msg = f"Converted '{col}' dollar signs to numeric scale and categories"
            report['normalizations'].append(normalization_msg)
            report['price_ranges_created'].extend([numeric_col, category_col])
        else:
            normalization_msg = f"Converted '{col}' dollar signs to numeric scale"
            report['normalizations'].append(normalization_msg)
        
        self.normalizations_applied.append(normalization_msg)
        
        return df
    
    def _create_dollar_sign_mapping(self, df: pd.DataFrame, col: str) -> Dict[str, int]:
        """Create mapping from dollar signs to numeric values"""
        
        unique_values = df[col].dropna().unique()
        
        # Use AI to create intelligent mapping
        messages = [
            {"role": "system", "content": """Create a mapping from dollar sign patterns to numeric values.

Common patterns:
- $ = 1 (Budget)
- $$ = 2 (Moderate)  
- $$$ = 3 (Expensive)
- $$$$ = 4 (Luxury)

Handle variations and edge cases appropriately.

Return JSON mapping like: {"$": 1, "$$": 2, ...}"""},
            {"role": "user", "content": f"""
Create numeric mapping for these price values:
{unique_values.tolist()}

Return JSON mapping from string to number.
"""}
        ]
        
        try:
            response = self.call_llm(messages)
            mapping = json.loads(response.choices[0].message.content)
            return mapping
        except:
            # Fallback mapping
            mapping = {}
            for val in unique_values:
                dollar_count = str(val).count('$')
                if dollar_count > 0:
                    mapping[str(val)] = min(dollar_count, 4)
            return mapping
    
    def _standardize_price_text(self, df: pd.DataFrame, col: str, analysis: Dict, report: Dict) -> pd.DataFrame:
        """Standardize text price descriptions"""
        
        # Get standardization mapping
        text_mapping = self._create_price_text_mapping(df, col)
        
        # Apply standardization
        standardized_col = f"{col}_standardized"
        df[standardized_col] = df[col].astype(str).map(text_mapping).fillna(df[col])
        
        normalization_msg = f"Standardized price text in '{col}'"
        self.normalizations_applied.append(normalization_msg)
        report['normalizations'].append(normalization_msg)
        
        return df
    
    def _create_price_text_mapping(self, df: pd.DataFrame, col: str) -> Dict[str, str]:
        """Create mapping for standardizing price text"""
        
        unique_values = df[col].dropna().unique()
        
        messages = [
            {"role": "system", "content": """Create a standardization mapping for price-related text values.

Standardize to consistent categories like:
- "Budget" / "Low" / "Cheap" → "Budget"
- "Moderate" / "Medium" / "Average" → "Moderate"  
- "Expensive" / "High" / "Premium" → "Expensive"
- "Luxury" / "Very Expensive" → "Luxury"

Return JSON mapping: {"original": "standardized", ...}"""},
            {"role": "user", "content": f"""
Standardize these price text values:
{unique_values.tolist()}

Return JSON mapping from original to standardized values.
"""}
        ]
        
        try:
            response = self.call_llm(messages)
            mapping = json.loads(response.choices[0].message.content)
            return mapping
        except:
            # Fallback mapping
            mapping = {}
            for val in unique_values:
                val_lower = str(val).lower()
                if any(word in val_lower for word in ['cheap', 'low', 'budget', 'inexpensive']):
                    mapping[str(val)] = 'Budget'
                elif any(word in val_lower for word in ['moderate', 'medium', 'average', 'mid']):
                    mapping[str(val)] = 'Moderate'
                elif any(word in val_lower for word in ['expensive', 'high', 'premium', 'costly']):
                    mapping[str(val)] = 'Expensive'
                elif any(word in val_lower for word in ['luxury', 'very expensive', 'premium']):
                    mapping[str(val)] = 'Luxury'
            return mapping
    
    def _create_price_ranges(self, df: pd.DataFrame, price_columns: List[str], report: Dict) -> pd.DataFrame:
        """Create price range categories for numeric price columns"""
        
        for col in price_columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                # Create price ranges using AI
                price_ranges = self._determine_price_ranges(df, col)
                
                if price_ranges:
                    range_col = f"{col}_range"
                    df[range_col] = pd.cut(df[col], 
                                         bins=price_ranges['bins'], 
                                         labels=price_ranges['labels'],
                                         include_lowest=True)
                    
                    range_msg = f"Created price ranges for '{col}'"
                    self.normalizations_applied.append(range_msg)
                    report['price_ranges_created'].append(range_col)
        
        return df
    
    def _determine_price_ranges(self, df: pd.DataFrame, col: str) -> Dict[str, Any] | None:
        """Use AI to determine appropriate price ranges"""
        
        price_stats = df[col].describe()
        
        messages = [
            {"role": "system", "content": """Determine appropriate price ranges for creating categories.

Consider the data distribution and create meaningful ranges like:
- Budget: $0-25
- Moderate: $25-50  
- Expensive: $50-100
- Luxury: $100+

Return JSON:
{
    "bins": [0, 25, 50, 100, float('inf')],
    "labels": ["Budget", "Moderate", "Expensive", "Luxury"]
}

Or return null if ranges don't make sense for this data."""},
            {"role": "user", "content": f"""
Column: {col}
Price statistics: {price_stats.to_dict()}

What price ranges should be created?
"""}
        ]
        
        try:
            response = self.call_llm(messages)
            ranges = json.loads(response.choices[0].message.content)
            return ranges
        except:
            return None
    
    def _standardize_currency_formats(self, df: pd.DataFrame, report: Dict) -> pd.DataFrame:
        """Standardize currency representations across the dataset"""
        
        currency_columns = []
        
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if column contains currency symbols
                sample_values = df[col].dropna().head(20).astype(str).tolist()
                has_currency = any(any(sym in val for sym in ['$', '£', '€', '¥', '₹']) for val in sample_values)
                
                if has_currency:
                    currency_columns.append(col)
                    df = self._normalize_currency_column(df, col, report)
        
        return df
    
    def _normalize_currency_column(self, df: pd.DataFrame, col: str, report: Dict) -> pd.DataFrame:
        """Normalize currency format in a specific column"""
        
        # Standardize currency symbols and formatting
        df[col] = df[col].astype(str).str.replace(r'[\$£€¥₹]+', '$', regex=True)  # Standardize to $
        df[col] = df[col].str.replace(r'(\d),(\d)', r'\1\2', regex=True)  # Remove commas
        
        # Extract numeric values if possible
        numeric_col = f"{col}_amount"
        df[numeric_col] = df[col].str.extract(r'(\d+(?:\.\d{2})?)', expand=False)
        df[numeric_col] = pd.to_numeric(df[numeric_col], errors='coerce')
        
        currency_msg = f"Standardized currency format in '{col}'"
        self.normalizations_applied.append(currency_msg)
        report['currency_standardizations'].append(currency_msg)
        
        return df 