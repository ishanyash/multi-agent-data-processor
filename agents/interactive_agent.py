#!/usr/bin/env python3
"""
Interactive Agent for Multi-Agent Data Processor
Uses OpenAI to analyze data and present intelligent options to users
"""
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Any, Optional, Tuple
from agents.base_agent_simple import BaseAgent

class InteractiveDataAgent(BaseAgent):
    """AI-powered interactive agent that analyzes data issues and suggests solutions"""
    
    def __init__(self):
        super().__init__("InteractiveDataAgent", "gpt-4")
        self.analysis_cache = {}
        self.decision_log = []
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Main processing method - analyzes data and returns decision interface"""
        import pandas as pd
        
        if 'dataframe' in data:
            df = data['dataframe']
            
            # Analyze data issues
            issues = self.analyze_data_issues(df)
            
            # Generate decision interface
            decision_interface = self.generate_decision_interface(issues)
            
            return {
                'status': 'success',
                'data_issues': issues,
                'decision_interface': decision_interface,
                'message': f'Analyzed {df.shape[0]} rows and {df.shape[1]} columns'
            }
        else:
            return {
                'status': 'error',
                'message': 'No dataframe provided in data'
            }
    
    def analyze_data_issues(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive analysis of data quality issues"""
        
        issues = {
            'missing_values': self._analyze_missing_values(df),
            'duplicates': self._analyze_duplicates(df),
            'outliers': self._analyze_outliers(df),
            'data_types': self._analyze_data_types(df)
        }
        
        # Get AI insights on the issues
        ai_analysis = self._get_ai_analysis(df, issues)
        issues['ai_insights'] = ai_analysis
        
        return issues
    
    def _analyze_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detailed missing value analysis"""
        missing_info = {}
        
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                missing_percentage = (missing_count / len(df)) * 100
                
                missing_info[col] = {
                    'count': int(missing_count),
                    'percentage': float(missing_percentage),
                    'severity': self._classify_missing_severity(missing_percentage),
                    'recommendations': self._get_missing_value_recommendations(df, col, missing_percentage)
                }
        
        return missing_info
    
    def _classify_missing_severity(self, percentage: float) -> str:
        """Classify severity of missing values"""
        if percentage < 5:
            return 'low'
        elif percentage < 20:
            return 'medium'
        elif percentage < 50:
            return 'high'
        else:
            return 'critical'
    
    def _get_missing_value_recommendations(self, df: pd.DataFrame, col: str, percentage: float) -> List[Dict[str, Any]]:
        """Get AI-powered recommendations for handling missing values"""
        recommendations = []
        
        col_type = df[col].dtype
        
        if col_type in ['int64', 'float64']:
            # Numerical column recommendations
            mean_val = df[col].mean()
            median_val = df[col].median()
            
            recommendations.extend([
                {
                    'method': 'mean_imputation',
                    'description': f'Fill with mean value ({mean_val:.2f})',
                    'pros': ['Simple', 'Preserves overall distribution'],
                    'cons': ['Reduces variance', 'May not reflect true patterns'],
                    'suitability': 'good' if percentage < 20 else 'fair'
                },
                {
                    'method': 'median_imputation',
                    'description': f'Fill with median value ({median_val:.2f})',
                    'pros': ['Robust to outliers', 'Simple'],
                    'cons': ['May not reflect true patterns'],
                    'suitability': 'good' if percentage < 15 else 'fair'
                }
            ])
        else:
            # Categorical column recommendations
            mode_value = df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown'
            recommendations.extend([
                {
                    'method': 'mode_imputation',
                    'description': f'Fill with most frequent value ({mode_value})',
                    'pros': ['Simple', 'Uses existing patterns'],
                    'cons': ['May introduce bias'],
                    'suitability': 'good' if percentage < 15 else 'fair'
                },
                {
                    'method': 'category_unknown',
                    'description': 'Create new category "Unknown"',
                    'pros': ['Preserves missing information', 'No assumptions'],
                    'cons': ['Creates imbalanced categories'],
                    'suitability': 'good'
                }
            ])
        
        # Universal recommendations
        recommendations.extend([
            {
                'method': 'drop_rows',
                'description': f'Remove {int(df[col].isnull().sum())} rows with missing values',
                'pros': ['Clean dataset', 'No imputation bias'],
                'cons': ['Data loss', 'Potential sampling bias'],
                'suitability': 'good' if percentage < 5 else 'poor'
            }
        ])
        
        return recommendations
    
    def _analyze_duplicates(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze duplicate rows"""
        total_duplicates = df.duplicated().sum()
        
        if total_duplicates == 0:
            return {'count': 0, 'percentage': 0.0, 'analysis': 'No duplicates found'}
        
        duplicate_analysis = {
            'count': int(total_duplicates),
            'percentage': float((total_duplicates / len(df)) * 100),
            'recommendations': [
                {
                    'method': 'remove_duplicates',
                    'description': f'Remove {total_duplicates} duplicate rows',
                    'pros': ['Cleaner dataset', 'Reduced bias'],
                    'cons': ['Potential data loss if duplicates are valid'],
                    'suitability': 'good'
                },
                {
                    'method': 'keep_duplicates',
                    'description': 'Keep all rows (duplicates may be valid)',
                    'pros': ['No data loss', 'Preserves all information'],
                    'cons': ['May bias analysis'],
                    'suitability': 'fair'
                }
            ]
        }
        
        return duplicate_analysis
    
    def _analyze_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze outliers in numerical columns"""
        outlier_analysis = {}
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_count = len(outliers)
            
            if outlier_count > 0:
                outlier_analysis[col] = {
                    'count': outlier_count,
                    'percentage': (outlier_count / len(df)) * 100,
                    'bounds': {'lower': float(lower_bound), 'upper': float(upper_bound)},
                    'recommendations': self._get_outlier_recommendations(col, outlier_count, len(df))
                }
        
        return outlier_analysis
    
    def _get_outlier_recommendations(self, col: str, outlier_count: int, total_count: int) -> List[Dict[str, Any]]:
        """Get recommendations for handling outliers"""
        percentage = (outlier_count / total_count) * 100
        
        return [
            {
                'method': 'cap_outliers',
                'description': f'Cap outliers to bounds (affect {outlier_count} values)',
                'pros': ['Preserves data points', 'Reduces extreme influence'],
                'cons': ['Changes actual values'],
                'suitability': 'good' if percentage < 10 else 'fair'
            },
            {
                'method': 'remove_outliers',
                'description': f'Remove {outlier_count} outlier rows',
                'pros': ['Clean distribution', 'No modified values'],
                'cons': ['Data loss', 'Potential bias'],
                'suitability': 'good' if percentage < 5 else 'poor'
            },
            {
                'method': 'keep_outliers',
                'description': 'Keep outliers (may represent important patterns)',
                'pros': ['No data loss', 'Preserves all information'],
                'cons': ['May skew analysis'],
                'suitability': 'fair'
            }
        ]
    
    def _analyze_data_types(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data type issues"""
        type_issues = {}
        
        for col in df.columns:
            current_type = str(df[col].dtype)
            
            # Check if object columns could be numerical
            if df[col].dtype == 'object':
                try:
                    # Simple check for numeric conversion
                    non_null_values = df[col].dropna()
                    if len(non_null_values) > 0:
                        numeric_count = 0
                        for val in non_null_values:
                            try:
                                float(val)
                                numeric_count += 1
                            except (ValueError, TypeError):
                                pass
                        numeric_rate = numeric_count / len(non_null_values)
                    
                    if numeric_rate > 0.8:
                        type_issues[col] = {
                            'current_type': current_type,
                            'suggested_type': 'numeric',
                            'confidence': numeric_rate,
                            'recommendations': [{
                                'method': 'convert_to_numeric',
                                'description': f'Convert "{col}" to numeric type',
                                'pros': ['Enables numerical operations', 'Proper data type'],
                                'cons': ['May lose non-numeric values'],
                                'suitability': 'good' if numeric_rate > 0.9 else 'fair'
                            }]
                        }
                except:
                    pass
        
        return type_issues
    
    def _get_ai_analysis(self, df: pd.DataFrame, issues: Dict[str, Any]) -> Dict[str, Any]:
        """Get AI-powered analysis of data issues"""
        
        # Prepare summary for LLM
        summary = {
            'dataset_shape': df.shape,
            'columns': df.columns.tolist()[:10],  # Limit for token efficiency
            'issues_summary': {
                'missing_values_columns': len(issues['missing_values']),
                'duplicate_rows': issues['duplicates'].get('count', 0),
                'outlier_columns': len(issues['outliers']),
                'type_issues_columns': len(issues['data_types'])
            }
        }
        
        messages = [
            {"role": "system", "content": """You are an expert data analyst AI. Analyze data quality issues and provide:
1. Overall data quality score (0-100)
2. Priority ranking of issues to address
3. Recommended processing strategy
4. Business impact assessment

Be concise and actionable."""},
            {"role": "user", "content": f"""
Analyze this dataset and its quality issues:

Dataset: {df.shape[0]} rows × {df.shape[1]} columns
Issues:
- Missing Values: {len(issues['missing_values'])} columns affected
- Duplicates: {issues['duplicates'].get('count', 0)} rows
- Outliers: {len(issues['outliers'])} columns affected
- Data Type Issues: {len(issues['data_types'])} columns

Provide analysis in JSON format:
{{
    "quality_score": <0-100>,
    "priority_issues": ["issue1", "issue2"],
    "recommended_strategy": "brief description",
    "business_impact": "high/medium/low"
}}
"""}
        ]
        
        try:
            response = self.call_llm(messages)
            ai_analysis = json.loads(response.choices[0].message.content)
        except Exception as e:
            self.logger.error(f"AI analysis failed: {str(e)}")
            ai_analysis = {
                "quality_score": 70,
                "priority_issues": ["missing_values", "duplicates"],
                "recommended_strategy": "Address missing values and duplicates first",
                "business_impact": "medium"
            }
        
        return ai_analysis
    
    def generate_decision_interface(self, issues: Dict[str, Any]) -> Dict[str, Any]:
        """Generate structured decision points for user interface"""
        
        decision_points = []
        
        # Missing values decisions
        for col, info in issues['missing_values'].items():
            decision_points.append({
                'id': f'missing_{col}',
                'type': 'missing_values',
                'column': col,
                'title': f'Missing values in "{col}"',
                'description': f'{info["count"]} missing values ({info["percentage"]:.1f}%)',
                'severity': info['severity'],
                'options': info['recommendations'],
                'requires_user_input': info['severity'] in ['high', 'critical']
            })
        
        # Duplicate decisions
        if issues['duplicates']['count'] > 0:
            decision_points.append({
                'id': 'duplicates',
                'type': 'duplicates',
                'title': 'Duplicate rows detected',
                'description': f'{issues["duplicates"]["count"]} duplicate rows ({issues["duplicates"]["percentage"]:.1f}%)',
                'options': issues['duplicates']['recommendations'],
                'requires_user_input': issues['duplicates']['percentage'] > 10
            })
        
        # Outlier decisions
        for col, info in issues['outliers'].items():
            decision_points.append({
                'id': f'outliers_{col}',
                'type': 'outliers',
                'column': col,
                'title': f'Outliers in "{col}"',
                'description': f'{info["count"]} outliers ({info["percentage"]:.1f}%)',
                'options': info['recommendations'],
                'requires_user_input': info['percentage'] > 5
            })
        
        return {
            'decision_points': decision_points,
            'ai_insights': issues['ai_insights'],
            'summary': {
                'total_decisions': len(decision_points),
                'user_input_required': sum(1 for dp in decision_points if dp['requires_user_input']),
                'auto_decisions': sum(1 for dp in decision_points if not dp['requires_user_input'])
            }
        }
    
    def apply_decisions(self, df: pd.DataFrame, decisions: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, List[str]]:
        """Apply user decisions to the dataset"""
        
        processed_df = df.copy()
        processing_log = []
        
        for decision in decisions:
            decision_type = decision['type']
            method = decision['method']
            
            try:
                if decision_type == 'missing_values':
                    processed_df, log_entry = self._apply_missing_value_decision(
                        processed_df, decision['column'], method
                    )
                elif decision_type == 'duplicates':
                    processed_df, log_entry = self._apply_duplicate_decision(
                        processed_df, method
                    )
                elif decision_type == 'outliers':
                    processed_df, log_entry = self._apply_outlier_decision(
                        processed_df, decision['column'], method
                    )
                
                processing_log.append(log_entry)
                
            except Exception as e:
                error_log = f"Failed to apply {method} for {decision_type}: {str(e)}"
                processing_log.append(error_log)
                self.logger.error(error_log)
        
        return processed_df, processing_log
    
    def _apply_missing_value_decision(self, df: pd.DataFrame, column: str, method: str) -> Tuple[pd.DataFrame, str]:
        """Apply missing value handling decision"""
        
        original_missing = df[column].isnull().sum()
        
        if method == 'mean_imputation':
            df[column].fillna(df[column].mean(), inplace=True)
            log_entry = f"Filled {original_missing} missing values in '{column}' with mean"
            
        elif method == 'median_imputation':
            df[column].fillna(df[column].median(), inplace=True)
            log_entry = f"Filled {original_missing} missing values in '{column}' with median"
            
        elif method == 'mode_imputation':
            mode_value = df[column].mode().iloc[0] if not df[column].mode().empty else 'Unknown'
            df[column].fillna(mode_value, inplace=True)
            log_entry = f"Filled {original_missing} missing values in '{column}' with mode: {mode_value}"
            
        elif method == 'category_unknown':
            df[column].fillna('Unknown', inplace=True)
            log_entry = f"Filled {original_missing} missing values in '{column}' with 'Unknown'"
            
        elif method == 'drop_rows':
            initial_rows = len(df)
            df.dropna(subset=[column], inplace=True)
            rows_dropped = initial_rows - len(df)
            log_entry = f"Dropped {rows_dropped} rows with missing values in '{column}'"
            
        else:
            log_entry = f"Applied {method} to missing values in '{column}'"
        
        return df, log_entry
    
    def _apply_duplicate_decision(self, df: pd.DataFrame, method: str) -> Tuple[pd.DataFrame, str]:
        """Apply duplicate handling decision"""
        
        if method == 'remove_duplicates':
            initial_rows = len(df)
            df.drop_duplicates(inplace=True)
            rows_removed = initial_rows - len(df)
            log_entry = f"Removed {rows_removed} duplicate rows"
            
        elif method == 'keep_duplicates':
            log_entry = "Kept all duplicate rows unchanged"
            
        else:
            log_entry = f"Applied {method} to duplicate rows"
        
        return df, log_entry
    
    def _apply_outlier_decision(self, df: pd.DataFrame, column: str, method: str) -> Tuple[pd.DataFrame, str]:
        """Apply outlier handling decision"""
        
        # Calculate outlier bounds
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
        original_outliers = outlier_mask.sum()
        
        if method == 'cap_outliers':
            df.loc[df[column] < lower_bound, column] = lower_bound
            df.loc[df[column] > upper_bound, column] = upper_bound
            log_entry = f"Capped {original_outliers} outliers in '{column}'"
            
        elif method == 'remove_outliers':
            initial_rows = len(df)
            df = df[~outlier_mask]
            rows_removed = initial_rows - len(df)
            log_entry = f"Removed {rows_removed} outlier rows from '{column}'"
            
        elif method == 'keep_outliers':
            log_entry = f"Kept {original_outliers} outliers in '{column}' unchanged"
            
        else:
            log_entry = f"Applied {method} to outliers in '{column}'"
        
        return df, log_entry 