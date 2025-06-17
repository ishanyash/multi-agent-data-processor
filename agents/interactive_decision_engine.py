#!/usr/bin/env python3
"""
Interactive Decision Engine for Multi-Agent Data Processor
Uses OpenAI to analyze data and present intelligent options to users
"""
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Any, Optional, Tuple
from agents.base_agent_simple import BaseAgent

class InteractiveDecisionEngine(BaseAgent):
    """AI-powered decision engine that analyzes data issues and suggests solutions"""
    
    def __init__(self):
        super().__init__("InteractiveDecisionEngine", "gpt-4")
        self.analysis_cache = {}
        self.decision_log = []
    
    def analyze_data_issues(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive analysis of data quality issues"""
        
        issues = {
            'missing_values': self._analyze_missing_values(df),
            'duplicates': self._analyze_duplicates(df),
            'outliers': self._analyze_outliers(df),
            'data_types': self._analyze_data_types(df),
            'data_consistency': self._analyze_consistency(df),
            'data_distribution': self._analyze_distribution(df)
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
                
                # Analyze missing patterns
                patterns = self._analyze_missing_patterns(df, col)
                
                missing_info[col] = {
                    'count': int(missing_count),
                    'percentage': float(missing_percentage),
                    'patterns': patterns,
                    'severity': self._classify_missing_severity(missing_percentage),
                    'recommendations': self._get_missing_value_recommendations(df, col, missing_percentage)
                }
        
        return missing_info
    
    def _analyze_missing_patterns(self, df: pd.DataFrame, col: str) -> Dict[str, Any]:
        """Analyze patterns in missing data"""
        patterns = {}
        
        # Check if missing values correlate with other columns
        for other_col in df.columns:
            if other_col != col and df[other_col].dtype in ['object', 'category']:
                correlation = df.groupby(other_col)[col].apply(lambda x: x.isnull().sum() / len(x))
                if correlation.var() > 0.1:  # Significant variation
                    patterns[f'correlated_with_{other_col}'] = correlation.to_dict()
        
        # Check for sequential patterns
        missing_indices = df[df[col].isnull()].index.tolist()
        if len(missing_indices) > 1:
            gaps = np.diff(missing_indices)
            if np.std(gaps) < np.mean(gaps) * 0.5:  # Regular intervals
                patterns['sequential_pattern'] = {
                    'type': 'regular_intervals',
                    'average_gap': float(np.mean(gaps))
                }
        
        return patterns
    
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
            recommendations.extend([
                {
                    'method': 'mean_imputation',
                    'description': f'Fill with mean value ({df[col].mean():.2f})',
                    'pros': ['Simple', 'Preserves overall distribution'],
                    'cons': ['Reduces variance', 'May not reflect true patterns'],
                    'suitability': 'good' if percentage < 20 else 'fair'
                },
                {
                    'method': 'median_imputation',
                    'description': f'Fill with median value ({df[col].median():.2f})',
                    'pros': ['Robust to outliers', 'Simple'],
                    'cons': ['May not reflect true patterns'],
                    'suitability': 'good' if percentage < 15 else 'fair'
                },
                {
                    'method': 'interpolation',
                    'description': 'Use linear interpolation based on surrounding values',
                    'pros': ['Preserves trends', 'More realistic values'],
                    'cons': ['Assumes linear relationships'],
                    'suitability': 'good' if percentage < 30 else 'poor'
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
                    'description': 'Create new category "Unknown" or "Missing"',
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
            },
            {
                'method': 'drop_column',
                'description': f'Remove entire column "{col}"',
                'pros': ['Eliminates problem completely'],
                'cons': ['Loss of potentially valuable information'],
                'suitability': 'fair' if percentage > 50 else 'poor'
            }
        ])
        
        return recommendations
    
    def _analyze_duplicates(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze duplicate rows"""
        total_duplicates = df.duplicated().sum()
        
        if total_duplicates == 0:
            return {'count': 0, 'percentage': 0.0, 'analysis': 'No duplicates found'}
        
        # Analyze duplicate patterns
        duplicate_analysis = {
            'count': int(total_duplicates),
            'percentage': float((total_duplicates / len(df)) * 100),
            'complete_duplicates': int(df.duplicated(keep=False).sum()),
            'recommendations': [
                {
                    'method': 'remove_duplicates',
                    'description': f'Remove {total_duplicates} duplicate rows',
                    'pros': ['Cleaner dataset', 'Reduced bias'],
                    'cons': ['Potential data loss if duplicates are valid'],
                    'suitability': 'good'
                },
                {
                    'method': 'mark_duplicates',
                    'description': 'Add flag column to mark duplicates',
                    'pros': ['Preserves all data', 'Maintains transparency'],
                    'cons': ['Larger dataset', 'May need special handling'],
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
                    'outlier_values': outliers[col].tolist()[:10],  # First 10 outliers
                    'recommendations': self._get_outlier_recommendations(col, outlier_count, len(df))
                }
        
        return outlier_analysis
    
    def _get_outlier_recommendations(self, col: str, outlier_count: int, total_count: int) -> List[Dict[str, Any]]:
        """Get recommendations for handling outliers"""
        percentage = (outlier_count / total_count) * 100
        
        return [
            {
                'method': 'cap_outliers',
                'description': f'Cap outliers to bounds (replace {outlier_count} values)',
                'pros': ['Preserves data points', 'Reduces extreme influence'],
                'cons': ['Changes actual values', 'May lose important information'],
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
                'method': 'transform_data',
                'description': f'Apply log/sqrt transformation to reduce outlier impact',
                'pros': ['Preserves all data', 'Natural distribution'],
                'cons': ['Changes interpretation', 'May complicate analysis'],
                'suitability': 'fair'
            },
            {
                'method': 'keep_outliers',
                'description': 'Keep outliers as they may represent important patterns',
                'pros': ['No data loss', 'Preserves all information'],
                'cons': ['May skew analysis', 'Affects model performance'],
                'suitability': 'fair'
            }
        ]
    
    def _analyze_data_types(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data type issues"""
        type_issues = {}
        
        for col in df.columns:
            issues = []
            current_type = str(df[col].dtype)
            
            # Check if object columns could be numerical
            if df[col].dtype == 'object':
                try:
                    pd.to_numeric(df[col], errors='coerce')
                    numeric_conversion_rate = (df[col].notna() & pd.to_numeric(df[col], errors='coerce').notna()).sum() / len(df)
                    if numeric_conversion_rate > 0.8:
                        issues.append({
                            'issue': 'potential_numeric',
                            'description': f'Column appears to be numeric but stored as text',
                            'confidence': numeric_conversion_rate
                        })
                except:
                    pass
                
                # Check for datetime
                if any(keyword in col.lower() for keyword in ['date', 'time', 'created', 'updated']):
                    issues.append({
                        'issue': 'potential_datetime',
                        'description': 'Column name suggests datetime but stored as text',
                        'confidence': 0.7
                    })
            
            if issues:
                type_issues[col] = {
                    'current_type': current_type,
                    'issues': issues,
                    'recommendations': self._get_type_conversion_recommendations(col, issues)
                }
        
        return type_issues
    
    def _get_type_conversion_recommendations(self, col: str, issues: List[Dict]) -> List[Dict[str, Any]]:
        """Get type conversion recommendations"""
        recommendations = []
        
        for issue in issues:
            if issue['issue'] == 'potential_numeric':
                recommendations.append({
                    'method': 'convert_to_numeric',
                    'description': f'Convert "{col}" to numeric type',
                    'pros': ['Enables numerical operations', 'Proper data type'],
                    'cons': ['May lose non-numeric values', 'Conversion errors'],
                    'suitability': 'good' if issue['confidence'] > 0.9 else 'fair'
                })
            elif issue['issue'] == 'potential_datetime':
                recommendations.append({
                    'method': 'convert_to_datetime',
                    'description': f'Convert "{col}" to datetime type',
                    'pros': ['Enables time operations', 'Proper data type'],
                    'cons': ['May lose malformed dates', 'Conversion errors'],
                    'suitability': 'fair'
                })
        
        return recommendations
    
    def _analyze_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data consistency issues"""
        consistency_issues = {}
        
        # Check for inconsistent categorical values
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            unique_values = df[col].dropna().unique()
            
            # Check for potential inconsistencies (case sensitivity, whitespace)
            normalized_values = set()
            potential_duplicates = []
            
            for value in unique_values:
                normalized = str(value).lower().strip()
                if normalized in normalized_values:
                    potential_duplicates.append(value)
                normalized_values.add(normalized)
            
            if potential_duplicates:
                consistency_issues[col] = {
                    'issue_type': 'case_whitespace_inconsistency',
                    'potential_duplicates': potential_duplicates,
                    'total_unique': len(unique_values),
                    'recommendations': [
                        {
                            'method': 'normalize_values',
                            'description': f'Standardize case and whitespace in "{col}"',
                            'pros': ['Consistent data', 'Better analysis'],
                            'cons': ['May change original values'],
                            'suitability': 'good'
                        }
                    ]
                }
        
        return consistency_issues
    
    def _analyze_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data distribution patterns"""
        distribution_analysis = {}
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            skewness = df[col].skew()
            kurtosis = df[col].kurtosis()
            
            distribution_analysis[col] = {
                'skewness': float(skewness),
                'kurtosis': float(kurtosis),
                'distribution_type': self._classify_distribution(skewness, kurtosis),
                'recommendations': self._get_distribution_recommendations(col, skewness, kurtosis)
            }
        
        return distribution_analysis
    
    def _classify_distribution(self, skewness: float, kurtosis: float) -> str:
        """Classify distribution type"""
        if abs(skewness) < 0.5 and abs(kurtosis) < 3:
            return 'normal'
        elif skewness > 1:
            return 'highly_right_skewed'
        elif skewness < -1:
            return 'highly_left_skewed'
        elif kurtosis > 3:
            return 'heavy_tailed'
        else:
            return 'moderately_skewed'
    
    def _get_distribution_recommendations(self, col: str, skewness: float, kurtosis: float) -> List[Dict[str, Any]]:
        """Get distribution normalization recommendations"""
        recommendations = []
        
        if abs(skewness) > 1:
            recommendations.append({
                'method': 'log_transform',
                'description': f'Apply log transformation to reduce skewness',
                'pros': ['More normal distribution', 'Better for modeling'],
                'cons': ['Changes interpretation', 'Cannot handle zero/negative values'],
                'suitability': 'good' if skewness > 1 else 'fair'
            })
        
        if abs(kurtosis) > 3:
            recommendations.append({
                'method': 'outlier_treatment',
                'description': 'Address heavy tails through outlier treatment',
                'pros': ['More normal distribution'],
                'cons': ['May lose important information'],
                'suitability': 'fair'
            })
        
        return recommendations
    
    def _get_ai_analysis(self, df: pd.DataFrame, issues: Dict[str, Any]) -> Dict[str, Any]:
        """Get AI-powered analysis of data issues"""
        
        # Prepare summary for LLM
        summary = {
            'dataset_shape': df.shape,
            'columns': df.columns.tolist(),
            'data_types': df.dtypes.astype(str).to_dict(),
            'issues_summary': {
                'missing_values_columns': len(issues['missing_values']),
                'duplicate_rows': issues['duplicates'].get('count', 0),
                'outlier_columns': len(issues['outliers']),
                'type_issues_columns': len(issues['data_types']),
                'consistency_issues_columns': len(issues['data_consistency'])
            }
        }
        
        messages = [
            {"role": "system", "content": """You are an expert data analyst AI. Analyze the provided data quality issues and provide:
1. Overall data quality assessment (score 0-100)
2. Priority ranking of issues to address
3. Business impact assessment
4. Recommended processing strategy
5. Potential risks and mitigation strategies

Be concise but thorough. Focus on actionable insights."""},
            {"role": "user", "content": f"""
Analyze this dataset and its quality issues:

Dataset Overview:
{json.dumps(summary, indent=2)}

Detailed Issues:
- Missing Values: {len(issues['missing_values'])} columns affected
- Duplicates: {issues['duplicates'].get('count', 0)} rows
- Outliers: {len(issues['outliers'])} columns affected
- Data Type Issues: {len(issues['data_types'])} columns
- Consistency Issues: {len(issues['data_consistency'])} columns

Provide your analysis in JSON format with the following structure:
{{
    "quality_score": <0-100>,
    "priority_issues": ["issue1", "issue2", ...],
    "business_impact": "high/medium/low",
    "recommended_strategy": "brief description",
    "risks": ["risk1", "risk2", ...],
    "mitigation": ["action1", "action2", ...]
}}
"""}
        ]
        
        try:
            response = self.call_llm(messages)
            ai_analysis = json.loads(response.choices[0].message.content)
        except Exception as e:
            self.logger.error(f"AI analysis failed: {str(e)}")
            ai_analysis = {
                "quality_score": 50,
                "priority_issues": ["missing_values", "duplicates"],
                "business_impact": "medium",
                "recommended_strategy": "Address missing values and duplicates first",
                "risks": ["Biased analysis due to data quality issues"],
                "mitigation": ["Clean data before analysis", "Document all transformations"]
            }
        
        return ai_analysis
    
    def generate_user_decisions_interface(self, issues: Dict[str, Any]) -> Dict[str, Any]:
        """Generate structured decision points for user interface"""
        
        decision_points = []
        
        # Missing values decisions
        for col, info in issues['missing_values'].items():
            decision_points.append({
                'type': 'missing_values',
                'column': col,
                'title': f'How to handle {info["count"]} missing values in "{col}"?',
                'description': f'{info["percentage"]:.1f}% of data is missing (severity: {info["severity"]})',
                'options': info['recommendations'],
                'ai_recommendation': self._get_ai_recommendation(info['recommendations']),
                'requires_user_input': info['severity'] in ['high', 'critical']
            })
        
        # Duplicate decisions
        if issues['duplicates']['count'] > 0:
            decision_points.append({
                'type': 'duplicates',
                'title': f'How to handle {issues["duplicates"]["count"]} duplicate rows?',
                'description': f'{issues["duplicates"]["percentage"]:.1f}% of data is duplicated',
                'options': issues['duplicates']['recommendations'],
                'ai_recommendation': self._get_ai_recommendation(issues['duplicates']['recommendations']),
                'requires_user_input': issues['duplicates']['percentage'] > 10
            })
        
        # Outlier decisions
        for col, info in issues['outliers'].items():
            decision_points.append({
                'type': 'outliers',
                'column': col,
                'title': f'How to handle {info["count"]} outliers in "{col}"?',
                'description': f'{info["percentage"]:.1f}% of values are outliers',
                'options': info['recommendations'],
                'ai_recommendation': self._get_ai_recommendation(info['recommendations']),
                'requires_user_input': info['percentage'] > 5
            })
        
        return {
            'decision_points': decision_points,
            'ai_insights': issues['ai_insights'],
            'summary': {
                'total_decisions': len(decision_points),
                'user_input_required': sum(1 for dp in decision_points if dp['requires_user_input']),
                'automatic_decisions': sum(1 for dp in decision_points if not dp['requires_user_input'])
            }
        }
    
    def _get_ai_recommendation(self, recommendations: List[Dict[str, Any]]) -> str:
        """Get AI's top recommendation from list"""
        # Simple scoring based on suitability
        scores = {'good': 3, 'fair': 2, 'poor': 1}
        
        best_rec = max(recommendations, key=lambda x: scores.get(x.get('suitability', 'poor'), 0))
        return best_rec['method']
    
    def apply_decisions(self, df: pd.DataFrame, decisions: Dict[str, Any]) -> Tuple[pd.DataFrame, List[str]]:
        """Apply user decisions to the dataset"""
        
        processed_df = df.copy()
        processing_log = []
        
        for decision in decisions.get('decisions', []):
            decision_type = decision['type']
            method = decision['method']
            
            try:
                if decision_type == 'missing_values':
                    processed_df, log_entry = self._apply_missing_value_decision(
                        processed_df, decision['column'], method, decision.get('parameters', {})
                    )
                elif decision_type == 'duplicates':
                    processed_df, log_entry = self._apply_duplicate_decision(
                        processed_df, method, decision.get('parameters', {})
                    )
                elif decision_type == 'outliers':
                    processed_df, log_entry = self._apply_outlier_decision(
                        processed_df, decision['column'], method, decision.get('parameters', {})
                    )
                
                processing_log.append(log_entry)
                
            except Exception as e:
                error_log = f"Failed to apply {method} for {decision_type}: {str(e)}"
                processing_log.append(error_log)
                self.logger.error(error_log)
        
        return processed_df, processing_log
    
    def _apply_missing_value_decision(self, df: pd.DataFrame, column: str, method: str, parameters: Dict) -> Tuple[pd.DataFrame, str]:
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
            
        elif method == 'interpolation':
            df[column].interpolate(inplace=True)
            log_entry = f"Filled {original_missing} missing values in '{column}' using interpolation"
            
        elif method == 'drop_rows':
            initial_rows = len(df)
            df.dropna(subset=[column], inplace=True)
            rows_dropped = initial_rows - len(df)
            log_entry = f"Dropped {rows_dropped} rows with missing values in '{column}'"
            
        elif method == 'drop_column':
            df.drop(columns=[column], inplace=True)
            log_entry = f"Dropped column '{column}' due to missing values"
            
        else:
            log_entry = f"Unknown method '{method}' for missing values in '{column}'"
        
        return df, log_entry
    
    def _apply_duplicate_decision(self, df: pd.DataFrame, method: str, parameters: Dict) -> Tuple[pd.DataFrame, str]:
        """Apply duplicate handling decision"""
        
        original_duplicates = df.duplicated().sum()
        
        if method == 'remove_duplicates':
            initial_rows = len(df)
            df.drop_duplicates(inplace=True)
            rows_removed = initial_rows - len(df)
            log_entry = f"Removed {rows_removed} duplicate rows"
            
        elif method == 'mark_duplicates':
            df['is_duplicate'] = df.duplicated(keep=False)
            log_entry = f"Added duplicate flag column (marked {original_duplicates} duplicates)"
            
        else:
            log_entry = f"Unknown method '{method}' for duplicates"
        
        return df, log_entry
    
    def _apply_outlier_decision(self, df: pd.DataFrame, column: str, method: str, parameters: Dict) -> Tuple[pd.DataFrame, str]:
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
            
        elif method == 'transform_data':
            # Apply log transformation (add 1 to handle zeros)
            df[f'{column}_log'] = np.log1p(df[column] - df[column].min() + 1)
            log_entry = f"Applied log transformation to '{column}' (new column: {column}_log)"
            
        elif method == 'keep_outliers':
            log_entry = f"Kept {original_outliers} outliers in '{column}' unchanged"
            
        else:
            log_entry = f"Unknown method '{method}' for outliers in '{column}'"
        
        return df, log_entry 