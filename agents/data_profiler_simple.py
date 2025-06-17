import pandas as pd
import numpy as np
from agents.base_agent_simple import BaseAgent
from typing import Dict, Any
import json

class DataProfilerAgent(BaseAgent):
    def __init__(self):
        super().__init__("DataProfiler", "gpt-4")
        
    def create_system_prompt(self) -> str:
        return """You are a Data Profiler Agent specialized in comprehensive dataset analysis.
        
        Your responsibilities:
        1. Generate statistical profiles of datasets
        2. Identify data quality issues
        3. Understand business context of data
        4. Create data quality baselines
        5. Suggest optimal processing strategies
        
        Provide detailed, actionable insights about data characteristics."""
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive data profiling"""
        dataset_path = data.get("dataset_path")
        
        # Load and analyze dataset
        df = pd.read_csv(dataset_path)
        
        # Generate statistical profile
        stats_profile = self._generate_statistical_profile(df)
        
        # Analyze data quality
        quality_assessment = self._assess_data_quality(df)
        
        # Generate LLM insights
        llm_insights = self._generate_llm_insights(df, stats_profile, quality_assessment)
        
        return {
            "status": "profiling_complete",
            "dataset_shape": df.shape,
            "statistical_profile": stats_profile,
            "quality_assessment": quality_assessment,
            "llm_insights": llm_insights,
            "recommendations": llm_insights.get("recommendations", [])
        }
    
    def _generate_statistical_profile(self, df: pd.DataFrame) -> Dict:
        """Generate comprehensive statistical profile"""
        profile = {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "missing_percentage": (df.isnull().sum() / len(df) * 100).to_dict(),
            "unique_counts": df.nunique().to_dict(),
            "memory_usage": df.memory_usage(deep=True).sum()
        }
        
        # Numerical statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            profile["numerical_stats"] = df[numeric_cols].describe().to_dict()
        
        # Categorical analysis
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            profile["categorical_stats"] = {}
            for col in categorical_cols:
                profile["categorical_stats"][col] = {
                    "top_values": df[col].value_counts().head(10).to_dict(),
                    "unique_count": df[col].nunique()
                }
        
        return profile
    
    def _assess_data_quality(self, df: pd.DataFrame) -> Dict:
        """Assess data quality across multiple dimensions"""
        quality = {
            "completeness": {
                "overall_completeness": (1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
                "column_completeness": ((1 - df.isnull().sum() / len(df)) * 100).to_dict()
            },
            "consistency": {
                "duplicate_rows": df.duplicated().sum(),
                "duplicate_percentage": (df.duplicated().sum() / len(df)) * 100
            },
            "validity": {
                "data_type_issues": self._detect_type_issues(df),
                "outliers": self._detect_outliers(df)
            }
        }
        
        return quality
    
    def _detect_type_issues(self, df: pd.DataFrame) -> Dict:
        """Detect potential data type issues"""
        issues = {}
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if numeric data is stored as text
                try:
                    pd.to_numeric(df[col], errors='coerce')
                    non_numeric_count = df[col].apply(lambda x: pd.to_numeric(x, errors='coerce')).isnull().sum()
                    if non_numeric_count < len(df) * 0.1:  # Less than 10% non-numeric
                        issues[col] = "potentially_numeric"
                except:
                    pass
        
        return issues
    
    def _detect_outliers(self, df: pd.DataFrame) -> Dict:
        """Detect outliers in numerical columns"""
        outliers = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            outliers[col] = {
                "count": outlier_count,
                "percentage": (outlier_count / len(df)) * 100
            }
        
        return outliers
    
    def _generate_llm_insights(self, df: pd.DataFrame, stats: Dict, quality: Dict) -> Dict:
        """Generate LLM-powered insights about the dataset"""
        # Prepare summary for LLM
        summary = {
            "shape": df.shape,
            "columns": df.columns.tolist()[:10],  # Limit for token efficiency
            "data_types": df.dtypes.astype(str).to_dict(),
            "missing_data": {k: v for k, v in stats["missing_percentage"].items() if v > 0},
            "quality_summary": {
                "completeness": quality["completeness"]["overall_completeness"],
                "duplicates": quality["consistency"]["duplicate_percentage"]
            }
        }
        
        messages = [
            {"role": "system", "content": self.create_system_prompt()},
            {"role": "user", "content": f"""
            Analyze this dataset profile and provide insights:
            
            {json.dumps(summary, indent=2)}
            
            Provide insights on:
            1. Dataset characteristics and potential use cases
            2. Data quality issues and their severity
            3. Recommended preprocessing steps
            4. Potential challenges in processing
            5. Business context interpretation
            
            Respond in JSON format with clear recommendations.
            """}
        ]
        
        response = self.call_llm(messages)
        try:
            insights = json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            insights = {
                "summary": "Dataset profiling completed",
                "recommendations": ["Review data quality", "Handle missing values", "Check for outliers"],
                "issues": ["Some data quality issues detected"],
                "use_cases": ["General data analysis"]
            }
        
        return insights
