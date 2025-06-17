import pandas as pd
import numpy as np
from agents.base_agent_simple import BaseAgent
from typing import Dict, Any, List
import json

class DataCleaningAgent(BaseAgent):
    def __init__(self):
        super().__init__("DataCleaner", "gpt-3.5-turbo")
        
    def create_system_prompt(self) -> str:
        return """You are a Data Cleaning Agent specialized in intelligent data cleaning.
        
        Your responsibilities:
        1. Handle missing values with context-aware strategies
        2. Detect and clean outliers intelligently
        3. Standardize data formats
        4. Remove duplicates with smart deduplication
        5. Fix data type issues
        
        Always explain your cleaning decisions and provide alternatives when appropriate."""
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Main data cleaning process"""
        dataset_path = data.get("dataset_path")
        cleaning_strategy = data.get("cleaning_strategy", {})
        
        # Load dataset
        df = pd.read_csv(dataset_path)
        original_shape = df.shape
        
        # Create cleaning plan
        cleaning_plan = self._create_cleaning_plan(df, cleaning_strategy)
        
        # Apply cleaning steps
        cleaned_df, cleaning_log = self._apply_cleaning(df, cleaning_plan)
        
        # Save cleaned dataset
        output_path = dataset_path.replace('.csv', '_cleaned.csv')
        cleaned_df.to_csv(output_path, index=False)
        
        return {
            "status": "cleaning_complete",
            "original_shape": original_shape,
            "cleaned_shape": cleaned_df.shape,
            "output_path": output_path,
            "cleaning_plan": cleaning_plan,
            "cleaning_log": cleaning_log,
            "data_quality_improvement": self._calculate_improvement(df, cleaned_df)
        }
    
    def _create_cleaning_plan(self, df: pd.DataFrame, strategy: Dict) -> Dict:
        """Create intelligent cleaning plan using LLM"""
        # Analyze data issues
        issues = self._identify_issues(df)
        
        messages = [
            {"role": "system", "content": self.create_system_prompt()},
            {"role": "user", "content": f"""
            Create a data cleaning plan for this dataset:
            
            Dataset shape: {df.shape}
            Identified issues: {json.dumps(issues, indent=2)}
            User strategy preferences: {json.dumps(strategy, indent=2)}
            
            Create a step-by-step cleaning plan with:
            1. Priority order of cleaning steps
            2. Specific methods for each issue
            3. Rationale for each decision
            4. Alternative approaches if primary method fails
            
            Respond in JSON format.
            """}
        ]
        
        try:
            response = self.call_llm(messages)
            plan = json.loads(response.choices[0].message.content)
        except (json.JSONDecodeError, Exception):
            # Fallback plan if LLM fails
            plan = {
                "cleaning_steps": [
                    {"step": "remove_duplicates", "method": "drop_duplicates"},
                    {"step": "handle_missing_values", "method": "smart_imputation", "columns": df.columns.tolist()},
                    {"step": "handle_outliers", "method": "iqr", "columns": df.select_dtypes(include=[np.number]).columns.tolist()},
                    {"step": "fix_data_types", "method": "auto_convert", "conversions": {}}
                ]
            }
        
        return plan
    
    def _identify_issues(self, df: pd.DataFrame) -> Dict:
        """Identify specific data quality issues"""
        issues = {
            "missing_values": {},
            "duplicates": {},
            "outliers": {},
            "data_types": {},
            "formatting": {}
        }
        
        # Missing values analysis
        missing = df.isnull().sum()
        issues["missing_values"] = {
            col: {"count": int(missing[col]), "percentage": float(missing[col] / len(df) * 100)}
            for col in missing.index if missing[col] > 0
        }
        
        # Duplicate analysis
        duplicate_count = df.duplicated().sum()
        issues["duplicates"] = {
            "total_duplicates": int(duplicate_count),
            "percentage": float(duplicate_count / len(df) * 100)
        }
        
        # Data type issues
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check for mixed types or convertible data
                sample_values = df[col].dropna().head(100).tolist()
                issues["data_types"][col] = {
                    "current_type": str(df[col].dtype),
                    "sample_values": sample_values[:5]
                }
        
        return issues
    
    def _apply_cleaning(self, df: pd.DataFrame, plan: Dict) -> tuple:
        """Apply cleaning plan and log all changes"""
        cleaned_df = df.copy()
        cleaning_log = []
        
        for step in plan.get("cleaning_steps", []):
            step_name = step.get("step")
            
            if step_name == "handle_missing_values":
                cleaned_df, log_entry = self._handle_missing_values(cleaned_df, step)
            elif step_name == "remove_duplicates":
                cleaned_df, log_entry = self._remove_duplicates(cleaned_df, step)
            elif step_name == "fix_data_types":
                cleaned_df, log_entry = self._fix_data_types(cleaned_df, step)
            elif step_name == "handle_outliers":
                cleaned_df, log_entry = self._handle_outliers(cleaned_df, step)
            else:
                log_entry = {"step": step_name, "changes": ["Step not implemented"]}
            
            cleaning_log.append(log_entry)
        
        return cleaned_df, cleaning_log
    
    def _handle_missing_values(self, df: pd.DataFrame, step: Dict) -> tuple:
        """Handle missing values based on strategy"""
        method = step.get("method", "drop")
        columns = step.get("columns", df.columns.tolist())
        
        log_entry = {"step": "handle_missing_values", "changes": []}
        
        for col in columns:
            if col not in df.columns:
                continue
                
            missing_count = df[col].isnull().sum()
            if missing_count == 0:
                continue
            
            if method == "drop_rows":
                df = df.dropna(subset=[col])
                log_entry["changes"].append(f"Dropped {missing_count} rows with missing {col}")
            
            elif method == "smart_imputation" or method == "mean_imputation":
                if df[col].dtype in ['int64', 'float64']:
                    mean_value = df[col].mean()
                    df[col].fillna(mean_value, inplace=True)
                    log_entry["changes"].append(f"Filled {missing_count} missing values in {col} with mean: {mean_value:.2f}")
                else:
                    mode_value = df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown"
                    df[col].fillna(mode_value, inplace=True)
                    log_entry["changes"].append(f"Filled {missing_count} missing values in {col} with mode: {mode_value}")
            
            elif method == "mode_imputation":
                mode_value = df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown"
                df[col].fillna(mode_value, inplace=True)
                log_entry["changes"].append(f"Filled {missing_count} missing values in {col} with mode: {mode_value}")
            
            elif method == "forward_fill":
                df[col].fillna(method='ffill', inplace=True)
                log_entry["changes"].append(f"Forward filled {missing_count} missing values in {col}")
        
        return df, log_entry
    
    def _remove_duplicates(self, df: pd.DataFrame, step: Dict) -> tuple:
        """Remove duplicate rows"""
        initial_count = len(df)
        df = df.drop_duplicates()
        removed_count = initial_count - len(df)
        
        log_entry = {
            "step": "remove_duplicates",
            "changes": [f"Removed {removed_count} duplicate rows"]
        }
        
        return df, log_entry
    
    def _fix_data_types(self, df: pd.DataFrame, step: Dict) -> tuple:
        """Fix data type issues"""
        conversions = step.get("conversions", {})
        log_entry = {"step": "fix_data_types", "changes": []}
        
        for col, target_type in conversions.items():
            if col not in df.columns:
                continue
                
            try:
                if target_type == "numeric":
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                elif target_type == "datetime":
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                elif target_type == "category":
                    df[col] = df[col].astype('category')
                
                log_entry["changes"].append(f"Converted {col} to {target_type}")
            except Exception as e:
                log_entry["changes"].append(f"Failed to convert {col} to {target_type}: {str(e)}")
        
        return df, log_entry
    
    def _handle_outliers(self, df: pd.DataFrame, step: Dict) -> tuple:
        """Handle outliers based on strategy"""
        method = step.get("method", "iqr")
        columns = step.get("columns", [])
        
        log_entry = {"step": "handle_outliers", "changes": []}
        
        for col in columns:
            if col not in df.columns or df[col].dtype not in ['int64', 'float64']:
                continue
            
            if method == "iqr":
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                outlier_count = outlier_mask.sum()
                
                # Cap outliers instead of removing them
                df.loc[df[col] < lower_bound, col] = lower_bound
                df.loc[df[col] > upper_bound, col] = upper_bound
                
                log_entry["changes"].append(f"Capped {outlier_count} outliers in {col}")
        
        return df, log_entry
    
    def _calculate_improvement(self, original_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> Dict:
        """Calculate data quality improvement metrics"""
        original_missing = original_df.isnull().sum().sum()
        cleaned_missing = cleaned_df.isnull().sum().sum()
        
        original_duplicates = original_df.duplicated().sum()
        cleaned_duplicates = cleaned_df.duplicated().sum()
        
        return {
            "missing_values_reduced": int(original_missing - cleaned_missing),
            "duplicates_removed": int(original_duplicates - cleaned_duplicates),
            "rows_retained": cleaned_df.shape[0] / original_df.shape[0] * 100,
            "data_completeness": (1 - cleaned_missing / (cleaned_df.shape[0] * cleaned_df.shape[1])) * 100 if cleaned_df.shape[0] > 0 else 0
        }
