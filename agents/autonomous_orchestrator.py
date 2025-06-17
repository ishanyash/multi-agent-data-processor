#!/usr/bin/env python3
"""
Autonomous Agent Orchestrator
Coordinates specialized agents to work autonomously on their expertise areas
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Any, Tuple
from agents.base_agent_simple import BaseAgent

class AutonomousOrchestrator(BaseAgent):
    """Orchestrates autonomous specialized agents"""
    
    def __init__(self):
        super().__init__("AutonomousOrchestrator", "gpt-4")
        self.specialized_agents = []
        self.processing_pipeline = []
        self.orchestration_log = []
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Main processing method - orchestrates autonomous agents"""
        
        if 'dataframe' not in data:
            return {'status': 'error', 'message': 'No dataframe provided'}
        
        df = data['dataframe'].copy()
        original_shape = df.shape
        
        # Initialize specialized agents
        self._initialize_agents()
        
        # Autonomous processing pipeline
        df, orchestration_report = self._autonomous_processing_pipeline(df)
        
        return {
            'status': 'success',
            'dataframe': df,
            'original_shape': original_shape,
            'final_shape': df.shape,
            'orchestration_report': orchestration_report,
            'orchestration_log': self.orchestration_log,
            'agent': 'AutonomousOrchestrator'
        }
    
    def _initialize_agents(self):
        """Initialize specialized autonomous agents"""
        
        # Import agents dynamically to avoid circular imports
        try:
            from agents.autonomous_data_validator import AutonomousDataValidator
            from agents.autonomous_text_processor import AutonomousTextProcessor
            from agents.autonomous_price_normalizer import AutonomousPriceNormalizer
            
            self.specialized_agents = [
                AutonomousDataValidator(),
                AutonomousTextProcessor(), 
                AutonomousPriceNormalizer()
            ]
            
            self.orchestration_log.append("Initialized 3 specialized autonomous agents")
            
        except ImportError as e:
            self.logger.warning(f"Could not import some agents: {str(e)}")
            # Fallback to basic processing
            self.specialized_agents = []
    
    def _autonomous_processing_pipeline(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Run autonomous processing pipeline"""
        
        orchestration_report = {
            'agents_executed': [],
            'total_transformations': 0,
            'processing_sequence': [],
            'improvements': {}
        }
        
        # Analyze data to determine optimal agent sequence
        agent_sequence = self._determine_optimal_sequence(df)
        orchestration_report['processing_sequence'] = agent_sequence
        
        # Execute agents in sequence
        for agent_name in agent_sequence:
            agent = self._get_agent_by_name(agent_name)
            
            if agent:
                self.orchestration_log.append(f"Executing {agent_name}...")
                
                # Run agent autonomously
                agent_result = agent.process({'dataframe': df})
                
                if agent_result['status'] == 'success':
                    df = agent_result['dataframe']
                    
                    # Log agent results
                    agent_info = {
                        'agent': agent_name,
                        'transformations': len(agent_result.get('fixes_applied', []) + 
                                               agent_result.get('transformations_applied', []) +
                                               agent_result.get('normalizations_applied', [])),
                        'shape_change': f"{agent_result['original_shape']} → {agent_result['final_shape']}",
                        'report': agent_result.get('validation_report', agent_result.get('processing_report', agent_result.get('normalization_report', {})))
                    }
                    
                    orchestration_report['agents_executed'].append(agent_info)
                    orchestration_report['total_transformations'] += agent_info['transformations']
                    
                    self.orchestration_log.append(f"✅ {agent_name} completed: {agent_info['transformations']} transformations")
                else:
                    self.orchestration_log.append(f"❌ {agent_name} failed: {agent_result.get('message', 'Unknown error')}")
        
        # Calculate overall improvements
        orchestration_report['improvements'] = self._calculate_improvements(df, orchestration_report)
        
        return df, orchestration_report
    
    def _determine_optimal_sequence(self, df: pd.DataFrame) -> List[str]:
        """Use AI to determine optimal agent execution sequence"""
        
        # Analyze data characteristics
        data_analysis = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'data_types': df.dtypes.astype(str).to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'sample_data': df.head(3).to_dict('records')
        }
        
        messages = [
            {"role": "system", "content": """You are a data processing expert. Analyze the dataset and determine the optimal sequence for running autonomous agents.

Available agents:
1. AutonomousDataValidator - Fixes negative values, impossible ranges, removes invalid rows
2. AutonomousTextProcessor - Splits composite fields, normalizes text, extracts structured data
3. AutonomousPriceNormalizer - Standardizes price formats, creates price ranges

Consider dependencies:
- Data validation should usually come first to clean obvious errors
- Text processing should happen before price normalization if prices are in text format
- Price normalization should happen after text is cleaned

Return JSON array of agent names in optimal order: ["agent1", "agent2", "agent3"]"""},
            {"role": "user", "content": f"""
Analyze this dataset and determine optimal agent sequence:

{json.dumps(data_analysis, indent=2, default=str)}

Return JSON array of agent names in execution order.
"""}
        ]
        
        try:
            response = self.call_llm(messages)
            sequence = json.loads(response.choices[0].message.content)
            
            if isinstance(sequence, list):
                return sequence
        except Exception as e:
            self.logger.error(f"AI sequence determination failed: {str(e)}")
        
        # Fallback sequence
        return ["AutonomousDataValidator", "AutonomousTextProcessor", "AutonomousPriceNormalizer"]
    
    def _get_agent_by_name(self, agent_name: str):
        """Get agent instance by name"""
        
        for agent in self.specialized_agents:
            if agent.__class__.__name__ == agent_name:
                return agent
        
        return None
    
    def _calculate_improvements(self, df: pd.DataFrame, report: Dict) -> Dict[str, Any]:
        """Calculate overall data improvements"""
        
        improvements = {
            'data_quality_score': self._calculate_quality_score(df),
            'total_transformations': report['total_transformations'],
            'agents_used': len(report['agents_executed']),
            'new_columns_created': self._count_new_columns(report),
            'data_standardization': self._assess_standardization(df)
        }
        
        return improvements
    
    def _calculate_quality_score(self, df: pd.DataFrame) -> float:
        """Calculate overall data quality score"""
        
        # Basic quality metrics
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        completeness = (total_cells - missing_cells) / total_cells if total_cells > 0 else 0
        
        # Data type consistency
        consistent_types = sum(1 for col in df.columns 
                              if not (df[col].dtype == 'object' and 
                                     pd.api.types.is_numeric_dtype(pd.to_numeric(df[col], errors='coerce'))))
        type_consistency = consistent_types / len(df.columns) if len(df.columns) > 0 else 0
        
        # Overall score (weighted average)
        quality_score = (completeness * 0.6 + type_consistency * 0.4) * 100
        
        return round(quality_score, 1)
    
    def _count_new_columns(self, report: Dict) -> int:
        """Count new columns created by agents"""
        
        new_columns = 0
        for agent_info in report['agents_executed']:
            agent_report = agent_info.get('report', {})
            
            # Check different report formats
            if 'new_columns_created' in agent_report:
                new_columns += len(agent_report['new_columns_created'])
            elif 'price_ranges_created' in agent_report:
                new_columns += len(agent_report['price_ranges_created'])
        
        return new_columns
    
    def _assess_standardization(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess level of data standardization"""
        
        standardization = {
            'text_columns_standardized': 0,
            'price_columns_normalized': 0,
            'categorical_consistency': 0
        }
        
        for col in df.columns:
            col_lower = col.lower()
            
            # Check for standardized text columns
            if '_standardized' in col_lower or '_category' in col_lower:
                standardization['text_columns_standardized'] += 1
            
            # Check for normalized price columns  
            if 'price' in col_lower and ('_numeric' in col_lower or '_range' in col_lower):
                standardization['price_columns_normalized'] += 1
            
            # Check categorical consistency
            if df[col].dtype == 'object':
                unique_count = df[col].nunique()
                total_count = len(df[col].dropna())
                
                if unique_count < total_count * 0.1:  # Likely categorical
                    # Check if values are consistently formatted
                    values = df[col].dropna().astype(str)
                    case_consistency = (values.str.istitle().all() or 
                                       values.str.isupper().all() or 
                                       values.str.islower().all())
                    
                    if case_consistency:
                        standardization['categorical_consistency'] += 1
        
        return standardization


# Simplified autonomous agents for immediate testing
class SimpleAutonomousValidator(BaseAgent):
    """Simplified autonomous validator for immediate testing"""
    
    def __init__(self):
        super().__init__("SimpleAutonomousValidator", "gpt-4")
        self.fixes_applied = []
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data autonomously"""
        
        df = data['dataframe'].copy()
        original_shape = df.shape
        
        # Fix negative review counts
        for col in df.columns:
            if 'review' in col.lower() and 'count' in col.lower():
                if pd.api.types.is_numeric_dtype(df[col]):
                    negative_count = (df[col] < 0).sum()
                    if negative_count > 0:
                        df[col] = df[col].abs()  # Convert to positive
                        self.fixes_applied.append(f"Fixed {negative_count} negative values in '{col}'")
        
        return {
            'status': 'success',
            'dataframe': df,
            'original_shape': original_shape,
            'final_shape': df.shape,
            'fixes_applied': self.fixes_applied,
            'validation_report': {'fixes': self.fixes_applied}
        }


class SimpleAutonomousTextProcessor(BaseAgent):
    """Simplified autonomous text processor for immediate testing"""
    
    def __init__(self):
        super().__init__("SimpleAutonomousTextProcessor", "gpt-4")
        self.transformations_applied = []
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process text data autonomously"""
        
        df = data['dataframe'].copy()
        original_shape = df.shape
        
        # Split cuisine field
        for col in df.columns:
            if 'cuisine' in col.lower():
                if '•' in str(df[col].iloc[0]):
                    # Split by bullet points
                    split_data = df[col].astype(str).str.split('•', expand=True)
                    
                    if split_data.shape[1] >= 2:
                        df[f'{col}_type'] = split_data[1].str.strip()
                        df[f'{col}_location'] = split_data[2].str.strip() if split_data.shape[1] > 2 else None
                        
                        self.transformations_applied.append(f"Split '{col}' into cuisine type and location")
        
        return {
            'status': 'success',
            'dataframe': df,
            'original_shape': original_shape,
            'final_shape': df.shape,
            'transformations_applied': self.transformations_applied,
            'processing_report': {'transformations': self.transformations_applied}
        }


class SimpleAutonomousPriceNormalizer(BaseAgent):
    """Simplified autonomous price normalizer for immediate testing"""
    
    def __init__(self):
        super().__init__("SimpleAutonomousPriceNormalizer", "gpt-4")
        self.normalizations_applied = []
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize price data autonomously"""
        
        df = data['dataframe'].copy()
        original_shape = df.shape
        
        # Normalize price columns
        for col in df.columns:
            if 'price' in col.lower():
                if df[col].dtype == 'object':
                    # Count dollar signs and convert to numeric
                    dollar_mapping = {'$': 1, '$$': 2, '$$$': 3, '$$$$': 4}
                    
                    df[f'{col}_numeric'] = df[col].map(dollar_mapping)
                    df[f'{col}_category'] = df[f'{col}_numeric'].map({
                        1: 'Budget', 2: 'Moderate', 3: 'Expensive', 4: 'Luxury'
                    })
                    
                    self.normalizations_applied.append(f"Converted '{col}' to numeric scale and categories")
        
        return {
            'status': 'success',
            'dataframe': df,
            'original_shape': original_shape,
            'final_shape': df.shape,
            'normalizations_applied': self.normalizations_applied,
            'normalization_report': {'normalizations': self.normalizations_applied}
        } 