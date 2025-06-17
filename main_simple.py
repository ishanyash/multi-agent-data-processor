import argparse
import pandas as pd
import json
import logging
from pathlib import Path
from typing import Dict

# Import simplified agents
from agents.data_profiler_simple import DataProfilerAgent
from agents.data_cleaning_simple import DataCleaningAgent

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleDataProcessingPipeline:
    def __init__(self):
        self.agents = {
            "profiler": DataProfilerAgent(),
            "cleaner": DataCleaningAgent()
        }
    
    def process_dataset(self, dataset_path: str, job_id: str = None) -> Dict:
        """Main pipeline execution (synchronous)"""
        if not job_id:
            import time
            job_id = f"job_{int(time.time())}"
        
        logger.info(f"Starting processing pipeline for job: {job_id}")
        
        # Step 1: Data Profiling
        logger.info("Step 1: Data Profiling")
        profiling_result = self.agents["profiler"].process({
            "dataset_path": dataset_path,
            "job_id": job_id
        })
        
        logger.info(f"Profiling complete. Found {profiling_result.get('dataset_shape', 'unknown')} shape dataset")
        
        # Step 2: Data Cleaning
        logger.info("Step 2: Data Cleaning")
        cleaning_strategy = {
            "missing_values": "smart_imputation",
            "outliers": "cap",
            "duplicates": "remove",
            "data_types": "auto_convert"
        }
        
        cleaning_result = self.agents["cleaner"].process({
            "dataset_path": dataset_path,
            "cleaning_strategy": cleaning_strategy,
            "profiling_insights": profiling_result.get("llm_insights", {}),
            "job_id": job_id
        })
        
        logger.info(f"Cleaning complete. Output: {cleaning_result.get('output_path')}")
        
        # Step 3: Final Quality Assessment
        logger.info("Step 3: Final Quality Assessment")
        final_profiling = self.agents["profiler"].process({
            "dataset_path": cleaning_result.get("output_path"),
            "job_id": job_id
        })
        
        # Compile final results
        final_results = {
            "job_id": job_id,
            "status": "completed",
            "input_dataset": dataset_path,
            "output_dataset": cleaning_result.get("output_path"),
            "processing_summary": {
                "initial_profiling": profiling_result,
                "cleaning_results": cleaning_result,
                "final_assessment": final_profiling
            },
            "quality_improvement": cleaning_result.get("data_quality_improvement", {}),
            "recommendations": final_profiling.get("recommendations", [])
        }
        
        # Save results
        results_path = f"output/processing_results_{job_id}.json"
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        logger.info(f"Pipeline completed. Results saved to: {results_path}")
        return final_results

def main():
    parser = argparse.ArgumentParser(description='Simple Multi-Agent Data Preprocessing Pipeline')
    parser.add_argument('--dataset', required=True, help='Path to input dataset (CSV)')
    parser.add_argument('--job-id', help='Custom job ID')
    parser.add_argument('--output-dir', default='output', help='Output directory')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    Path(args.output_dir).mkdir(exist_ok=True)
    Path('data').mkdir(exist_ok=True)
    
    # Validate input file
    if not Path(args.dataset).exists():
        logger.error(f"Dataset file not found: {args.dataset}")
        return
    
    # Initialize and run pipeline
    pipeline = SimpleDataProcessingPipeline()
    
    try:
        results = pipeline.process_dataset(
            dataset_path=args.dataset,
            job_id=args.job_id
        )
        
        print("\n" + "="*50)
        print("PROCESSING COMPLETED SUCCESSFULLY")
        print("="*50)
        print(f"Job ID: {results['job_id']}")
        print(f"Output Dataset: {results['output_dataset']}")
        print(f"Quality Improvement: {results['quality_improvement']}")
        print("\nKey Recommendations:")
        for rec in results['recommendations'][:3]:
            print(f"  • {rec}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
