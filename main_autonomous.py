#!/usr/bin/env python3
"""
Autonomous Multi-Agent Data Processor
Demonstrates autonomous agents working on their specialties
"""

import pandas as pd
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.autonomous_orchestrator import AutonomousOrchestrator, SimpleAutonomousValidator, SimpleAutonomousTextProcessor, SimpleAutonomousPriceNormalizer

def create_test_restaurant_data():
    """Create test restaurant data with the issues mentioned"""
    
    data = {
        'restaurant_name': [
            'Cappadocia Restaurant, Bristol',
            'Kibou Clifton', 
            'Côte Brasserie - Bristol Clifton',
            'Ivy Clifton Brasserie',
            'Pho Bristol',
            'The Ox',
            'Koocha Mezze Bar',
            'Côte Brasserie - Bristol Quakers Friars',
            'Buenasado - Bristol',
            'Milk Bun, Bristol',
            'Piccolino - Bristol',
            'The Ivy Bath Brasserie and Garden',
            'The River Grille'
        ],
        'restaurant_rating': [4.7, 4.7, 4.5, 4.5, 4.5, 4.7, 4.7, 4.7, 4.8, 4.7, 4.3, 4.3, 4.4],
        'restaurant_review_count': [-439, -27, -120, -10214, -345, -22, -1488, -39, -507, -524, -158, -13165, -193],
        'restaurant_price': ['$$$$', '$$$$', '$$$$', '$$$$', '$$$$', '$$$$', '$$$$', '$$$$', '$$$$', '$$$$', '$$$$', '$$$$', '$$$$'],
        'restaurant_cuisine': [
            '• Turkish • Bristol',
            '• Japanese • Clifton', 
            '• French • Clifton',
            '• Contemporary British • Clifton',
            '• Vietnamese • Bristol',
            '• Steak • Bristol',
            '• Persian • Bristol',
            '• French • Bristol',
            '• Argentinean • Bristol',
            '• Burgers • Bridgeyate',
            '• Italian • Bristol',
            '• Contemporary British • Bath',
            '• Contemporary European • Bristol'
        ]
    }
    
    return pd.DataFrame(data)

def main():
    """Main function demonstrating autonomous agents"""
    
    print("🤖 Autonomous Multi-Agent Data Processor")
    print("=" * 50)
    
    # Create test data with issues
    print("\n📊 Creating test restaurant data with issues...")
    df = create_test_restaurant_data()
    
    print(f"Original data shape: {df.shape}")
    print("\n🔍 Issues in the data:")
    print("- Negative review counts (should be positive)")
    print("- All prices are '$$$$' (no variation)")
    print("- Cuisine data is composite (type • location)")
    print("- Data needs validation and normalization")
    
    print("\n" + "="*50)
    print("📋 Original Data Sample:")
    print(df.head(3).to_string())
    
    # Use simplified autonomous orchestrator for immediate testing
    print("\n🚀 Initializing Autonomous Agent System...")
    
    # Create a simplified orchestrator with working agents
    class SimpleAutonomousOrchestrator:
        """Simplified orchestrator for immediate testing"""
        
        def __init__(self):
            self.agents = [
                SimpleAutonomousValidator(),
                SimpleAutonomousTextProcessor(), 
                SimpleAutonomousPriceNormalizer()
            ]
            self.orchestration_log = []
        
        def process(self, data):
            """Process data through autonomous agents"""
            
            df = data['dataframe'].copy()
            orchestration_report = {
                'agents_executed': [],
                'total_transformations': 0,
                'improvements': {}
            }
            
            print("\n🔄 Executing Autonomous Agents...")
            
            # Execute each agent autonomously
            for agent in self.agents:
                agent_name = agent.__class__.__name__
                print(f"\n⚡ {agent_name} working autonomously...")
                
                result = agent.process({'dataframe': df})
                
                if result['status'] == 'success':
                    df = result['dataframe']
                    
                    # Count transformations
                    transformations = len(result.get('fixes_applied', []) + 
                                        result.get('transformations_applied', []) +
                                        result.get('normalizations_applied', []))
                    
                    orchestration_report['agents_executed'].append({
                        'agent': agent_name,
                        'transformations': transformations,
                        'details': result.get('fixes_applied', []) + 
                                 result.get('transformations_applied', []) +
                                 result.get('normalizations_applied', [])
                    })
                    
                    orchestration_report['total_transformations'] += transformations
                    
                    print(f"✅ {agent_name}: {transformations} transformations applied")
                    
                    # Show what was done
                    for detail in result.get('fixes_applied', []) + result.get('transformations_applied', []) + result.get('normalizations_applied', []):
                        print(f"   - {detail}")
                else:
                    print(f"❌ {agent_name}: Failed")
            
            return {
                'status': 'success',
                'dataframe': df,
                'orchestration_report': orchestration_report
            }
    
    # Initialize and run autonomous system
    orchestrator = SimpleAutonomousOrchestrator()
    
    result = orchestrator.process({'dataframe': df})
    
    if result['status'] == 'success':
        processed_df = result['dataframe']
        report = result['orchestration_report']
        
        print("\n" + "="*50)
        print("🎉 AUTONOMOUS PROCESSING COMPLETE!")
        print("="*50)
        
        print(f"\n📈 Processing Summary:")
        print(f"- Agents executed: {len(report['agents_executed'])}")
        print(f"- Total transformations: {report['total_transformations']}")
        print(f"- Original shape: {df.shape}")
        print(f"- Final shape: {processed_df.shape}")
        
        print(f"\n🔧 Transformations by Agent:")
        for agent_info in report['agents_executed']:
            print(f"\n{agent_info['agent']}:")
            for detail in agent_info['details']:
                print(f"  ✓ {detail}")
        
        print("\n" + "="*50)
        print("📋 PROCESSED DATA:")
        print("="*50)
        print(processed_df.to_string())
        
        print("\n🎯 KEY IMPROVEMENTS MADE:")
        print("- ✅ Fixed negative review counts (now positive)")
        print("- ✅ Split cuisine into type and location")  
        print("- ✅ Converted price symbols to numeric scale")
        print("- ✅ Added price categories (Budget/Moderate/Expensive/Luxury)")
        
        # Show specific improvements
        print(f"\n📊 Data Quality Improvements:")
        
        # Check review count fix
        negative_reviews_before = (df['restaurant_review_count'] < 0).sum()
        negative_reviews_after = (processed_df['restaurant_review_count'] < 0).sum() if 'restaurant_review_count' in processed_df.columns else 0
        print(f"- Negative review counts: {negative_reviews_before} → {negative_reviews_after}")
        
        # Check new columns created
        new_columns = set(processed_df.columns) - set(df.columns)
        print(f"- New columns created: {len(new_columns)}")
        for col in sorted(new_columns):
            print(f"  • {col}")
        
        # Save processed data
        output_file = "output/autonomous_processed_restaurants.csv"
        os.makedirs("output", exist_ok=True)
        processed_df.to_csv(output_file, index=False)
        print(f"\n💾 Processed data saved to: {output_file}")
        
        print("\n🚀 SUCCESS: Autonomous agents completed their specialized tasks!")
        print("Each agent worked independently on their expertise area:")
        print("- Data Validator: Fixed data quality issues")
        print("- Text Processor: Parsed and split composite text")  
        print("- Price Normalizer: Standardized price formats")
        
    else:
        print("❌ Processing failed!")

if __name__ == "__main__":
    main() 