#!/usr/bin/env python3
"""
Autonomous Multi-Agent Data Processor - Streamlit App
Showcases autonomous agents working on their specialties
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import time
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.autonomous_orchestrator import SimpleAutonomousValidator, SimpleAutonomousTextProcessor, SimpleAutonomousPriceNormalizer

# Page configuration
st.set_page_config(
    page_title="Autonomous Multi-Agent Data Processor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .agent-card {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #f8f9fa;
    }
    .success-metric {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 0.5rem;
        margin: 0.25rem 0;
    }
    .issue-highlight {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 0.5rem;
        margin: 0.25rem 0;
    }
</style>
""", unsafe_allow_html=True)

def create_sample_data():
    """Create sample data with various issues for testing"""
    
    # Restaurant data with issues
    restaurant_data = {
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
            'Milk Bun, Bristol'
        ],
        'restaurant_rating': [4.7, 4.7, 4.5, 4.5, 4.5, 4.7, 4.7, 4.7, 4.8, 4.7],
        'restaurant_review_count': [-439, -27, -120, -10214, -345, -22, -1488, -39, -507, -524],
        'restaurant_price': ['$$$$', '$$$$', '$$$$', '$$$$', '$$$$', '$$$$', '$$$$', '$$$$', '$$$$', '$$$$'],
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
            '• Burgers • Bridgeyate'
        ]
    }
    
    return pd.DataFrame(restaurant_data)

def analyze_data_issues(df):
    """Analyze and identify data issues"""
    
    issues = []
    
    # Check for negative values
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            negative_count = (df[col] < 0).sum()
            if negative_count > 0:
                issues.append(f"❌ {negative_count} negative values in '{col}'")
    
    # Check for composite text fields
    for col in df.columns:
        if df[col].dtype == 'object':
            sample_val = str(df[col].iloc[0])
            if '•' in sample_val:
                issues.append(f"📝 Composite text field '{col}' needs splitting")
    
    # Check for repetitive values
    for col in df.columns:
        if df[col].dtype == 'object':
            unique_count = df[col].nunique()
            if unique_count == 1:
                issues.append(f"🔄 All values in '{col}' are identical")
    
    return issues

def run_autonomous_agents(df):
    """Run autonomous agents and return results"""
    
    # Initialize agents
    agents = [
        SimpleAutonomousValidator(),
        SimpleAutonomousTextProcessor(),
        SimpleAutonomousPriceNormalizer()
    ]
    
    processing_results = []
    current_df = df.copy()
    
    # Process through each agent
    for agent in agents:
        agent_name = agent.__class__.__name__
        
        # Show progress
        with st.spinner(f"🤖 {agent_name} working autonomously..."):
            time.sleep(1)  # Simulate processing time
            
            result = agent.process({'dataframe': current_df})
            
            if result['status'] == 'success':
                current_df = result['dataframe']
                
                # Collect results
                transformations = (result.get('fixes_applied', []) + 
                                 result.get('transformations_applied', []) +
                                 result.get('normalizations_applied', []))
                
                processing_results.append({
                    'agent': agent_name,
                    'transformations': transformations,
                    'shape_before': result['original_shape'],
                    'shape_after': result['final_shape']
                })
    
    return current_df, processing_results

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<div class="main-header">🤖 Autonomous Multi-Agent Data Processor</div>', unsafe_allow_html=True)
    
    st.markdown("""
    This system demonstrates **autonomous AI agents** working independently on their specialties:
    - 🔍 **Data Validator**: Fixes data quality issues
    - 📝 **Text Processor**: Parses and normalizes text
    - 💰 **Price Normalizer**: Standardizes price formats
    """)
    
    # Sidebar
    st.sidebar.header("🎛️ Control Panel")
    
    # Data source selection
    data_source = st.sidebar.selectbox(
        "Select Data Source",
        ["Sample Restaurant Data", "Upload CSV File"]
    )
    
    df = None
    
    if data_source == "Sample Restaurant Data":
        df = create_sample_data()
        st.sidebar.success("✅ Sample data loaded")
        
    elif data_source == "Upload CSV File":
        uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success(f"✅ File uploaded: {uploaded_file.name}")
    
    if df is not None:
        # Main content
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.header("📊 Original Data")
            st.dataframe(df, use_container_width=True)
            
            # Data info
            st.subheader("📈 Data Information")
            st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
            st.write(f"**Columns:** {', '.join(df.columns)}")
            
            # Identify issues
            st.subheader("🔍 Issues Detected")
            issues = analyze_data_issues(df)
            
            if issues:
                for issue in issues:
                    st.markdown(f'<div class="issue-highlight">{issue}</div>', unsafe_allow_html=True)
            else:
                st.success("No obvious issues detected!")
        
        with col2:
            st.header("🤖 Autonomous Processing")
            
            if st.button("🚀 Run Autonomous Agents", type="primary"):
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Run agents
                status_text.text("Initializing autonomous agents...")
                progress_bar.progress(25)
                
                processed_df, results = run_autonomous_agents(df)
                
                progress_bar.progress(100)
                status_text.text("✅ Autonomous processing complete!")
                
                # Store results in session state
                st.session_state['processed_df'] = processed_df
                st.session_state['processing_results'] = results
                st.session_state['original_df'] = df
        
        # Show results if available
        if 'processed_df' in st.session_state:
            st.markdown("---")
            
            # Results summary
            st.header("🎉 Processing Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Agents Executed",
                    len(st.session_state['processing_results']),
                    delta=None
                )
            
            with col2:
                total_transformations = sum(len(r['transformations']) for r in st.session_state['processing_results'])
                st.metric(
                    "Total Transformations",
                    total_transformations,
                    delta=None
                )
            
            with col3:
                original_cols = st.session_state['original_df'].shape[1]
                processed_cols = st.session_state['processed_df'].shape[1]
                new_cols = processed_cols - original_cols
                st.metric(
                    "New Columns Created",
                    new_cols,
                    delta=f"+{new_cols}" if new_cols > 0 else None
                )
            
            # Agent details
            st.subheader("🔧 Agent Transformations")
            
            for result in st.session_state['processing_results']:
                with st.expander(f"🤖 {result['agent']} - {len(result['transformations'])} transformations"):
                    st.write(f"**Shape Change:** {result['shape_before']} → {result['shape_after']}")
                    
                    if result['transformations']:
                        st.write("**Transformations Applied:**")
                        for transformation in result['transformations']:
                            st.markdown(f'<div class="success-metric">✅ {transformation}</div>', unsafe_allow_html=True)
                    else:
                        st.info("No transformations needed")
            
            # Processed data
            st.subheader("📋 Processed Data")
            st.dataframe(st.session_state['processed_df'], use_container_width=True)
            
            # Before/After comparison
            st.subheader("📊 Before vs After Comparison")
            
            comparison_col1, comparison_col2 = st.columns(2)
            
            with comparison_col1:
                st.write("**Original Data Issues:**")
                original_issues = analyze_data_issues(st.session_state['original_df'])
                for issue in original_issues:
                    st.markdown(f'<div class="issue-highlight">{issue}</div>', unsafe_allow_html=True)
            
            with comparison_col2:
                st.write("**Processed Data Quality:**")
                processed_issues = analyze_data_issues(st.session_state['processed_df'])
                if processed_issues:
                    for issue in processed_issues:
                        st.markdown(f'<div class="issue-highlight">{issue}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="success-metric">✅ All major issues resolved!</div>', unsafe_allow_html=True)
            
            # Download processed data
            st.subheader("💾 Download Results")
            
            csv = st.session_state['processed_df'].to_csv(index=False)
            st.download_button(
                label="📥 Download Processed Data",
                data=csv,
                file_name=f"autonomous_processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    else:
        st.info("👆 Please select a data source from the sidebar to begin.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        🤖 Powered by Autonomous AI Agents | Each agent specializes in their domain expertise
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 