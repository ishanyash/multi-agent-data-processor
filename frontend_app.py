#!/usr/bin/env python3
"""
Streamlit Frontend for Multi-Agent Data Processor
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import time
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Import the processing pipeline
from main_simple import SimpleDataProcessingPipeline
from minimal_test import MinimalDataProcessor

# Page configuration
st.set_page_config(
    page_title="Multi-Agent Data Processor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_directories():
    """Create necessary directories"""
    Path('uploads').mkdir(exist_ok=True)
    Path('output').mkdir(exist_ok=True)
    Path('data').mkdir(exist_ok=True)

def save_uploaded_file(uploaded_file):
    """Save uploaded file and return path"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{uploaded_file.name}"
    filepath = Path('uploads') / filename
    
    with open(filepath, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    return str(filepath)

def create_summary_metrics(original_df, cleaned_df, results):
    """Create summary metrics for display"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Rows", 
            f"{cleaned_df.shape[0]:,}", 
            f"{cleaned_df.shape[0] - original_df.shape[0]:+,}"
        )
    
    with col2:
        st.metric(
            "Columns", 
            cleaned_df.shape[1],
            f"{cleaned_df.shape[1] - original_df.shape[1]:+}"
        )
    
    with col3:
        original_missing = original_df.isnull().sum().sum()
        cleaned_missing = cleaned_df.isnull().sum().sum()
        st.metric(
            "Missing Values", 
            f"{cleaned_missing:,}",
            f"{cleaned_missing - original_missing:+,}"
        )
    
    with col4:
        if 'improvements' in results:
            completeness = results['improvements'].get('data_completeness', 0)
            st.metric("Data Completeness", f"{completeness:.1f}%")

def create_data_quality_chart(original_df, cleaned_df):
    """Create data quality comparison chart"""
    # Calculate missing values percentage
    original_missing = (original_df.isnull().sum() / len(original_df) * 100).to_dict()
    cleaned_missing = (cleaned_df.isnull().sum() / len(cleaned_df) * 100).to_dict()
    
    # Create comparison dataframe
    comparison_data = []
    for col in original_df.columns:
        comparison_data.append({
            'Column': col,
            'Original Missing %': original_missing.get(col, 0),
            'Cleaned Missing %': cleaned_missing.get(col, 0)
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Create bar chart
    fig = px.bar(
        df_comparison, 
        x='Column', 
        y=['Original Missing %', 'Cleaned Missing %'],
        title='Missing Values Comparison (Before vs After)',
        barmode='group'
    )
    fig.update_layout(height=400)
    
    return fig

def display_processing_log(results):
    """Display processing log in an expandable section"""
    with st.expander("📋 Processing Log", expanded=False):
        if 'cleaning' in results and 'cleaning_log' in results['cleaning']:
            for i, log_entry in enumerate(results['cleaning']['cleaning_log'], 1):
                st.write(f"{i}. {log_entry}")
        else:
            st.write("No detailed processing log available.")

def main():
    """Main Streamlit application"""
    
    # Initialize directories
    initialize_directories()
    
    # Header
    st.title("🤖 Multi-Agent Data Processor")
    st.markdown("Upload your dataset, process it with AI-powered cleaning, and download the improved version!")
    
    # Sidebar for settings
    st.sidebar.header("⚙️ Settings")
    
    # Processing mode selection
    processing_mode = st.sidebar.selectbox(
        "Processing Mode",
        ["Minimal (No OpenAI)", "Full AI-Powered"],
        help="Minimal mode works without OpenAI API key"
    )
    
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("📁 Upload Dataset")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload your dataset in CSV format"
        )
        
        if uploaded_file is not None:
            # Save uploaded file
            filepath = save_uploaded_file(uploaded_file)
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            
            # Load and preview data
            try:
                df = pd.read_csv(filepath)
                st.subheader("📊 Data Preview")
                st.dataframe(df.head(), use_container_width=True)
                
                # Basic info
                st.write(f"**Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns")
                st.write(f"**Missing Values:** {df.isnull().sum().sum():,}")
                st.write(f"**Duplicates:** {df.duplicated().sum():,}")
                
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
                return
    
    with col2:
        st.header("🔄 Processing")
        
        if uploaded_file is not None:
            # Process button
            if st.button("🚀 Process Dataset", type="primary"):
                
                # Processing progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Initialize processor based on mode
                    if processing_mode == "Minimal (No OpenAI)":
                        processor = MinimalDataProcessor()
                        status_text.text("🔄 Using minimal processor...")
                        progress_bar.progress(20)
                        
                        # Process dataset
                        job_id = f"frontend_{int(time.time())}"
                        results = processor.process_dataset(filepath, job_id)
                        progress_bar.progress(100)
                        
                    else:
                        pipeline = SimpleDataProcessingPipeline()
                        status_text.text("🔄 Using AI-powered processor...")
                        progress_bar.progress(20)
                        
                        # Process dataset
                        job_id = f"frontend_{int(time.time())}"
                        results = pipeline.process_dataset(filepath, job_id)
                        progress_bar.progress(100)
                    
                    status_text.text("✅ Processing completed!")
                    
                    # Store results in session state
                    st.session_state['results'] = results
                    st.session_state['original_file'] = filepath
                    st.session_state['processing_mode'] = processing_mode
                    
                except Exception as e:
                    st.error(f"❌ Processing failed: {str(e)}")
                    st.error("Please check your OpenAI API key configuration if using AI mode.")
                    return
    
    # Results section
    if 'results' in st.session_state:
        st.header("📈 Processing Results")
        
        results = st.session_state['results']
        original_df = pd.read_csv(st.session_state['original_file'])
        
        # Load cleaned dataset
        if processing_mode == "Minimal (No OpenAI)":
            cleaned_path = results['output_dataset']
        else:
            cleaned_path = results['output_dataset']
        
        try:
            cleaned_df = pd.read_csv(cleaned_path)
            
            # Summary metrics
            create_summary_metrics(original_df, cleaned_df, results)
            
            # Data quality chart
            st.subheader("📊 Data Quality Improvement")
            quality_chart = create_data_quality_chart(original_df, cleaned_df)
            st.plotly_chart(quality_chart, use_container_width=True)
            
            # Before/After comparison
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📋 Original Dataset")
                st.dataframe(original_df.head(), use_container_width=True)
                st.write(f"Shape: {original_df.shape}")
            
            with col2:
                st.subheader("✨ Cleaned Dataset")
                st.dataframe(cleaned_df.head(), use_container_width=True)
                st.write(f"Shape: {cleaned_df.shape}")
            
            # Processing details
            display_processing_log(results)
            
            # Download section
            st.header("⬇️ Download Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Download cleaned dataset
                csv_data = cleaned_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Cleaned Dataset",
                    data=csv_data,
                    file_name=f"cleaned_{uploaded_file.name}",
                    mime="text/csv",
                    type="primary"
                )
            
            with col2:
                # Download processing report
                report_data = json.dumps(results, indent=2, default=str)
                st.download_button(
                    label="📄 Download Processing Report",
                    data=report_data,
                    file_name=f"report_{uploaded_file.name.replace('.csv', '.json')}",
                    mime="application/json"
                )
            
            # Success message
            st.success("🎉 Processing completed successfully! You can download your cleaned dataset above.")
            
        except Exception as e:
            st.error(f"❌ Error loading results: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit and Multi-Agent Data Processing")

if __name__ == "__main__":
    main() 