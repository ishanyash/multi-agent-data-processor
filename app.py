#!/usr/bin/env python3
"""
Enhanced Multi-Agent Data Processor Web Interface
Supports multiple file formats, interactive decision-making, and intelligent processing
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime
import os
import sys
from pathlib import Path
import io
import zipfile

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import our modules
from utils.file_handler import EnhancedFileHandler
from agents.interactive_agent import InteractiveDataAgent
from agents.data_profiler_simple import DataProfilerAgent
from agents.data_cleaning_simple import DataCleaningAgent

# Page configuration
st.set_page_config(
    page_title="🤖 Multi-Agent Data Processor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .decision-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
    }
    
    .success-card {
        background: #d4edda;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    
    .warning-card {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    
    .error-card {
        background: #f8d7da;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #dc3545;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'uploaded_data' not in st.session_state:
        st.session_state.uploaded_data = None
    if 'file_metadata' not in st.session_state:
        st.session_state.file_metadata = None
    if 'data_issues' not in st.session_state:
        st.session_state.data_issues = None
    if 'decision_interface' not in st.session_state:
        st.session_state.decision_interface = None
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'processing_log' not in st.session_state:
        st.session_state.processing_log = []
    if 'user_decisions' not in st.session_state:
        st.session_state.user_decisions = []

def create_download_package(df: pd.DataFrame, metadata: dict, processing_log: list) -> bytes:
    """Create a ZIP package with multiple file formats"""
    
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        
        # Add CSV file
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        zip_file.writestr('processed_data.csv', csv_buffer.getvalue())
        
        # Add JSON file
        json_buffer = io.StringIO()
        df.to_json(json_buffer, orient='records', indent=2)
        zip_file.writestr('processed_data.json', json_buffer.getvalue())
        
        # Add Excel file
        excel_buffer = io.BytesIO()
        df.to_excel(excel_buffer, index=False, engine='openpyxl')
        zip_file.writestr('processed_data.xlsx', excel_buffer.getvalue())
        
        # Add processing report
        report = {
            'processing_timestamp': datetime.now().isoformat(),
            'original_metadata': metadata,
            'final_shape': df.shape,
            'final_columns': df.columns.tolist(),
            'processing_log': processing_log,
            'summary_statistics': df.describe().to_dict() if not df.empty else {}
        }
        
        zip_file.writestr('processing_report.json', json.dumps(report, indent=2))
        
        # Add data quality report
        quality_report = generate_quality_report(df)
        zip_file.writestr('data_quality_report.txt', quality_report)
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

def generate_quality_report(df: pd.DataFrame) -> str:
    """Generate a comprehensive data quality report"""
    
    report_lines = [
        "="*60,
        "DATA QUALITY REPORT",
        "="*60,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "DATASET OVERVIEW:",
        f"  • Shape: {df.shape[0]:,} rows × {df.shape[1]} columns",
        f"  • Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
        "",
        "COLUMN INFORMATION:",
    ]
    
    for col in df.columns:
        dtype = str(df[col].dtype)
        null_count = df[col].isnull().sum()
        null_pct = (null_count / len(df)) * 100
        unique_count = df[col].nunique()
        
        report_lines.append(f"  • {col}:")
        report_lines.append(f"    - Type: {dtype}")
        report_lines.append(f"    - Missing: {null_count} ({null_pct:.1f}%)")
        report_lines.append(f"    - Unique values: {unique_count}")
        
        if df[col].dtype in ['int64', 'float64']:
            report_lines.append(f"    - Range: {df[col].min():.2f} to {df[col].max():.2f}")
        
        report_lines.append("")
    
    # Data quality metrics
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    completeness = ((total_cells - missing_cells) / total_cells) * 100
    
    report_lines.extend([
        "DATA QUALITY METRICS:",
        f"  • Completeness: {completeness:.1f}%",
        f"  • Duplicate rows: {df.duplicated().sum()}",
        f"  • Total missing cells: {missing_cells:,}",
        "",
        "="*60
    ])
    
    return "\n".join(report_lines)

def display_data_issues(issues: dict):
    """Display data quality issues in an organized manner"""
    
    st.subheader("🔍 Data Quality Analysis")
    
    # AI Insights
    if 'ai_insights' in issues:
        ai_insights = issues['ai_insights']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #667eea; margin: 0;">Quality Score</h3>
                <h2 style="margin: 0;">{ai_insights.get('quality_score', 0)}/100</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            impact_color = {'high': '#dc3545', 'medium': '#ffc107', 'low': '#28a745'}
            impact = ai_insights.get('business_impact', 'medium')
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #667eea; margin: 0;">Business Impact</h3>
                <h2 style="margin: 0; color: {impact_color.get(impact, '#666')};">{impact.upper()}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #667eea; margin: 0;">Priority Issues</h3>
                <p style="margin: 0;">{len(ai_insights.get('priority_issues', []))}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #667eea; margin: 0;">Strategy</h3>
                <p style="margin: 0; font-size: 0.9em;">{ai_insights.get('recommended_strategy', 'Standard cleaning')[:30]}...</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Detailed Issues
    col1, col2 = st.columns(2)
    
    with col1:
        # Missing Values
        if issues['missing_values']:
            st.markdown("#### 🕳️ Missing Values")
            for col, info in issues['missing_values'].items():
                severity_color = {
                    'low': '#28a745', 
                    'medium': '#ffc107', 
                    'high': '#fd7e14', 
                    'critical': '#dc3545'
                }
                color = severity_color.get(info['severity'], '#666')
                
                st.markdown(f"""
                <div style="background: #f8f9fa; padding: 1rem; border-radius: 5px; margin: 0.5rem 0; border-left: 4px solid {color};">
                    <strong>{col}</strong><br>
                    Missing: {info['count']} ({info['percentage']:.1f}%)<br>
                    <span style="color: {color};">Severity: {info['severity'].upper()}</span>
                </div>
                """, unsafe_allow_html=True)
        
        # Data Type Issues
        if issues['data_types']:
            st.markdown("#### 🔄 Data Type Issues")
            for col, info in issues['data_types'].items():
                st.markdown(f"""
                <div style="background: #e3f2fd; padding: 1rem; border-radius: 5px; margin: 0.5rem 0;">
                    <strong>{col}</strong><br>
                    Current: {info['current_type']}<br>
                    Suggested: {info['suggested_type']}<br>
                    Confidence: {info['confidence']:.1%}
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        # Duplicates
        if issues['duplicates']['count'] > 0:
            st.markdown("#### 👥 Duplicate Rows")
            dup_info = issues['duplicates']
            st.markdown(f"""
            <div style="background: #fff3e0; padding: 1rem; border-radius: 5px; margin: 0.5rem 0;">
                <strong>Duplicates Found</strong><br>
                Count: {dup_info['count']}<br>
                Percentage: {dup_info['percentage']:.1f}%
            </div>
            """, unsafe_allow_html=True)
        
        # Outliers
        if issues['outliers']:
            st.markdown("#### 📊 Outliers")
            for col, info in issues['outliers'].items():
                st.markdown(f"""
                <div style="background: #fce4ec; padding: 1rem; border-radius: 5px; margin: 0.5rem 0;">
                    <strong>{col}</strong><br>
                    Outliers: {info['count']} ({info['percentage']:.1f}%)<br>
                    Range: [{info['bounds']['lower']:.2f}, {info['bounds']['upper']:.2f}]
                </div>
                """, unsafe_allow_html=True)

def display_decision_interface(decision_interface: dict):
    """Display interactive decision interface"""
    
    st.subheader("🤖 AI-Powered Decision Making")
    
    # Summary
    summary = decision_interface['summary']
    st.info(f"**Decision Summary:** {summary['total_decisions']} decisions needed "
           f"({summary['user_input_required']} require your input, "
           f"{summary['auto_decisions']} can be automated)")
    
    user_decisions = []
    
    for i, decision_point in enumerate(decision_interface['decision_points']):
        
        st.markdown(f"""
        <div class="decision-card">
            <h4>{decision_point['title']}</h4>
            <p>{decision_point['description']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display options
        options = decision_point['options']
        option_labels = []
        option_details = []
        
        for opt in options:
            label = f"{opt['method'].replace('_', ' ').title()}"
            detail = f"{opt['description']}"
            
            # Add suitability indicator
            suitability_emoji = {'good': '✅', 'fair': '⚠️', 'poor': '❌'}
            emoji = suitability_emoji.get(opt.get('suitability', 'fair'), '⚠️')
            
            option_labels.append(f"{emoji} {label}")
            option_details.append(f"{detail}\n• Pros: {', '.join(opt['pros'])}\n• Cons: {', '.join(opt['cons'])}")
        
        # User selection
        selected_idx = st.radio(
            f"Choose action for: **{decision_point.get('column', 'dataset')}**",
            range(len(options)),
            format_func=lambda x: option_labels[x],
            key=f"decision_{i}",
            help="Select the best option based on your data requirements"
        )
        
        # Show details of selected option
        with st.expander("📋 Option Details"):
            st.text(option_details[selected_idx])
        
        # Store decision
        user_decisions.append({
            'id': decision_point.get('id', f'decision_{i}'),
            'type': decision_point['type'],
            'column': decision_point.get('column'),
            'method': options[selected_idx]['method'],
            'description': options[selected_idx]['description']
        })
        
        st.markdown("---")
    
    st.session_state.user_decisions = user_decisions
    
    return user_decisions

def process_data_with_decisions(df: pd.DataFrame, decisions: list):
    """Process data using user decisions"""
    
    # Initialize agents
    interactive_agent = InteractiveDataAgent()
    
    # Apply decisions
    processed_df, processing_log = interactive_agent.apply_decisions(df, decisions)
    
    return processed_df, processing_log

def display_comparison_charts(original_df: pd.DataFrame, processed_df: pd.DataFrame):
    """Display before/after comparison charts"""
    
    st.subheader("📊 Before vs After Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Original Dataset")
        
        # Basic stats
        st.write(f"**Shape:** {original_df.shape[0]:,} rows × {original_df.shape[1]} columns")
        st.write(f"**Missing Values:** {original_df.isnull().sum().sum():,}")
        st.write(f"**Duplicates:** {original_df.duplicated().sum():,}")
        
        # Missing values chart
        missing_data = original_df.isnull().sum()
        missing_data = missing_data[missing_data > 0]
        
        if not missing_data.empty:
            fig = px.bar(
                x=missing_data.values,
                y=missing_data.index,
                orientation='h',
                title="Missing Values by Column",
                labels={'x': 'Missing Count', 'y': 'Column'}
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Processed Dataset")
        
        # Basic stats
        st.write(f"**Shape:** {processed_df.shape[0]:,} rows × {processed_df.shape[1]} columns")
        st.write(f"**Missing Values:** {processed_df.isnull().sum().sum():,}")
        st.write(f"**Duplicates:** {processed_df.duplicated().sum():,}")
        
        # Missing values chart
        missing_data_processed = processed_df.isnull().sum()
        missing_data_processed = missing_data_processed[missing_data_processed > 0]
        
        if not missing_data_processed.empty:
            fig = px.bar(
                x=missing_data_processed.values,
                y=missing_data_processed.index,
                orientation='h',
                title="Missing Values by Column",
                labels={'x': 'Missing Count', 'y': 'Column'}
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No missing values remaining!")
    
    # Improvement metrics
    st.markdown("#### 🎯 Improvement Metrics")
    
    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
    
    with metrics_col1:
        rows_change = processed_df.shape[0] - original_df.shape[0]
        st.metric("Rows", f"{processed_df.shape[0]:,}", f"{rows_change:+,}")
    
    with metrics_col2:
        missing_original = original_df.isnull().sum().sum()
        missing_processed = processed_df.isnull().sum().sum()
        missing_change = missing_processed - missing_original
        st.metric("Missing Values", f"{missing_processed:,}", f"{missing_change:+,}")
    
    with metrics_col3:
        dup_original = original_df.duplicated().sum()
        dup_processed = processed_df.duplicated().sum()
        dup_change = dup_processed - dup_original
        st.metric("Duplicates", f"{dup_processed:,}", f"{dup_change:+,}")
    
    with metrics_col4:
        completeness_original = ((original_df.shape[0] * original_df.shape[1] - missing_original) / 
                                (original_df.shape[0] * original_df.shape[1])) * 100
        completeness_processed = ((processed_df.shape[0] * processed_df.shape[1] - missing_processed) / 
                                 (processed_df.shape[0] * processed_df.shape[1])) * 100
        completeness_change = completeness_processed - completeness_original
        st.metric("Completeness", f"{completeness_processed:.1f}%", f"{completeness_change:+.1f}%")

def main():
    """Main application function"""
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Multi-Agent Data Processor</h1>
        <p>Intelligent data processing with AI-powered decision making</p>
        <p>Supports CSV, JSON, Excel, Parquet, and more formats</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🔧 Processing Options")
        
        processing_mode = st.selectbox(
            "Processing Mode",
            ["🤖 AI-Powered Interactive", "⚡ Quick Clean", "📊 Analysis Only"],
            help="Choose your processing approach"
        )
        
        st.markdown("### 📁 Supported Formats")
        file_handler = EnhancedFileHandler()
        formats = file_handler.get_supported_extensions()
        for fmt in formats:
            description = file_handler.get_format_description(fmt)
            st.write(f"• **{fmt}** - {description}")
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload & Analyze", "🤖 AI Decisions", "📊 Results", "💾 Download"])
    
    with tab1:
        st.subheader("📤 Upload Your Dataset")
        
        uploaded_file = st.file_uploader(
            "Choose your data file",
            type=['csv', 'json', 'xlsx', 'xls', 'tsv', 'parquet', 'pkl', 'feather'],
            help="Upload any supported data format for processing"
        )
        
        if uploaded_file is not None:
            try:
                # Save uploaded file temporarily
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Load file using enhanced handler
                file_handler = EnhancedFileHandler()
                df, metadata = file_handler.load_file(temp_path)
                
                # Clean up temp file
                os.remove(temp_path)
                
                # Store in session state
                st.session_state.uploaded_data = df
                st.session_state.file_metadata = metadata
                
                # Display file info
                st.success(f"✅ Successfully loaded {metadata['original_format']} file!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows", f"{df.shape[0]:,}")
                with col2:
                    st.metric("Columns", df.shape[1])
                with col3:
                    st.metric("Size", f"{metadata['file_size_mb']:.2f} MB")
                
                # Show conversion log
                if metadata['conversion_log']:
                    with st.expander("📋 File Processing Log"):
                        for log_entry in metadata['conversion_log']:
                            st.write(f"• {log_entry}")
                
                # Display data preview
                st.subheader("👀 Data Preview")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Analyze data issues
                if processing_mode == "🤖 AI-Powered Interactive":
                    with st.spinner("🤖 AI is analyzing your data quality..."):
                        interactive_agent = InteractiveDataAgent()
                        issues = interactive_agent.analyze_data_issues(df)
                        decision_interface = interactive_agent.generate_decision_interface(issues)
                        
                        st.session_state.data_issues = issues
                        st.session_state.decision_interface = decision_interface
                    
                    # Display issues
                    display_data_issues(issues)
                
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
                st.info("Please check that your file format is supported and not corrupted.")
    
    with tab2:
        if st.session_state.decision_interface is not None:
            st.subheader("🤖 AI-Powered Decision Making")
            
            # Display AI insights
            ai_insights = st.session_state.data_issues.get('ai_insights', {})
            
            st.markdown(f"""
            <div class="success-card">
                <h4>🎯 AI Recommendation</h4>
                <p><strong>Strategy:</strong> {ai_insights.get('recommended_strategy', 'Standard processing')}</p>
                <p><strong>Priority Issues:</strong> {', '.join(ai_insights.get('priority_issues', []))}</p>
                <p><strong>Business Impact:</strong> {ai_insights.get('business_impact', 'Medium').upper()}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Decision interface
            user_decisions = display_decision_interface(st.session_state.decision_interface)
            
            # Process button
            if st.button("🚀 Apply AI Decisions", type="primary", use_container_width=True):
                if st.session_state.uploaded_data is not None:
                    with st.spinner("🤖 Processing your data with AI decisions..."):
                        processed_df, processing_log = process_data_with_decisions(
                            st.session_state.uploaded_data, 
                            user_decisions
                        )
                        
                        st.session_state.processed_data = processed_df
                        st.session_state.processing_log = processing_log
                    
                    st.success("✅ Data processing completed!")
                    st.balloons()
        else:
            st.info("📤 Please upload a file first to see AI decision options.")
    
    with tab3:
        if st.session_state.processed_data is not None:
            st.subheader("📊 Processing Results")
            
            # Display processing log
            with st.expander("📋 Processing Log"):
                for log_entry in st.session_state.processing_log:
                    st.write(f"• {log_entry}")
            
            # Comparison charts
            display_comparison_charts(
                st.session_state.uploaded_data,
                st.session_state.processed_data
            )
            
            # Final data preview
            st.subheader("🎯 Final Processed Data")
            st.dataframe(st.session_state.processed_data.head(20), use_container_width=True)
            
            # Data quality summary
            final_df = st.session_state.processed_data
            
            st.subheader("📈 Final Data Quality")
            quality_col1, quality_col2, quality_col3, quality_col4 = st.columns(4)
            
            with quality_col1:
                st.metric("Total Rows", f"{final_df.shape[0]:,}")
            with quality_col2:
                st.metric("Total Columns", final_df.shape[1])
            with quality_col3:
                missing_pct = (final_df.isnull().sum().sum() / (final_df.shape[0] * final_df.shape[1])) * 100
                st.metric("Missing Data", f"{missing_pct:.1f}%")
            with quality_col4:
                completeness = 100 - missing_pct
                st.metric("Completeness", f"{completeness:.1f}%")
        else:
            st.info("🤖 Process your data first to see results.")
    
    with tab4:
        if st.session_state.processed_data is not None:
            st.subheader("💾 Download Options")
            
            processed_df = st.session_state.processed_data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Individual format downloads
            st.markdown("#### 📁 Individual Files")
            
            download_col1, download_col2, download_col3 = st.columns(3)
            
            with download_col1:
                # CSV Download
                csv_data = processed_df.to_csv(index=False)
                st.download_button(
                    label="📄 Download CSV",
                    data=csv_data,
                    file_name=f"processed_data_{timestamp}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with download_col2:
                # JSON Download
                json_data = processed_df.to_json(orient='records', indent=2)
                st.download_button(
                    label="📋 Download JSON",
                    data=json_data,
                    file_name=f"processed_data_{timestamp}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            with download_col3:
                # Excel Download
                excel_buffer = io.BytesIO()
                processed_df.to_excel(excel_buffer, index=False, engine='openpyxl')
                excel_data = excel_buffer.getvalue()
                
                st.download_button(
                    label="📊 Download Excel",
                    data=excel_data,
                    file_name=f"processed_data_{timestamp}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            
            # Complete package download
            st.markdown("#### 📦 Complete Package")
            
            zip_data = create_download_package(
                processed_df, 
                st.session_state.file_metadata, 
                st.session_state.processing_log
            )
            
            st.download_button(
                label="🎁 Download Complete Package (ZIP)",
                data=zip_data,
                file_name=f"data_processing_package_{timestamp}.zip",
                mime="application/zip",
                use_container_width=True,
                help="Includes CSV, JSON, Excel files plus processing report and quality analysis"
            )
            
            # Package contents info
            with st.expander("📋 Package Contents"):
                st.write("""
                **Complete Package includes:**
                • `processed_data.csv` - Clean data in CSV format
                • `processed_data.json` - Clean data in JSON format  
                • `processed_data.xlsx` - Clean data in Excel format
                • `processing_report.json` - Detailed processing report
                • `data_quality_report.txt` - Human-readable quality analysis
                """)
        else:
            st.info("🤖 Process your data first to access download options.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🤖 <strong>Multi-Agent Data Processor</strong> - Powered by OpenAI GPT-4</p>
        <p>Intelligent data processing with AI consciousness for better decision making</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 