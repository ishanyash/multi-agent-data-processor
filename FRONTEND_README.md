# 🤖 Multi-Agent Data Processor Frontend

A user-friendly web interface for the Multi-Agent Data Processing Pipeline. Upload datasets, process them with AI-powered cleaning, and download improved versions instantly!

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_frontend.txt
```

### 2. Launch the Frontend

```bash
python run_frontend.py
```

**OR** launch directly with Streamlit:

```bash
streamlit run app.py
```

### 3. Open Your Browser

The web interface will automatically open at: http://localhost:8501

## ✨ Features

### 📁 **Dataset Upload**
- **Drag & Drop**: Upload CSV files easily
- **File Validation**: Automatic format checking
- **Preview**: See your data before processing
- **File Info**: Shape, missing values, duplicates overview

### 🔄 **Processing Modes**
- **Minimal Mode**: Works without OpenAI API key
  - Data cleaning and basic processing
  - Fast and reliable
- **AI-Powered Mode**: Full intelligence with OpenAI
  - Advanced insights and recommendations
  - Context-aware cleaning strategies

### 📊 **Data Transformation Summary**
- **Before/After Metrics**: Row count, columns, missing values, duplicates
- **Visual Comparisons**: Interactive charts showing improvements
- **Processing Log**: Detailed actions performed on your data
- **Quality Score**: Data completeness percentage

### ⬇️ **Download Options**
- **Cleaned Dataset**: Download processed CSV file
- **Processing Report**: JSON report with full details
- **One-Click Download**: Instant file generation

### 📈 **Visual Analytics**
- **Missing Values Chart**: Before/after comparison
- **Data Quality Metrics**: Interactive dashboard
- **Progress Tracking**: Real-time processing status

## 🛠️ Usage Guide

### Step 1: Upload Your Dataset
1. Click "Choose a CSV file" or drag & drop
2. Review the data preview and statistics
3. Check dataset info (shape, missing values, etc.)

### Step 2: Choose Processing Mode
- **Minimal (No OpenAI)**: For quick testing without API key
- **Full AI-Powered**: For advanced processing (requires OpenAI API key)

### Step 3: Process Your Data
1. Click "🚀 Start Processing"
2. Watch the progress bar and status updates
3. Review the transformation results

### Step 4: Analyze Results
- Compare before/after data samples
- Review processing actions performed
- Check quality improvement metrics
- View missing values comparison chart

### Step 5: Download Results
- Download cleaned dataset as CSV
- Download processing report as JSON
- Save files with timestamped names

## 🔧 Configuration

### OpenAI API Setup (Optional)
For full AI-powered mode, configure your OpenAI API key:

1. Create `.env` file in project root:
```bash
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL_PRIMARY=gpt-4
OPENAI_MODEL_SECONDARY=gpt-3.5-turbo
```

2. Or run the setup script:
```bash
python create_env_file.py
```

### Directory Structure
The frontend automatically creates these directories:
- `uploads/` - Temporary uploaded files
- `output/` - Processed datasets and reports
- `data/` - Working data files

## 🎯 What the Tool Does

### 🧹 **Data Cleaning**
- Remove duplicate rows
- Handle missing values intelligently
- Fix data type inconsistencies
- Cap statistical outliers
- Standardize formats

### 🤖 **AI-Powered Analysis** (with OpenAI)
- Generate intelligent insights about your data
- Understand business context
- Provide quality recommendations
- Make automated cleaning decisions
- Suggest optimal processing strategies

### 📊 **Results & Reports**
- Comprehensive before/after comparison
- Quality improvement metrics
- Detailed processing logs
- Downloadable reports in multiple formats
- Visual analytics and charts

## 🚨 Troubleshooting

### Common Issues

1. **"Module not found" errors**
   ```bash
   pip install -r requirements_frontend.txt
   ```

2. **OpenAI authentication errors**
   - Check your API key in `.env` file
   - Try "Minimal (No OpenAI)" mode instead

3. **File upload issues**
   - Ensure CSV format
   - Check file size (keep under 100MB for best performance)
   - Verify file permissions

4. **Port already in use**
   ```bash
   streamlit run app.py --server.port 8502
   ```

### Error Messages
- **"Processing failed"**: Check your data format and try minimal mode
- **"API key not configured"**: Add OpenAI API key or use minimal mode
- **"File loading error"**: Verify CSV format and encoding

## 📝 Example Workflow

1. **Upload**: `sales_data.csv` (1000 rows, 20% missing values)
2. **Process**: Choose "Minimal (No OpenAI)" mode
3. **Results**: 
   - Missing values reduced from 200 to 0
   - 15 duplicate rows removed
   - Data completeness: 100%
4. **Download**: `cleaned_sales_data.csv` ready for analysis

## 🔄 Integration

The frontend integrates with your existing processing pipeline:
- Uses `SimpleDataProcessingPipeline` for AI mode
- Uses `MinimalDataProcessor` for basic mode
- Maintains all existing functionality
- Adds web interface layer

## 🎨 Customization

To customize the frontend:
1. Edit `app.py` for UI changes
2. Modify `run_frontend.py` for launch options
3. Update `requirements_frontend.txt` for dependencies

## 📞 Support

- Check processing logs for detailed error information
- Use minimal mode if OpenAI integration fails
- Ensure proper CSV formatting for uploads
- Monitor file sizes for optimal performance

---

🎉 **Ready to process your data?** Run `python run_frontend.py` to get started! 