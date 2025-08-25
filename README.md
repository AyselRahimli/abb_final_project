# Bank360 - GenAI Customer Analytics & Insights Platform

🏦 **A comprehensive Streamlit-based banking analytics platform with AI-powered insights for Azerbaijan banks**

## 🌟 Features

### 📊 **Multi-Module Analytics Platform**
- **Home Dashboard**: Executive KPI overview with real-time metrics
- **Complaints & Feedback Analysis**: AI-powered sentiment analysis and response generation
- **Credit Risk & Expected Loss**: Advanced PD/LGD/EAD calculations with scenario analysis
- **Product Insights & Cross-Sell**: Customer segmentation and propensity modeling
- **Knowledge Search (RAG)**: Retrieval-Augmented Generation for institutional knowledge

### 🤖 **AI-Powered Features**
- **Google Gemini Integration**: Advanced natural language processing
- **Sentiment Analysis**: Automated complaint categorization and severity assessment
- **Topic Modeling**: LDA-based topic discovery in customer feedback
- **Response Generation**: Professional, context-aware complaint responses
- **Risk Assessment**: AI-enhanced credit scoring and decision explanations

### 🌍 **Bilingual Support**
- **Azerbaijani (Azərbaycan)**: Native language support for local banking
- **English**: International language support
- **Dynamic Translation**: Real-time language switching

### 📈 **Advanced Analytics**
- **Customer Segmentation**: Multi-dimensional customer profiling
- **Propensity Scoring**: Product recommendation algorithms
- **Scenario Analysis**: Stress testing for credit portfolios
- **Performance Metrics**: Comprehensive KPI tracking

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager
- (Optional) Google Gemini API key for enhanced AI features

### Installation

1. **Clone or download the application**
```bash
# If using git
git clone <repository-url>
cd bank360-analytics

# Or download the files directly
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run bank360_app.py
```

4. **Access the application**
   - Open your browser and navigate to `http://localhost:8501`
   - The application will start with sample data for demonstration

## 🔧 Configuration

### Google Gemini API Setup (Optional)
For enhanced AI features, configure your Gemini API key:

1. **Get API Key**
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key

2. **Install Gemini SDK**
```bash
pip install google-generativeai
```

3. **Configure in App**
   - Open the application
   - Navigate to the sidebar "Settings" section
   - Enter your API key
   - The app will use enhanced AI features automatically

### Environment Variables (Alternative Setup)
```bash
export GEMINI_API_KEY="your-api-key-here"
```

## 📁 Project Structure

```
bank360-analytics/
├── bank360_app.py          # Main application file
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── sample_data/           # (Auto-generated) Sample datasets
    ├── complaints.csv
    ├── loans.csv
    └── customers.csv
```

## 🏗️ Architecture

### Core Components

#### **1. GeminiAPI Class**
- Handles Google Gemini API integration
- Provides fallback mock responses for demo mode
- Supports bilingual prompt engineering

#### **2. ComplaintAnalyzer**
- Advanced sentiment analysis with severity scoring
- Topic modeling using Latent Dirichlet Allocation
- Similarity search for complaint matching
- Professional response generation

#### **3. CreditRiskAnalyzer**
- Multi-factor PD (Probability of Default) calculation
- LGD (Loss Given Default) and EAD (Exposure at Default) modeling
- Expected Loss computation with unexpected loss estimation
- Comprehensive scenario analysis

#### **4. ProductInsightAnalyzer**
- Advanced customer segmentation algorithms
- Cross-sell propensity scoring
- Value tier classification
- Marketing strategy recommendations

#### **5. KnowledgeBase (RAG)**
- TF-IDF based document retrieval
- Context-aware answer generation
- Dynamic knowledge base management
- Similarity-based document ranking

## 📊 Data Input Formats

### Complaint Data
```csv
id,date,customer_id,channel,category,text_az,severity,status,region
1,2024-01-01,1001,Mobil App,Kart,"Mobil tətbiqdə problem var",medium,Open,Bakı
```

### Loan Data
```csv
customer_id,age,income,employment,credit_score,loan_amount,debt_to_income,collateral_value,loan_to_value,tenure_months,region
1,35,2500,employed,680,25000,0.35,30000,0.75,24,Bakı
```

### Customer Data
```csv
customer_id,age,income,tenure_months,num_products,region,last_transaction_days,digital_adoption
1,42,3200,36,3,Bakı,5,High
```

## 🔍 Usage Guide

### **1. Home Dashboard**
- View executive KPIs and trends
- Monitor complaint volumes and satisfaction scores
- Track risk metrics and segment performance
- Generate AI-powered business insights

### **2. Complaints Analysis**
- Upload complaint data (CSV/Excel/JSON)
- Perform sentiment analysis and categorization
- Discover topics using machine learning
- Generate professional responses to complaints
- Find similar historical cases

### **3. Credit Risk Assessment**
- Input customer and loan parameters
- Calculate advanced risk metrics (PD/LGD/EAD)
- Visualize risk factor contributions
- Perform stress testing scenarios
- Get AI explanations for decisions

### **4. Product Insights**
- Analyze customer segments and value tiers
- Calculate cross-sell propensities
- View product performance metrics
- Generate marketing strategies

### **5. Knowledge Search**
- Add institutional documents to knowledge base
- Search using natural language queries
- Get contextual answers with source references
- Manage document library

## 🛠️ Customization

### Adding New Languages
Extend the `TRANSLATIONS` dictionary in `bank360_app.py`:
```python
TRANSLATIONS['tr'] = {
    'title': 'Bank360 - Türkçe Başlık',
    # ... other translations
}
```

### Custom Risk Models
Modify the `calculate_pd_advanced` method in `CreditRiskAnalyzer`:
```python
def calculate_pd_advanced(self, features):
    # Implement your custom scoring logic
    return {'pd': calculated_pd, 'components': components}
```

### New Data Sources
Extend file upload functionality:
```python
# Add support for new formats
if uploaded_file.type == 'application/your-format':
    data = your_custom_loader(uploaded_file)
```

## 🎯 Key Metrics & KPIs

### Complaint Analytics
- **CSAT Index**: Customer satisfaction scoring
- **Severity Distribution**: High/Medium/Low priority tracking
- **Resolution Time**: Average case closure metrics
- **Channel Performance**: Multi-channel analysis

### Risk Management
- **Portfolio PD**: Average probability of default
- **Expected Loss**: Financial risk exposure
- **Concentration Risk**: Sector/region analysis
- **Stress Test Results**: Scenario-based projections

### Product Performance
- **Cross-sell Success**: Propensity model accuracy
- **Segment Growth**: Customer lifecycle tracking
- **Revenue Attribution**: Product contribution analysis
- **Digital Adoption**: Channel preference metrics

## 🔐 Security & Compliance

### Data Privacy
- No persistent data storage by default
- Session-based data handling
- API key encryption in transit
- Configurable data retention policies

### Banking Compliance
- Risk calculation methodology aligned with Basel III
- Audit trail capabilities
- Data lineage tracking
- Regulatory reporting readiness

## 🚀 Deployment Options

### Local Development
```bash
streamlit run bank360_app.py
```

### Production Deployment
```bash
# Using Docker
docker build -t bank360 .
docker run -p 8501:8501 bank360

# Using cloud platforms
# - Streamlit Cloud
# - Heroku
# - AWS/Azure/GCP
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📋 Roadmap

- [ ] **Advanced ML Models**: Deep learning for credit scoring
- [ ] **Real-time Data Integration**: Live API connections
- [ ] **Mobile App**: Native mobile interface
- [ ] **Advanced Visualization**: 3D analytics and VR dashboards
- [ ] **Regulatory Modules**: Automated compliance reporting
- [ ] **Multi-language NLP**: Extended language support

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For support and questions:
- 📧 Email: support@bank360analytics.com
- 💬 Issues: GitHub Issues page
- 📚 Documentation: [Full Documentation](./docs/)

## 🙏 Acknowledgments

- **Streamlit Team**: For the amazing framework
- **Google AI**: For Gemini API capabilities
- **Plotly**: For interactive visualizations
- **Scikit-learn**: For machine learning algorithms
- **Azerbaijan Banking Community**: For domain expertise

---

**Made with ❤️ for Azerbaijan Banking Sector**

*Bank360 - Empowering financial institutions with AI-driven insights*