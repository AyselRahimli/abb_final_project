# Bank360 - Critical Fixes and Improvements
# This file contains the essential fixes for the main issues

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import io
import base64
from typing import List, Dict, Any, Optional
import re
import warnings
warnings.filterwarnings('ignore')

# Safe imports with fallbacks
try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.cluster import KMeans
    from sklearn.metrics import roc_curve, auc, precision_recall_curve
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    st.error("scikit-learn not installed. Run: pip install scikit-learn")

try:
    import scipy.stats as stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    st.error("scipy not installed. Run: pip install scipy")

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    st.warning("Google Gemini API not available. Install with: pip install google-generativeai")

# Configure Streamlit page
st.set_page_config(
    page_title="Bank360 Analytics",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    """Initialize session state with proper defaults"""
    defaults = {
        'language': 'az',
        'complaint_data': None,
        'loan_data': None,
        'customer_data': None,
        'gemini_api_key': "",
        'knowledge_base': None,
        'initialized': False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def safe_execute(func, *args, **kwargs):
    """Safely execute functions with error handling"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        st.error(f"Error in {func.__name__}: {str(e)}")
        return None

@st.cache_data
def generate_sample_data_fixed():
    """Generate sample data with proper error handling and consistency"""
    np.random.seed(42)  # For reproducibility
    
    # Fixed complaint texts - ensure exactly 100 entries
    base_complaint_texts = [
        "Mobil tətbiqdə problem var, giriş edə bilmirəm",
        "ATM-dən pul çıxarmaq mümkün olmur", 
        "Kart komissiyası çox yüksəkdir",
        "Filial xidməti çox yavaşdır",
        "Kredit məbləği kifayət etmir",
        "İnternet banking işləmir",
        "Hesabımdan səhv məbləğ silinib",
        "Telefon zəngləri çox tez-tez gəlir",
        "Online ödəniş sistemi yavaş işləyir",
        "Kart bloklanıb, səbəbi aydın deyil"
    ]
    
    # Generate exactly 100 complaint texts
    text_az = [np.random.choice(base_complaint_texts) for _ in range(100)]
    
    # Complaint data with consistent types
    complaint_data = {
        'id': list(range(1, 101)),
        'date': pd.date_range(start='2024-01-01', periods=100, freq='D'),
        'customer_id': np.random.randint(1000, 9999, 100),
        'channel': np.random.choice(['Mobil App', 'Filial', 'Call Center', 'Website'], 100),
        'category': np.random.choice(['Kart', 'ATM', 'Mobil', 'Komissiya', 'Filial', 'Kredit'], 100),
        'text_az': text_az,
        'severity': np.random.choice(['low', 'medium', 'high'], 100, p=[0.4, 0.4, 0.2]),
        'status': np.random.choice(['Open', 'In Progress', 'Closed'], 100, p=[0.2, 0.3, 0.5]),
        'region': np.random.choice(['Bakı', 'Gəncə', 'Sumqayıt', 'Mingəçevir', 'Şəki'], 100)
    }
    
    # Loan data with proper data types
    loan_data = {
        'customer_id': list(range(1, 201)),
        'age': np.clip(np.random.normal(40, 12, 200).astype(int), 18, 80),
        'income': np.clip(np.random.gamma(2, 1000, 200), 300, 15000),
        'employment': np.random.choice(['government', 'employed', 'self_employed', 'unemployed'], 200, p=[0.2, 0.5, 0.2, 0.1]),
        'credit_score': np.clip(np.random.normal(650, 100, 200).astype(int), 300, 850),
        'loan_amount': np.clip(np.random.gamma(2, 5000, 200), 1000, 100000),
        'debt_to_income': np.clip(np.random.beta(2, 3, 200), 0.05, 0.95),
        'collateral_value': np.random.gamma(1.5, 8000, 200),
        'loan_to_value': np.clip(np.random.beta(3, 2, 200), 0.1, 0.95),
        'tenure_months': np.random.randint(6, 120, 200),
        'region': np.random.choice(['Bakı', 'Gəncə', 'Sumqayıt', 'Mingəçevir', 'Şəki'], 200)
    }
    
    # Customer data
    customer_data = {
        'customer_id': list(range(1, 301)),
        'age': np.clip(np.random.normal(38, 15, 300).astype(int), 18, 80),
        'income': np.clip(np.random.gamma(2, 1200, 300), 300, 10000),
        'tenure_months': np.random.randint(1, 60, 300),
        'num_products': np.clip(np.random.poisson(2, 300) + 1, 1, 6),
        'region': np.random.choice(['Bakı', 'Gəncə', 'Sumqayıt', 'Mingəçevir', 'Şəki'], 300),
        'last_transaction_days': np.random.randint(1, 90, 300),
        'digital_adoption': np.random.choice(['High', 'Medium', 'Low'], 300, p=[0.3, 0.5, 0.2])
    }
    
    return (
        pd.DataFrame(complaint_data),
        pd.DataFrame(loan_data).head(100),
        pd.DataFrame(customer_data).head(100)
    )

class ImprovedGeminiAPI:
    """Improved Gemini API wrapper with better error handling"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.model = None
        self.initialized = False
        
        if api_key and GEMINI_AVAILABLE:
            try:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel('gemini-2.5-flash')
                self.initialized = True
                st.success("Gemini API initialized successfully!")
            except Exception as e:
                st.error(f"Gemini API initialization error: {str(e)}")
                self.initialized = False
        elif not GEMINI_AVAILABLE:
            st.info("Gemini API not available - using mock responses")
    
    def generate_response(self, prompt: str, language: str = 'az', max_retries: int = 3) -> str:
        """Generate response with retry logic and proper error handling"""
        if not self.initialized or not self.model:
            return self._mock_response(prompt, language)
        
        for attempt in range(max_retries):
            try:
                lang_instruction = "Cavabı Azərbaycan dilində verin" if language == 'az' else "Provide response in English"
                full_prompt = f"{lang_instruction}. {prompt}"
                
                response = self.model.generate_content(full_prompt)
                
                if response.text:
                    return response.text
                else:
                    raise Exception("Empty response from API")
                    
            except Exception as e:
                if attempt == max_retries - 1:
                    st.warning(f"API call failed after {max_retries} attempts: {str(e)}")
                    return self._mock_response(prompt, language)
                continue
        
        return self._mock_response(prompt, language)
    
    def _mock_response(self, prompt: str, language: str = 'az') -> str:
        """Enhanced mock response system"""
        prompt_lower = prompt.lower()
        
        # Complaint responses
        if any(word in prompt_lower for word in ['complaint', 'şikayət', 'problem']):
            if language == 'az':
                return "Hörmətli müştəri, şikayətinizi qəbul edirik və dərhal araşdırmaya başlayırıq. 2-3 iş günü ərzində sizinlə əlaqə saxlayacağıq. Səbiriniz üçün təşəkkür edirik."
            else:
                return "Dear customer, we acknowledge your complaint and will immediately investigate. We will contact you within 2-3 business days. Thank you for your patience."
        
        # Credit analysis responses
        elif any(word in prompt_lower for word in ['credit', 'kredit', 'loan', 'risk']):
            if language == 'az':
                return "Kredit analizi nəticəsində: müştərinin ödəmə qabiliyyəti orta səviyyədə qiymətləndirilir. Əlavə sənədlər və ya təminat tələb oluna bilər. Risk idarəetməsi departamenti ilə əlavə məsləhətləşmə tövsiyə olunur."
            else:
                return "Credit analysis results: customer's payment ability is assessed at medium level. Additional documents or collateral may be required. Consultation with risk management department is recommended."
        
        # Strategy responses
        elif any(word in prompt_lower for word in ['strategy', 'strategiya', 'recommend', 'tövsiyə']):
            if language == 'az':
                return "Marketinq strategiyası tövsiyələri: 1) Rəqəmsal platformaları inkişaf etdirin, 2) Müştəri seqmentlərinə uyğun məhsullar təklif edin, 3) Müştəri məmnuniyyətini artırmaq üçün xidmət keyfiyyətini yaxşılaşdırın, 4) Çarpaz satış imkanlarından istifadə edin."
            else:
                return "Marketing strategy recommendations: 1) Develop digital platforms, 2) Offer products tailored to customer segments, 3) Improve service quality to increase customer satisfaction, 4) Leverage cross-selling opportunities."
        
        # General response
        else:
            if language == 'az':
                return "Sorğunuz əsasında analiz aparılmış və müvafiq tövsiyələr hazırlanmışdır. Əlavə məlumat üçün müvafiq departamentlə əlaqə saxlayın."
            else:
                return "Analysis has been conducted based on your query and appropriate recommendations have been prepared. Contact the relevant department for additional information."

def validate_uploaded_file(uploaded_file) -> Optional[pd.DataFrame]:
    """Validate and process uploaded files safely"""
    if uploaded_file is None:
        return None
    
    try:
        file_type = uploaded_file.type
        file_size = uploaded_file.size
        
        # Check file size (max 50MB)
        if file_size > 50 * 1024 * 1024:
            st.error("File size too large. Maximum 50MB allowed.")
            return None
        
        # Process based on file type
        if file_type == 'text/csv':
            df = pd.read_csv(uploaded_file, encoding='utf-8')
        elif file_type in ['application/vnd.ms-excel', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet']:
            df = pd.read_excel(uploaded_file)
        elif file_type == 'application/json':
            df = pd.read_json(uploaded_file)
        else:
            st.error(f"Unsupported file type: {file_type}")
            return None
        
        # Basic validation
        if df.empty:
            st.error("Uploaded file is empty.")
            return None
        
        if len(df) > 10000:
            st.warning("Large file detected. Processing first 10,000 rows.")
            df = df.head(10000)
        
        st.success(f"File uploaded successfully! {len(df)} rows, {len(df.columns)} columns")
        return df
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

@st.cache_data
def safe_sentiment_analysis(texts: List[str]) -> List[Dict[str, Any]]:
    """Safe sentiment analysis with caching"""
    if not texts:
        return []
    
    results = []
    positive_words = ['yaxşı', 'əla', 'mükəmməl', 'razıyam', 'təşəkkür', 'good', 'excellent', 'perfect', 'satisfied', 'thank']
    negative_words = ['pis', 'səhv', 'problem', 'şikayət', 'narazıyam', 'yavaş', 'bad', 'wrong', 'error', 'complaint', 'slow', 'terrible']
    severity_words = ['təcili', 'dərhal', 'mütləq', 'vacib', 'ciddi', 'urgent', 'immediately', 'critical', 'serious', 'important']
    
    for text in texts:
        try:
            text_lower = str(text).lower()
            
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)
            severity_count = sum(1 for word in severity_words if word in text_lower)
            
            if pos_count > neg_count:
                sentiment = 'positive'
                score = min(0.9, 0.6 + (pos_count * 0.1))
            elif neg_count > pos_count:
                sentiment = 'negative'
                score = max(0.1, 0.4 - (neg_count * 0.1))
            else:
                sentiment = 'neutral'
                score = 0.5
            
            if severity_count >= 2 or neg_count >= 3:
                severity = 'high'
            elif severity_count == 1 or neg_count >= 2:
                severity = 'medium'
            else:
                severity = 'low'
            
            results.append({
                'sentiment': sentiment,
                'score': score,
                'severity': severity,
                'confidence': min(0.95, 0.7 + (pos_count + neg_count) * 0.05)
            })
        except Exception as e:
            # Return neutral for failed analysis
            results.append({
                'sentiment': 'neutral',
                'score': 0.5,
                'severity': 'low',
                'confidence': 0.5
            })
    
    return results

def improved_sidebar_navigation():
    """Improved sidebar with better error handling"""
    st.sidebar.markdown("### 🏦 Bank360 Analytics")
    
    # Language selector
    language_options = {'Azərbaycan': 'az', 'English': 'en'}
    current_lang_key = 'Azərbaycan' if st.session_state.language == 'az' else 'English'
    
    selected_language = st.sidebar.selectbox(
        "Language / Dil",
        list(language_options.keys()),
        index=list(language_options.keys()).index(current_lang_key)
    )
    st.session_state.language = language_options[selected_language]
    
    # API Key management
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Settings")
    
    api_key = st.sidebar.text_input(
        "Gemini API Key",
        type="password",
        value=st.session_state.gemini_api_key,
        help="Enter your Google Gemini API key for AI features",
        placeholder="AIza..."
    )
    
    if api_key != st.session_state.gemini_api_key:
        st.session_state.gemini_api_key = api_key
        if api_key:
            st.sidebar.success("API key updated!")
    
    # Navigation menu
    st.sidebar.markdown("---")
    st.sidebar.subheader("📍 Navigation")
    
    pages = {
        'az': ['Ana Səhifə', 'Şikayətlər', 'Kredit Riski', 'Məhsul Məlumatları', 'Bilik Axtarışı'],
        'en': ['Home', 'Complaints', 'Credit Risk', 'Product Insights', 'Knowledge Search']
    }
    
    selected_page = st.sidebar.radio(
        "Select Page",
        pages[st.session_state.language]
    )
    
    # System status
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 System Status")
    
    status_items = [
        ("Gemini API", "✅" if GEMINI_AVAILABLE and st.session_state.gemini_api_key else "❌"),
        ("scikit-learn", "✅" if SKLEARN_AVAILABLE else "❌"),
        ("scipy", "✅" if SCIPY_AVAILABLE else "❌")
    ]
    
    for item, status in status_items:
        st.sidebar.text(f"{item}: {status}")
    
    return selected_page

def main():
    """Main application with improved error handling"""
    # Initialize session state
    initialize_session_state()
    
    # Try to load API key from secrets
    if not st.session_state.gemini_api_key and not st.session_state.initialized:
        try:
            st.session_state.gemini_api_key = st.secrets.get("GEMINI_API_KEY", "")
            if st.session_state.gemini_api_key:
                st.toast("API key loaded from secrets", icon="🔑")
        except:
            pass  # No secrets file or key not found
        
        st.session_state.initialized = True
    
    # Initialize API
    gemini_api = safe_execute(ImprovedGeminiAPI, st.session_state.gemini_api_key)
    if not gemini_api:
        gemini_api = ImprovedGeminiAPI()  # Fallback to mock mode
    
    # Navigation
    try:
        selected_page = improved_sidebar_navigation()
        
        # Route to appropriate page
        if selected_page in ['Ana Səhifə', 'Home']:
            home_page_improved(gemini_api)
        elif selected_page in ['Şikayətlər', 'Complaints']:
            complaints_page_improved(gemini_api)
        elif selected_page in ['Kredit Riski', 'Credit Risk']:
            credit_risk_page_improved(gemini_api)
        elif selected_page in ['Məhsul Məlumatları', 'Product Insights']:
            product_insights_page_improved(gemini_api)
        elif selected_page in ['Bilik Axtarışı', 'Knowledge Search']:
            knowledge_search_page_improved(gemini_api)
            
    except Exception as e:
        st.error(f"Navigation error: {str(e)}")
        st.info("Please refresh the page and try again.")

def home_page_improved(gemini_api):
    """Improved home page with better error handling"""
    st.title("🏦 Bank360 Analytics Dashboard")
    st.markdown("---")
    
    # Load data safely
    try:
        complaint_df, loan_df, customer_df = generate_sample_data_fixed()
    except Exception as e:
        st.error(f"Error loading sample data: {str(e)}")
        return
    
    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    
    try:
        with col1:
            st.metric("Total Complaints", len(complaint_df), delta=f"+{np.random.randint(5, 15)}")
        
        with col2:
            csat_score = np.random.uniform(3.8, 4.5)
            st.metric("CSAT Score", f"{csat_score:.1f}/5.0", delta=f"+{np.random.uniform(0.1, 0.3):.1f}")
        
        with col3:
            high_severity = len(complaint_df[complaint_df['severity'] == 'high']) if 'severity' in complaint_df.columns else 0
            st.metric("High Severity", high_severity, delta=f"-{np.random.randint(1, 3)}")
        
        with col4:
            avg_pd = loan_df['debt_to_income'].mean() * 0.25 if 'debt_to_income' in loan_df.columns else 0.15
            st.metric("Avg PD", f"{avg_pd:.1%}", delta=f"{np.random.uniform(-0.01, 0.01):+.1%}")
    
    except Exception as e:
        st.error(f"Error displaying metrics: {str(e)}")
    
    st.markdown("---")
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        try:
            if 'category' in complaint_df.columns:
                category_counts = complaint_df['category'].value_counts()
                fig = px.pie(
                    values=category_counts.values,
                    names=category_counts.index,
                    title="Complaint Categories"
                )
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating category chart: {str(e)}")
    
    with col2:
        try:
            if 'date' in complaint_df.columns:
                daily_complaints = complaint_df.groupby(complaint_df['date'].dt.date).size()
                fig = px.line(
                    x=daily_complaints.index,
                    y=daily_complaints.values,
                    title="Daily Complaint Trends"
                )
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating trend chart: {str(e)}")
    
    # AI Insights section
    st.markdown("---")
    st.subheader("🤖 AI-Generated Insights")
    
    if st.button("Generate Business Insights"):
        with st.spinner("Analyzing data and generating insights..."):
            insights_prompt = f"""
            Analyze this bank's performance data and provide 3 key business insights:
            
            Data Summary:
            - Total complaints: {len(complaint_df)}
            - High severity complaints: {high_severity}
            - Average risk level: {avg_pd:.1%}
            - Most common complaint category: {complaint_df['category'].value_counts().index[0] if 'category' in complaint_df.columns else 'N/A'}
            
            Focus on actionable recommendations for improvement.
            """
            
            insights = gemini_api.generate_response(insights_prompt, st.session_state.language)
            st.write(insights)

def complaints_page_improved(gemini_api):
    """Improved complaints page with better error handling"""
    st.title("Complaints & Feedback Analysis")
    st.markdown("---")
    
    # File upload section
    st.subheader("Upload Data")
    uploaded_file = st.file_uploader(
        "Choose a CSV, Excel, or JSON file",
        type=['csv', 'xlsx', 'json'],
        help="Upload complaint data for analysis"
    )
    
    # Load data
    if uploaded_file is not None:
        data = validate_uploaded_file(uploaded_file)
        if data is not None:
            st.session_state.complaint_data = data
    else:
        # Use sample data
        try:
            complaint_df, _, _ = generate_sample_data_fixed()
            st.session_state.complaint_data = complaint_df
            st.info("Using sample data. Upload your own file to analyze real complaints.")
        except Exception as e:
            st.error(f"Error loading sample data: {str(e)}")
            return
    
    data = st.session_state.complaint_data
    
    if data is None or data.empty:
        st.warning("No data available. Please upload a valid file.")
        return
    
    # Data overview
    col1, col2, col3, col4 = st.columns(4)
    
    try:
        with col1:
            st.metric("Total Records", len(data))
        
        with col2:
            high_sev = len(data[data['severity'] == 'high']) if 'severity' in data.columns else 0
            st.metric("High Severity", high_sev)
        
        with col3:
            open_cases = len(data[data['status'] == 'Open']) if 'status' in data.columns else 0
            st.metric("Open Cases", open_cases)
        
        with col4:
            avg_days = np.random.randint(2, 7)  # Mock resolution time
            st.metric("Avg Resolution (days)", avg_days)
    
    except Exception as e:
        st.error(f"Error calculating metrics: {str(e)}")
    
    # Analysis tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Sentiment Analysis", 
        "Category Analysis", 
        "Response Generator", 
        "Trends & Patterns"
    ])
    
    with tab1:
        st.subheader("Sentiment Analysis")
        
        if 'text_az' in data.columns:
            try:
                sample_size = min(50, len(data))
                sample_texts = data['text_az'].dropna().head(sample_size).tolist()
                
                if st.button("Analyze Sentiments", key="sentiment_btn"):
                    with st.spinner("Analyzing sentiments..."):
                        sentiments = safe_sentiment_analysis(sample_texts)
                        
                        if sentiments:
                            sentiment_labels = [s['sentiment'] for s in sentiments]
                            severity_labels = [s['severity'] for s in sentiments]
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                sentiment_counts = pd.Series(sentiment_labels).value_counts()
                                fig = px.pie(
                                    values=sentiment_counts.values,
                                    names=sentiment_counts.index,
                                    title="Sentiment Distribution"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                severity_counts = pd.Series(severity_labels).value_counts()
                                colors = {'high': 'red', 'medium': 'orange', 'low': 'green'}
                                fig = px.bar(
                                    x=severity_counts.index,
                                    y=severity_counts.values,
                                    title="Severity Distribution",
                                    color=severity_counts.index,
                                    color_discrete_map=colors
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("Could not analyze sentiments")
            except Exception as e:
                st.error(f"Error in sentiment analysis: {str(e)}")
        else:
            st.warning("No text column found for sentiment analysis")
    
    with tab2:
        st.subheader("Category Analysis")
        
        if 'category' in data.columns:
            try:
                category_counts = data['category'].value_counts()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.bar(
                        x=category_counts.values,
                        y=category_counts.index,
                        orientation='h',
                        title="Complaints by Category"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    if 'severity' in data.columns:
                        severity_by_cat = pd.crosstab(data['category'], data['severity'])
                        fig = px.bar(
                            severity_by_cat,
                            title="Severity Distribution by Category",
                            barmode='stack'
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            except Exception as e:
                st.error(f"Error in category analysis: {str(e)}")
        else:
            st.warning("No category column found")
    
    with tab3:
        st.subheader("AI Response Generator")
        
        if 'text_az' in data.columns:
            complaint_options = data['text_az'].dropna().head(10).tolist()
            
            if complaint_options:
                selected_complaint = st.selectbox(
                    "Select a complaint to generate response:",
                    complaint_options,
                    key="response_complaint"
                )
                
                if st.button("Generate Professional Response", key="generate_response_btn"):
                    with st.spinner("Generating response..."):
                        try:
                            response = gemini_api.generate_response(
                                f"Generate a professional response to this bank complaint: {selected_complaint}",
                                st.session_state.language
                            )
                            
                            st.success("Response generated successfully!")
                            st.write("**Generated Response:**")
                            st.write(response)
                            
                        except Exception as e:
                            st.error(f"Error generating response: {str(e)}")
            else:
                st.warning("No complaints available for response generation")
        else:
            st.warning("No text data available")
    
    with tab4:
        st.subheader("Trends & Patterns")
        
        try:
            if 'date' in data.columns:
                # Daily complaint trends
                data['date'] = pd.to_datetime(data['date'])
                daily_complaints = data.groupby(data['date'].dt.date).size()
                
                fig = px.line(
                    x=daily_complaints.index,
                    y=daily_complaints.values,
                    title="Daily Complaint Volume"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Weekly patterns
                data['day_of_week'] = data['date'].dt.day_name()
                weekly_pattern = data['day_of_week'].value_counts()
                
                fig = px.bar(
                    x=weekly_pattern.index,
                    y=weekly_pattern.values,
                    title="Complaints by Day of Week"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.info("Date column not found. Cannot show temporal trends.")
                
        except Exception as e:
            st.error(f"Error in trend analysis: {str(e)}")

def credit_risk_page_improved(gemini_api):
    """Improved credit risk page with better error handling"""
    st.title("Credit Risk & Expected Loss Analysis")
    st.markdown("---")
    
    # Input section
    st.subheader("Customer Risk Assessment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Personal Information**")
        age = st.slider("Age", 18, 80, 35, key="risk_age")
        income = st.number_input("Monthly Income (AZN)", 300.0, 15000.0, 1500.0, key="risk_income")
        employment = st.selectbox("Employment Status", 
                                ['government', 'employed', 'self_employed', 'unemployed'], 
                                key="risk_employment")
        credit_score = st.slider("Credit Score", 300, 850, 650, key="risk_credit_score")
    
    with col2:
        st.write("**Loan Information**")
        loan_amount = st.number_input("Loan Amount (AZN)", 1000.0, 100000.0, 25000.0, key="risk_loan_amount")
        debt_to_income = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.3, key="risk_dti")
        collateral_value = st.number_input("Collateral Value (AZN)", 0.0, 200000.0, 30000.0, key="risk_collateral")
        loan_to_value = st.slider("Loan-to-Value Ratio", 0.0, 1.0, 0.8, key="risk_ltv")
    
    # Calculate risk button
    if st.button("Calculate Risk Metrics", key="calc_risk_btn"):
        try:
            # Calculate PD using simplified model
            pd_score = calculate_pd_simple(age, income, employment, credit_score, debt_to_income, loan_to_value)
            
            # Calculate LGD
            if collateral_value >= loan_amount:
                lgd = 0.2  # Low LGD with sufficient collateral
            else:
                collateral_ratio = collateral_value / loan_amount if loan_amount > 0 else 0
                lgd = max(0.3, 0.8 - (collateral_ratio * 0.5))
            
            # Calculate EAD (simplified)
            ead = loan_amount * 0.85
            
            # Calculate Expected Loss
            expected_loss = pd_score * lgd * ead
            unexpected_loss = ead * lgd * np.sqrt(pd_score * (1 - pd_score))
            
            # Display results
            st.markdown("---")
            st.subheader("Risk Assessment Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                risk_level = "High" if pd_score > 0.2 else "Medium" if pd_score > 0.1 else "Low"
                risk_color = "red" if risk_level == "High" else "orange" if risk_level == "Medium" else "green"
                
                st.metric("Probability of Default (PD)", f"{pd_score:.2%}")
                st.markdown(f"**Risk Level:** <span style='color:{risk_color}'>{risk_level}</span>", 
                          unsafe_allow_html=True)
            
            with col2:
                st.metric("Loss Given Default (LGD)", f"{lgd:.2%}")
                st.metric("Exposure at Default (EAD)", f"{ead:,.0f} AZN")
            
            with col3:
                st.metric("Expected Loss (EL)", f"{expected_loss:,.0f} AZN")
                st.metric("Unexpected Loss (UL)", f"{unexpected_loss:,.0f} AZN")
            
            # Risk explanation
            st.subheader("Risk Assessment Explanation")
            with st.expander("View Detailed Analysis"):
                explanation_prompt = f"""
                Provide a detailed credit risk assessment explanation:
                
                Customer Profile:
                - Age: {age} years
                - Monthly Income: {income:,.0f} AZN
                - Employment: {employment}
                - Credit Score: {credit_score}
                
                Loan Details:
                - Amount: {loan_amount:,.0f} AZN
                - Debt-to-Income: {debt_to_income:.1%}
                - Loan-to-Value: {loan_to_value:.1%}
                
                Risk Metrics:
                - PD: {pd_score:.2%}
                - Expected Loss: {expected_loss:,.0f} AZN
                - Risk Level: {risk_level}
                
                Explain the key risk factors and provide recommendations.
                """
                
                try:
                    explanation = gemini_api.generate_response(explanation_prompt, st.session_state.language)
                    st.write(explanation)
                except Exception as e:
                    st.error(f"Error generating explanation: {str(e)}")
            
        except Exception as e:
            st.error(f"Error in risk calculation: {str(e)}")

def calculate_pd_simple(age, income, employment, credit_score, debt_to_income, loan_to_value):
    """Simplified PD calculation"""
    base_pd = 0.15
    
    # Age factor
    if age < 25 or age > 65:
        age_factor = 0.03
    elif 35 <= age <= 50:
        age_factor = -0.02
    else:
        age_factor = 0
    
    # Income factor
    income_factor = -0.00002 * income if income > 0 else 0.1
    
    # Employment factor
    emp_factors = {'government': -0.03, 'employed': -0.01, 'self_employed': 0.02, 'unemployed': 0.15}
    employment_factor = emp_factors.get(employment, 0)
    
    # Credit score factor
    credit_factor = -0.0002 * (credit_score - 600)
    
    # DTI factor
    dti_factor = debt_to_income * 0.1
    
    # LTV factor
    ltv_factor = loan_to_value * 0.05
    
    pd = base_pd + age_factor + income_factor + employment_factor + credit_factor + dti_factor + ltv_factor
    return max(0.01, min(0.95, pd))

def product_insights_page_improved(gemini_api):
    """Improved product insights page"""
    st.title("Product Insights & Cross-Sell Analysis")
    st.markdown("---")
    
    try:
        # Load sample data
        _, _, customer_df = generate_sample_data_fixed()
        
        # Customer segmentation
        st.subheader("Customer Segmentation")
        
        # Add segments to customer data
        def assign_segment(row):
            age, income, tenure = row['age'], row['income'], row['tenure_months']
            
            if 25 <= age <= 35 and income >= 1200 and tenure <= 24:
                return 'Young Professional'
            elif 35 <= age <= 50 and income >= 1800:
                return 'Established'
            elif income >= 3000:
                return 'Premium'
            elif age >= 55:
                return 'Senior'
            elif age <= 25:
                return 'Student/Starter'
            else:
                return 'Mass Market'
        
        customer_df['segment'] = customer_df.apply(assign_segment, axis=1)
        
        # Display segments
        col1, col2 = st.columns(2)
        
        with col1:
            segment_counts = customer_df['segment'].value_counts()
            fig = px.pie(values=segment_counts.values, names=segment_counts.index, 
                        title="Customer Segments")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Income distribution by segment
            fig = px.box(customer_df, x='segment', y='income', 
                        title="Income Distribution by Segment")
            fig.update_xaxis(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        # Cross-sell analysis
        st.subheader("Cross-Sell Opportunities")
        
        selected_customer_id = st.selectbox(
            "Select Customer for Analysis:",
            customer_df['customer_id'].head(20).tolist(),
            key="product_customer_select"
        )
        
        if selected_customer_id:
            customer_data = customer_df[customer_df['customer_id'] == selected_customer_id].iloc[0]
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.write("**Customer Profile:**")
                st.write(f"Age: {customer_data['age']}")
                st.write(f"Income: {customer_data['income']:,.0f} AZN")
                st.write(f"Segment: {customer_data['segment']}")
                st.write(f"Tenure: {customer_data['tenure_months']} months")
                st.write(f"Current Products: {customer_data['num_products']}")
            
            with col2:
                # Calculate product propensities (simplified)
                products = {
                    'Credit Card': calculate_product_propensity(customer_data, 'credit_card'),
                    'Personal Loan': calculate_product_propensity(customer_data, 'personal_loan'),
                    'Mortgage': calculate_product_propensity(customer_data, 'mortgage'),
                    'Investment Account': calculate_product_propensity(customer_data, 'investment'),
                    'Insurance': calculate_product_propensity(customer_data, 'insurance')
                }
                
                prop_df = pd.DataFrame(list(products.items()), columns=['Product', 'Propensity'])
                prop_df = prop_df.sort_values('Propensity', ascending=True)
                
                fig = px.bar(prop_df, x='Propensity', y='Product', orientation='h',
                           title=f"Product Propensity for Customer {selected_customer_id}",
                           color='Propensity', color_continuous_scale='viridis')
                st.plotly_chart(fig, use_container_width=True)
                
                # Top recommendations
                st.write("**Top 3 Recommendations:**")
                top_3 = prop_df.tail(3)
                for _, row in top_3.iterrows():
                    st.write(f"• {row['Product']}: {row['Propensity']:.1%} likelihood")
        
        # Marketing strategy
        st.subheader("Marketing Strategy Recommendations")
        
        if st.button("Generate Strategy", key="strategy_btn"):
            with st.spinner("Generating marketing strategy..."):
                strategy_prompt = f"""
                Generate marketing strategy recommendations based on customer segments:
                
                Segment Distribution:
                {dict(customer_df['segment'].value_counts())}
                
                Average Income by Segment:
                {customer_df.groupby('segment')['income'].mean().to_dict()}
                
                Provide specific product recommendations and marketing approaches for each segment.
                """
                
                try:
                    strategy = gemini_api.generate_response(strategy_prompt, st.session_state.language)
                    st.write(strategy)
                except Exception as e:
                    st.error(f"Error generating strategy: {str(e)}")
    
    except Exception as e:
        st.error(f"Error in product insights page: {str(e)}")

def calculate_product_propensity(customer_data, product):
    """Calculate product propensity score"""
    age = customer_data['age']
    income = customer_data['income']
    segment = customer_data['segment']
    
    base_scores = {
        'credit_card': 0.4,
        'personal_loan': 0.25,
        'mortgage': 0.15,
        'investment': 0.2,
        'insurance': 0.3
    }
    
    score = base_scores.get(product, 0.25)
    
    # Age adjustments
    if product == 'credit_card' and 25 <= age <= 45:
        score += 0.15
    elif product == 'mortgage' and 28 <= age <= 45:
        score += 0.2
    elif product == 'investment' and age >= 35:
        score += 0.15
    
    # Income adjustments
    if income >= 2500:
        score += 0.1
    elif income >= 1500:
        score += 0.05
    
    # Segment adjustments
    if segment == 'Premium':
        score += 0.15
    elif segment == 'Young Professional':
        if product in ['credit_card', 'personal_loan']:
            score += 0.1
    
    return min(0.95, score)

def knowledge_search_page_improved(gemini_api):
    """Improved knowledge search page"""
    st.title("Knowledge Search & RAG System")
    st.markdown("---")
    
    # Initialize knowledge base if not exists
    if 'kb_docs' not in st.session_state:
        st.session_state.kb_docs = [
            {
                'title': 'Kredit Kartı Qaydaları',
                'content': 'Kredit kartının istifadə qaydaları: Aylıq komissiya 2 AZN, nağd pul çıxarma 1.5%, minimum ödəniş 5%. 24/7 online idarəetmə. Cashback proqramı mövcuddur.',
                'category': 'products'
            },
            {
                'title': 'Mobil Banking Xidmətləri',
                'content': 'Mobil tətbiq vasitəsilə: pul köçürmələri, hesab yoxlanması, kommunal ödənişlər, kredit ödənişləri. Biometrik giriş, push bildirişlər.',
                'category': 'digital'
            },
            {
                'title': 'Kredit Şərtləri',
                'content': 'Fərdi kreditlər: minimum gəlir 500 AZN, maksimum 50,000 AZN, müddət 60 aya qədər, faiz 12-18%. Zəmanət və ya girov tələb olunur.',
                'category': 'loans'
            }
        ]
    
    # Document management
    st.subheader("Knowledge Base Management")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        with st.expander("Add New Document"):
            title = st.text_input("Document Title", key="kb_title")
            category = st.selectbox("Category", ['products', 'digital', 'loans', 'general'], key="kb_category")
            content = st.text_area("Content", height=100, key="kb_content")
            
            if st.button("Add Document", key="add_doc_btn"):
                if title and content:
                    new_doc = {
                        'title': title,
                        'content': content,
                        'category': category
                    }
                    st.session_state.kb_docs.append(new_doc)
                    st.success(f"Document '{title}' added successfully!")
                    st.rerun()
                else:
                    st.warning("Please fill in both title and content.")
    
    with col2:
        st.metric("Total Documents", len(st.session_state.kb_docs))
        
        categories = [doc['category'] for doc in st.session_state.kb_docs]
        if categories:
            cat_counts = pd.Series(categories).value_counts()
            for cat, count in cat_counts.items():
                st.write(f"{cat}: {count}")
    
    # Search interface
    st.subheader("Knowledge Search")
    
    query = st.text_input(
        "Ask a question about bank services:",
        placeholder="Kredit kartının komissiyası nə qədərdir?" if st.session_state.language == 'az' 
                   else "What are the credit card fees?",
        key="kb_query"
    )
    
    if query:
        try:
            # Simple search implementation
            relevant_docs = search_documents(st.session_state.kb_docs, query)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write("**AI Response:**")
                
                if relevant_docs:
                    context = " ".join([doc['content'] for doc in relevant_docs[:2]])
                    
                    answer_prompt = f"""
                    Answer the question based on this information:
                    
                    Context: {context}
                    Question: {query}
                    
                    Provide a helpful and accurate answer.
                    """
                    
                    with st.spinner("Generating answer..."):
                        answer = gemini_api.generate_response(answer_prompt, st.session_state.language)
                        st.write(answer)
                else:
                    st.write("Sorry, I couldn't find relevant information for your question.")
            
            with col2:
                st.write("**Relevant Documents:**")
                
                for i, doc in enumerate(relevant_docs[:3]):
                    with st.expander(f"{doc['title']} ({doc.get('score', 0):.2f})"):
                        st.write(doc['content'][:200] + "...")
        
        except Exception as e:
            st.error(f"Error in search: {str(e)}")

def search_documents(docs, query):
    """Simple document search implementation"""
    query_words = query.lower().split()
    
    scored_docs = []
    for doc in docs:
        content_lower = doc['content'].lower()
        title_lower = doc['title'].lower()
        
        # Calculate simple relevance score
        content_score = sum(1 for word in query_words if word in content_lower)
        title_score = sum(2 for word in query_words if word in title_lower)  # Title matches are worth more
        
        total_score = content_score + title_score
        
        if total_score > 0:
            doc_copy = doc.copy()
            doc_copy['score'] = total_score / len(query_words)
            scored_docs.append(doc_copy)
    
    # Sort by score, descending
    return sorted(scored_docs, key=lambda x: x['score'], reverse=True)

# Run the improved application
if __name__ == "__main__":
    main()
