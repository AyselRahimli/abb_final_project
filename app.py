# Bank360 - GenAI Customer Analytics & Insights Platform
# Complete Streamlit Application for Azerbaijan Banking with Gemini API Integration

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import io
import base64
from typing import List, Dict, Any
import re
import warnings
warnings.filterwarnings('ignore')

# ML and NLP libraries
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats

# PDF generation
from reportlab.lib.pagesizes import letter, A4
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

# Configure Streamlit page
st.set_page_config(
    page_title="Bank360 Analytics",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #1f4e79;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background-color: #f0f8ff;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1f4e79;
}
.risk-high { background-color: #ffebee; border-left-color: #f44336; }
.risk-medium { background-color: #fff3e0; border-left-color: #ff9800; }
.risk-low { background-color: #e8f5e8; border-left-color: #4caf50; }
.sidebar-section {
    padding: 1rem;
    margin: 0.5rem 0;
    background-color: #f8f9fa;
    border-radius: 0.5rem;
    border: 1px solid #dee2e6;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'language' not in st.session_state:
    st.session_state.language = 'az'
if 'complaint_data' not in st.session_state:
    st.session_state.complaint_data = None
if 'loan_data' not in st.session_state:
    st.session_state.loan_data = None
if 'customer_data' not in st.session_state:
    st.session_state.customer_data = None
if 'knowledge_base' not in st.session_state:
    st.session_state.knowledge_base = []
if 'gemini_api_key' not in st.session_state:
    st.session_state.gemini_api_key = ""

# Language translations
TRANSLATIONS = {
    'az': {
        'title': 'Bank360 - GenAI Müştəri Analitikası və Məlumat Platforması',
        'home': 'Ana Səhifə',
        'complaints': 'Şikayətlər və Rəylər',
        'credit_risk': 'Kredit Riski və Gözlənilən İtki',
        'product_insights': 'Məhsul Məlumatları və Çarpaz Satış',
        'knowledge_search': 'Bilik Axtarışı (RAG)',
        'overview': 'Ümumi Baxış',
        'total_complaints': 'Ümumi Şikayətlər',
        'csat_index': 'CSAT İndeksi',
        'high_severity': 'Yüksək Prioritet',
        'avg_pd': 'Orta PD',
        'expected_loss': 'Gözlənilən İtki',
        'top_products': 'Ən Yaxşı Məhsullar',
        'churn_risk': 'Müştəri İtirmə Riski',
        'upload_data': 'Məlumatları Yüklə',
        'analyze': 'Təhlil Et',
        'generate_report': 'Hesabat Yarat',
        'language': 'Dil',
        'settings': 'Tənzimləmələr',
        'api_key': 'Gemini API Açarı',
        'enter_api_key': 'API açarını daxil edin',
        'processing': 'İşlənir...',
        'analysis_complete': 'Təhlil tamamlandı',
        'download_report': 'Hesabatı Yüklə',
        'customer_segments': 'Müştəri Seqmentləri',
        'risk_analysis': 'Risk Analizi',
        'product_performance': 'Məhsul Performansı',
        'recommendations': 'Tövsiyələr',
        'sentiment_analysis': 'Hissiyyat Analizi',
        'category_distribution': 'Kateqoriya Bölgüsü',
        'pd_calculation': 'PD Hesablaması',
        'scenario_analysis': 'Ssenari Analizi',
        'cross_sell': 'Çarpaz Satış İmkanları',
        'propensity_score': 'Meyillik Xalı',
        'search_knowledge': 'Bilik bazasında axtarış',
        'ask_question': 'Sual verin'
    },
    'en': {
        'title': 'Bank360 - GenAI Customer Analytics & Insights Platform',
        'home': 'Home',
        'complaints': 'Complaints & Feedback',
        'credit_risk': 'Credit Risk & Expected Loss',
        'product_insights': 'Product Insights & Cross-Sell',
        'knowledge_search': 'Knowledge Search (RAG)',
        'overview': 'Overview',
        'total_complaints': 'Total Complaints',
        'csat_index': 'CSAT Index',
        'high_severity': 'High Severity',
        'avg_pd': 'Avg PD',
        'expected_loss': 'Expected Loss',
        'top_products': 'Top Products',
        'churn_risk': 'Churn Risk',
        'upload_data': 'Upload Data',
        'analyze': 'Analyze',
        'generate_report': 'Generate Report',
        'language': 'Language',
        'settings': 'Settings',
        'api_key': 'Gemini API Key',
        'enter_api_key': 'Enter API key',
        'processing': 'Processing...',
        'analysis_complete': 'Analysis Complete',
        'download_report': 'Download Report',
        'customer_segments': 'Customer Segments',
        'risk_analysis': 'Risk Analysis',
        'product_performance': 'Product Performance',
        'recommendations': 'Recommendations',
        'sentiment_analysis': 'Sentiment Analysis',
        'category_distribution': 'Category Distribution',
        'pd_calculation': 'PD Calculation',
        'scenario_analysis': 'Scenario Analysis',
        'cross_sell': 'Cross-Sell Opportunities',
        'propensity_score': 'Propensity Score',
        'search_knowledge': 'Search knowledge base',
        'ask_question': 'Ask a question'
    }
}

def t(key: str) -> str:
    """Get translation for current language"""
    return TRANSLATIONS[st.session_state.language].get(key, key)

class GeminiAPI:
    """Wrapper for Google Gemini API"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.model = None
        
        if api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel('gemini-2.5-flash')
            except Exception as e:
                st.error(f"Gemini API initialization error: {str(e)}")
    
    def generate_response(self, prompt: str, language: str = 'az') -> str:
        """Generate response using Gemini API"""
        if not self.model:
            return self._mock_response(prompt, language)
        
        try:
            lang_instruction = "Cavabı Azərbaycan dilində verin" if language == 'az' else "Provide response in English"
            full_prompt = f"{lang_instruction}. {prompt}"
            response = self.model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            return self._mock_response(prompt, language)
    
    def _mock_response(self, prompt: str, language: str = 'az') -> str:
        """Mock response for demo purposes"""
        if 'complaint' in prompt.lower() or 'şikayət' in prompt.lower():
            if language == 'az':
                return "Hörmətli müştəri, şikayətinizi qəbul edirik. Problemin həlli üçün müvafiq şöbə ilə əlaqə saxlayacağıq və 3 iş günü ərzində geri dönüş edəcəyik. Göstərdiyiniz səbr üçün təşəkkür edirik."
            else:
                return "Dear customer, we acknowledge your complaint. We will contact the relevant department to resolve the issue and get back to you within 3 business days. Thank you for your patience."
        elif 'credit' in prompt.lower() or 'kredit' in prompt.lower():
            if language == 'az':
                return "Kredit analizi əsasında, müştərinin gəlir səviyyəsi və kredit tarixi nəzərə alınmışdır. Risk səviyyəsi orta hesablanır və əlavə təminat tələb olunur."
            else:
                return "Based on credit analysis, customer's income level and credit history have been considered. Risk level is assessed as medium and additional collateral is required."
        elif 'strategy' in prompt.lower() or 'strategiya' in prompt.lower():
            if language == 'az':
                return "Marketinq strategiyası: 1) Gənc peşəkarlar seqmentinə kredit kartları təklif edin, 2) Premium müştərilər üçün investisiya məhsulları, 3) Yaşlı müştərilər üçün pensiya planları."
            else:
                return "Marketing strategy: 1) Offer credit cards to young professionals segment, 2) Investment products for premium customers, 3) Retirement plans for senior customers."
        else:
            if language == 'az':
                return "Sorğunuza əsasən təhlil aparılmış və uyğun tövsiyələr hazırlanmışdır. Daha ətraflı məlumat üçün müvafiq bölməyə müraciət edin."
            else:
                return "Analysis has been conducted based on your query and appropriate recommendations have been prepared. Please refer to the relevant section for more details."

class ComplaintAnalyzer:
    """Advanced complaint and feedback analysis module"""
    
    def __init__(self, gemini_api: GeminiAPI):
        self.gemini_api = gemini_api
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words=['və', 'bu', 'olan', 'the', 'and', 'or'])
        self.lda_model = None
    
    def categorize_complaint(self, text: str) -> str:
        """Categorize complaint using keyword matching and LLM enhancement"""
        categories = {
            'kart': ['kart', 'card', 'debit', 'kredit kartı', 'visa', 'mastercard'],
            'mobil': ['mobil', 'app', 'tətbiq', 'telefon', 'mobile', 'application'],
            'filial': ['filial', 'branch', 'xidmət', 'service', 'kassir', 'teller'],
            'komissiya': ['komissiya', 'fee', 'ödəniş', 'payment', 'charge', 'cost'],
            'atm': ['atm', 'bankomat', 'nağd', 'cash', 'withdrawal'],
            'internet': ['internet', 'online', 'sayt', 'website', 'portal'],
            'kredit': ['kredit', 'loan', 'borc', 'debt', 'mortgage'],
            'depozit': ['depozit', 'deposit', 'savings', 'account', 'hesab']
        }
        
        text_lower = text.lower()
        scores = {}
        
        for category, keywords in categories.items():
            score = sum(2 if keyword in text_lower else 0 for keyword in keywords)
            if score > 0:
                scores[category] = score
        
        return max(scores.items(), key=lambda x: x[1])[0] if scores else 'digər'
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Advanced sentiment and severity analysis"""
        positive_words = [
            'yaxşı', 'əla', 'mükəmməl', 'razıyam', 'təşəkkür',
            'good', 'excellent', 'perfect', 'satisfied', 'thank'
        ]
        negative_words = [
            'pis', 'səhv', 'problem', 'şikayət', 'narazıyam', 'yavaş',
            'bad', 'wrong', 'error', 'complaint', 'slow', 'terrible'
        ]
        severity_words = [
            'təcili', 'dərhal', 'mütləq', 'vacib', 'ciddi',
            'urgent', 'immediately', 'critical', 'serious', 'important'
        ]
        
        text_lower = text.lower()
        
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
        
        return {
            'sentiment': sentiment,
            'score': score,
            'severity': severity,
            'confidence': min(0.95, 0.7 + (pos_count + neg_count) * 0.05)
        }
    
    def topic_modeling(self, texts: List[str], n_topics: int = 5) -> Dict[str, Any]:
        """Perform topic modeling using LDA"""
        if len(texts) < 2:
            return {'topics': [], 'doc_topics': []}
        
        try:
            vectorizer = CountVectorizer(max_features=100, stop_words=['və', 'bu', 'olan'])
            doc_term_matrix = vectorizer.fit_transform(texts)
            
            lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
            lda.fit(doc_term_matrix)
            
            feature_names = vectorizer.get_feature_names_out()
            
            topics = []
            for topic_idx, topic in enumerate(lda.components_):
                top_words = [feature_names[i] for i in topic.argsort()[-10:]]
                topics.append({
                    'id': topic_idx,
                    'words': top_words[::-1],
                    'weight': float(topic.max())
                })
            
            doc_topics = lda.transform(doc_term_matrix)
            
            return {
                'topics': topics,
                'doc_topics': doc_topics.tolist()
            }
        except Exception as e:
            return {'topics': [], 'doc_topics': []}
    
    def generate_response(self, complaint: str, language: str = 'az') -> str:
        """Generate professional response to complaint"""
        if language == 'az':
            prompt = f"""
            Aşağıdakı bank müştərisi şikayətinə professional və həlledici cavab yazın:
            
            Şikayət: {complaint}
            
            Cavab aşağıdakı elementləri əhatə etməlidir:
            1. Müştəriyə təşəkkür və üzr istəmə
            2. Problemin qəbul edilməsi və başa düşülməsi
            3. Konkret həll yolları və addımlar
            4. Vaxt çərçivəsi və gələcək təminat
            5. Əlaqə məlumatları
            
            Ton: professional, empatik, həlledici
            """
        else:
            prompt = f"""
            Write a professional and solution-oriented response to this bank customer complaint:
            
            Complaint: {complaint}
            
            Response should include:
            1. Thank customer and apologize
            2. Acknowledge and understand the problem
            3. Specific solutions and action steps
            4. Timeline and future assurance
            5. Contact information
            
            Tone: professional, empathetic, solution-focused
            """
        
        return self.gemini_api.generate_response(prompt, language)

class CreditRiskAnalyzer:
    """Advanced credit risk and expected loss analysis module"""
    
    def __init__(self, gemini_api: GeminiAPI):
        self.gemini_api = gemini_api
        self.scaler = StandardScaler()
    
    def calculate_pd_advanced(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Calculate Probability of Default using advanced scoring"""
        base_score = 0.15
        
        age = features.get('age', 35)
        if age < 25:
            age_factor = 0.05
        elif age < 35:
            age_factor = -0.02
        elif age < 50:
            age_factor = -0.03
        elif age < 65:
            age_factor = 0.01
        else:
            age_factor = 0.04
        
        income = features.get('income', 1000)
        if income <= 0:
            income_factor = 0.2
        else:
            income_factor = -0.05 * np.log(income / 500)
        
        employment = features.get('employment', 'employed')
        employment_factors = {
            'government': -0.04,
            'employed': -0.02,
            'self_employed': 0.03,
            'unemployed': 0.15,
            'retired': 0.02
        }
        employment_factor = employment_factors.get(employment, 0)
        
        credit_score = features.get('credit_score', 600)
        if credit_score >= 750:
            credit_factor = -0.08
        elif credit_score >= 650:
            credit_factor = -0.03
        elif credit_score >= 550:
            credit_factor = 0.02
        else:
            credit_factor = 0.1
        
        dti = features.get('debt_to_income', 0.3)
        if dti > 0.5:
            dti_factor = 0.06
        elif dti > 0.3:
            dti_factor = 0.02
        else:
            dti_factor = -0.01
        
        ltv = features.get('loan_to_value', 0)
        if ltv > 0.9:
            ltv_factor = 0.04
        elif ltv > 0.7:
            ltv_factor = 0.02
        else:
            ltv_factor = -0.01
        
        pd = base_score + age_factor + income_factor + employment_factor + credit_factor + dti_factor + ltv_factor
        pd = max(0.005, min(0.95, pd))
        
        return {
            'pd': pd,
            'components': {
                'base': base_score,
                'age': age_factor,
                'income': income_factor,
                'employment': employment_factor,
                'credit_score': credit_factor,
                'debt_to_income': dti_factor,
                'loan_to_value': ltv_factor
            }
        }
    
    def calculate_lgd(self, collateral_value: float, loan_amount: float, recovery_rate: float = 0.6) -> float:
        """Calculate Loss Given Default"""
        if collateral_value >= loan_amount:
            return max(0.1, 1 - recovery_rate)
        else:
            collateral_coverage = collateral_value / loan_amount
            return max(0.2, 1 - (recovery_rate * collateral_coverage))
    
    def calculate_ead(self, loan_amount: float, utilization_rate: float = 0.85) -> float:
        """Calculate Exposure at Default"""
        return loan_amount * utilization_rate
    
    def expected_loss_calculation(self, pd: float, lgd: float, ead: float) -> Dict[str, float]:
        """Calculate Expected Loss and related metrics"""
        el = pd * lgd * ead
        unexpected_loss = ead * lgd * np.sqrt(pd * (1 - pd))
        
        return {
            'expected_loss': el,
            'unexpected_loss': unexpected_loss,
            'total_risk': el + unexpected_loss,
            'risk_adjusted_return': -el
        }
    
    def scenario_analysis(self, base_pd: float, lgd: float, ead: float) -> Dict[str, Dict]:
        """Perform scenario analysis"""
        scenarios = {
            'optimistic': {'pd_mult': 0.7, 'lgd_mult': 0.8, 'ead_mult': 0.9},
            'base': {'pd_mult': 1.0, 'lgd_mult': 1.0, 'ead_mult': 1.0},
            'stress': {'pd_mult': 1.5, 'lgd_mult': 1.2, 'ead_mult': 1.1},
            'severe_stress': {'pd_mult': 2.0, 'lgd_mult': 1.4, 'ead_mult': 1.2}
        }
        
        results = {}
        for scenario_name, multipliers in scenarios.items():
            scenario_pd = min(0.95, base_pd * multipliers['pd_mult'])
            scenario_lgd = min(1.0, lgd * multipliers['lgd_mult'])
            scenario_ead = ead * multipliers['ead_mult']
            
            el_result = self.expected_loss_calculation(scenario_pd, scenario_lgd, scenario_ead)
            
            results[scenario_name] = {
                'pd': scenario_pd,
                'lgd': scenario_lgd,
                'ead': scenario_ead,
                **el_result
            }
        
        return results

class ProductInsightAnalyzer:
    """Advanced product insights and cross-sell analysis module"""
    
    def __init__(self, gemini_api: GeminiAPI):
        self.gemini_api = gemini_api
    
    def segment_customers_advanced(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced customer segmentation using multiple criteria"""
        def assign_segment(row):
            age = row.get('age', 30)
            income = row.get('income', 1000)
            tenure = row.get('tenure_months', 12)
            products = row.get('num_products', 1)
            
            if 25 <= age <= 35 and income >= 1200 and tenure <= 24:
                return 'Young Professional'
            elif 35 <= age <= 50 and income >= 1800 and products >= 2:
                return 'Established'
            elif income >= 3000 or (income >= 2000 and tenure >= 36):
                return 'Premium'
            elif age >= 55:
                return 'Senior'
            elif age <= 25 and income <= 800:
                return 'Student/Starter'
            else:
                return 'Mass Market'
        
        df['segment'] = df.apply(assign_segment, axis=1)
        
        def assign_value_tier(row):
            score = 0
            score += row.get('income', 0) * 0.001
            score += row.get('num_products', 1) * 0.5
            score += min(row.get('tenure_months', 12) / 12, 5) * 0.3
            
            if score >= 3:
                return 'High Value'
            elif score >= 1.5:
                return 'Medium Value'
            else:
                return 'Low Value'
        
        df['value_tier'] = df.apply(assign_value_tier, axis=1)
        
        return df
    
    def calculate_propensity_advanced(self, customer_data: Dict[str, Any], product: str) -> Dict[str, float]:
        """Calculate advanced product propensity scores"""
        age = customer_data.get('age', 30)
        income = customer_data.get('income', 1000)
        current_products = customer_data.get('products', [])
        segment = customer_data.get('segment', 'Mass Market')
        tenure = customer_data.get('tenure_months', 12)
        
        base_propensities = {
            'credit_card': 0.35,
            'personal_loan': 0.25,
            'mortgage': 0.15,
            'savings_account': 0.45,
            'investment_account': 0.20,
            'insurance': 0.30,
            'business_account': 0.10
        }
        
        base_score = base_propensities.get(product, 0.25)
        
        # Product-specific adjustments
        if product == 'credit_card':
            if 25 <= age <= 45 and income >= 1000:
                base_score += 0.2
            if 'savings_account' in current_products:
                base_score += 0.15
        elif product == 'mortgage':
            if 28 <= age <= 45 and income >= 2000:
                base_score += 0.25
            if tenure >= 24:
                base_score += 0.1
        elif product == 'investment_account':
            if income >= 2500:
                base_score += 0.3
            if segment == 'Premium':
                base_score += 0.2
        
        return {
            'propensity': min(0.95, base_score),
            'confidence': 0.8,
            'factors': ['income', 'age', 'existing_products']
        }

class KnowledgeBase:
    """RAG (Retrieval Augmented Generation) Knowledge Base"""
    
    def __init__(self, gemini_api: GeminiAPI):
        self.gemini_api = gemini_api
        self.vectorizer = TfidfVectorizer(max_features=500)
        self.documents = []
        self.doc_vectors = None
    
    def add_document(self, title: str, content: str, metadata: Dict = None):
        """Add document to knowledge base"""
        doc = {
            'id': len(self.documents),
            'title': title,
            'content': content,
            'metadata': metadata or {},
            'timestamp': datetime.now()
        }
        self.documents.append(doc)
        self._reindex()
    
    def _reindex(self):
        """Reindex all documents"""
        if self.documents:
            contents = [doc['content'] for doc in self.documents]
            self.doc_vectors = self.vectorizer.fit_transform(contents)
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search knowledge base for relevant documents"""
        if not self.documents or self.doc_vectors is None:
            return []
        
        try:
            query_vector = self.vectorizer.transform([query])
            similarities = (self.doc_vectors * query_vector.T).toarray().flatten()
            
            top_indices = similarities.argsort()[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.1:
                    doc = self.documents[idx].copy()
                    doc['similarity'] = float(similarities[idx])
                    results.append(doc)
            
            return results
        except Exception as e:
            return []
    
    def generate_answer(self, query: str, language: str = 'az') -> str:
        """Generate answer using RAG approach"""
        relevant_docs = self.search(query, top_k=3)
        
        if not relevant_docs:
            if language == 'az':
                return "Üzr istəyirəm, bu mövzu üzrə məlumat tapılmadı. Zəhmət olmasa sualınızı daha dəqiq ifadə edin."
            else:
                return "I'm sorry, no information was found on this topic. Please try rephrasing your question."
        
        context = "\n".join([doc['content'][:500] for doc in relevant_docs])
        
        if language == 'az':
            prompt = f"""
            Aşağıdakı məlumatları əsasında sualı cavablandır:
            
            Kontekst: {context}
            
            Sual: {query}
            
            Cavab dəqiq, faydalı və Azərbaycan dilində olmalıdır.
            """
        else:
            prompt = f"""
            Answer the question based on the following information:
            
            Context: {context}
            
            Question: {query}
            
            Answer should be accurate, helpful, and in English.
            """
        
        return self.gemini_api.generate_response(prompt, language)

def generate_sample_data():
    """Generate sample data for demo purposes"""

    # Fix text_az to have exactly 100 entries
    base_texts = [
        "Mobil tətbiqdə problem var, giriş edə bilmirəm",
        "ATM-dən pul çıxarmaq mümkün olmur",
        "Kart komissiyası çox yüksəkdir",
        "Filial xidməti çox yavaşdır",
        "Kredit məbləği kifayət etmir",
        "İnternet banking işləmir"
    ]
    text_az = (base_texts * 17)[:100]  # 6 * 17 = 102 → truncate to 100

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

    loan_data = {
        'customer_id': list(range(1, 201)),
        'age': np.random.normal(40, 12, 200).astype(int),
        'income': np.random.gamma(2, 1000, 200),
        'employment': np.random.choice(['government', 'employed', 'self_employed', 'unemployed'], 200, p=[0.2, 0.5, 0.2, 0.1]),
        'credit_score': np.random.normal(650, 100, 200).astype(int),
        'loan_amount': np.random.gamma(2, 5000, 200),
        'debt_to_income': np.random.beta(2, 3, 200),
        'collateral_value': np.random.gamma(1.5, 8000, 200),
        'loan_to_value': np.random.beta(3, 2, 200),
        'tenure_months': np.random.randint(6, 120, 200),
        'region': np.random.choice(['Bakı', 'Gəncə', 'Sumqayıt', 'Mingəçevir', 'Şəki'], 200)
    }

    customer_data = {
        'customer_id': list(range(1, 301)),
        'age': np.random.normal(38, 15, 300).astype(int),
        'income': np.random.gamma(2, 1200, 300),
        'tenure_months': np.random.randint(1, 60, 300),
        'num_products': np.random.poisson(2, 300) + 1,
        'region': np.random.choice(['Bakı', 'Gəncə', 'Sumqayıt', 'Mingəçevir', 'Şəki'], 300),
        'last_transaction_days': np.random.randint(1, 90, 300),
        'digital_adoption': np.random.choice(['High', 'Medium', 'Low'], 300, p=[0.3, 0.5, 0.2])
    }

    return (
        pd.DataFrame(complaint_data),
        pd.DataFrame(loan_data).head(100),
        pd.DataFrame(customer_data).head(100)
    )


def create_pdf_report(data: Dict, title: str, language: str = 'az') -> bytes:
    """Generate PDF report"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=18, spaceAfter=30)
    story.append(Paragraph(title, title_style))
    story.append(Spacer(1, 12))
    
    # Add content based on data
    for section, content in data.items():
        story.append(Paragraph(section, styles['Heading2']))
        if isinstance(content, str):
            story.append(Paragraph(content, styles['Normal']))
        elif isinstance(content, dict):
            for key, value in content.items():
                story.append(Paragraph(f"{key}: {value}", styles['Normal']))
        story.append(Spacer(1, 12))
    
    doc.build(story)
    buffer.seek(0)
    return buffer.read()

def sidebar_navigation():
    """Create sidebar navigation"""
    st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    
    # Language selector
    language_options = {'Azərbaycan': 'az', 'English': 'en'}
    selected_language = st.sidebar.selectbox(
        t('language'),
        list(language_options.keys()),
        index=0 if st.session_state.language == 'az' else 1
    )
    st.session_state.language = language_options[selected_language]
    
    # API Key input
    st.sidebar.subheader(t('settings'))
    api_key = st.sidebar.text_input(
        t('api_key'),
        type="password",
        value=st.session_state.gemini_api_key,
        help="Enter your Google Gemini API key for enhanced AI features"
    )
    if api_key != st.session_state.gemini_api_key:
        st.session_state.gemini_api_key = api_key
        st.rerun()
    
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Navigation menu
    st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    menu_options = [
        t('home'),
        t('complaints'),
        t('credit_risk'),
        t('product_insights'),
        t('knowledge_search')
    ]
    
    selected_page = st.sidebar.radio("📍 Navigation", menu_options)
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    return selected_page

def home_page(gemini_api: GeminiAPI):
    """Home/Overview dashboard"""
    st.markdown(f'<h1 class="main-header">{t("title")}</h1>', unsafe_allow_html=True)
    
    # Generate sample data for overview
    complaint_df, loan_df, customer_df = generate_sample_data()
    
    # KPI Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label=t('total_complaints'),
            value=len(complaint_df),
            delta=f"+{np.random.randint(5, 15)} (7d)"
        )
    
    with col2:
        csat_score = np.random.uniform(3.2, 4.8)
        st.metric(
            label=t('csat_index'),
            value=f"{csat_score:.1f}/5.0",
            delta=f"+{np.random.uniform(0.1, 0.3):.1f}"
        )
    
    with col3:
        high_severity = len(complaint_df[complaint_df['severity'] == 'high'])
        st.metric(
            label=t('high_severity'),
            value=high_severity,
            delta=f"-{np.random.randint(1, 5)}"
        )
    
    with col4:
        avg_pd = loan_df['debt_to_income'].mean() * 0.3  # Simplified PD proxy
        st.metric(
            label=t('avg_pd'),
            value=f"{avg_pd:.1%}",
            delta=f"{np.random.uniform(-0.02, 0.02):+.1%}"
        )
    
    # Charts Row 1
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(t('category_distribution'))
        category_counts = complaint_df['category'].value_counts()
        fig = px.pie(
            values=category_counts.values,
            names=category_counts.index,
            title=t('category_distribution')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader(t('customer_segments'))
        analyzer = ProductInsightAnalyzer(gemini_api)
        customer_df_segmented = analyzer.segment_customers_advanced(customer_df)
        segment_counts = customer_df_segmented['segment'].value_counts()
        
        fig = px.bar(
            x=segment_counts.index,
            y=segment_counts.values,
            title=t('customer_segments')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Charts Row 2
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Complaint Trends")
        daily_complaints = complaint_df.groupby(complaint_df['date'].dt.date).size()
        fig = px.line(
            x=daily_complaints.index,
            y=daily_complaints.values,
            title="Daily Complaints"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader(t('risk_analysis'))
        risk_analyzer = CreditRiskAnalyzer(gemini_api)
        
        risk_levels = []
        for _, row in loan_df.head(50).iterrows():
            pd_result = risk_analyzer.calculate_pd_advanced(row.to_dict())
            if pd_result['pd'] > 0.2:
                risk_levels.append('High')
            elif pd_result['pd'] > 0.1:
                risk_levels.append('Medium')
            else:
                risk_levels.append('Low')
        
        risk_counts = pd.Series(risk_levels).value_counts()
        fig = px.bar(
            x=risk_counts.index,
            y=risk_counts.values,
            title=t('risk_analysis'),
            color=risk_counts.index,
            color_discrete_map={'High': 'red', 'Medium': 'orange', 'Low': 'green'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # AI Insights
    st.subheader(t('recommendations'))
    with st.expander("📊 AI-Generated Business Insights"):
        insights_prompt = f"""
        Bank performance özeti və tövsiyələr hazırla:
        
        Məlumatlar:
        - Ümumi şikayət: {len(complaint_df)}
        - CSAT: {csat_score:.1f}/5.0
        - Yüksək prioritet şikayətlər: {high_severity}
        - Orta PD: {avg_pd:.1%}
        - Ən çox şikayət kateqoriyası: {category_counts.index[0]}
        
        3 əsas tövsiyə ver.
        """
        
        insights = gemini_api.generate_response(insights_prompt, st.session_state.language)
        st.write(insights)

def complaints_page(gemini_api: GeminiAPI):
    """Complaints and Feedback Analysis Page"""
    st.header(t('complaints'))
    
    # Initialize complaint analyzer
    analyzer = ComplaintAnalyzer(gemini_api)
    
    # File upload section
    st.subheader(t('upload_data'))
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'xlsx', 'json'],
        help="Upload complaint data in CSV, Excel, or JSON format"
    )
    
    # Load sample or uploaded data
    if uploaded_file is not None:
        try:
            if uploaded_file.type == 'application/vnd.ms-excel':
                data = pd.read_excel(uploaded_file)
            elif uploaded_file.type == 'application/json':
                data = pd.read_json(uploaded_file)
            else:
                data = pd.read_csv(uploaded_file)
            st.session_state.complaint_data = data
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    else:
        # Use sample data
        complaint_df, _, _ = generate_sample_data()
        st.session_state.complaint_data = complaint_df
    
    data = st.session_state.complaint_data
    
    if data is not None:
        # Data overview
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(data))
        with col2:
            if 'severity' in data.columns:
                high_sev = len(data[data['severity'] == 'high'])
                st.metric("High Severity", high_sev)
        with col3:
            if 'status' in data.columns:
                open_cases = len(data[data['status'] == 'Open'])
                st.metric("Open Cases", open_cases)
        
        # Analysis tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            t('sentiment_analysis'), 
            "Topic Modeling", 
            "Response Generator", 
            "Similar Cases"
        ])
        
        with tab1:
            st.subheader(t('sentiment_analysis'))
            
            if 'text_az' in data.columns:
                # Analyze sentiments for sample of texts
                sample_texts = data['text_az'].dropna().head(20).tolist()
                
                sentiments = []
                severities = []
                
                for text in sample_texts:
                    result = analyzer.analyze_sentiment(text)
                    sentiments.append(result['sentiment'])
                    severities.append(result['severity'])
                
                # Sentiment distribution
                col1, col2 = st.columns(2)
                
                with col1:
                    sentiment_counts = pd.Series(sentiments).value_counts()
                    fig = px.pie(
                        values=sentiment_counts.values,
                        names=sentiment_counts.index,
                        title="Sentiment Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    severity_counts = pd.Series(severities).value_counts()
                    fig = px.bar(
                        x=severity_counts.index,
                        y=severity_counts.values,
                        title="Severity Distribution",
                        color=severity_counts.index
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Topic Discovery")
            
            if 'text_az' in data.columns:
                sample_texts = data['text_az'].dropna().head(50).tolist()
                
                if st.button("Run Topic Modeling"):
                    with st.spinner("Analyzing topics..."):
                        topic_result = analyzer.topic_modeling(sample_texts, n_topics=5)
                        
                        if topic_result['topics']:
                            for i, topic in enumerate(topic_result['topics']):
                                st.write(f"**Topic {i+1}:** {', '.join(topic['words'][:5])}")
        
        with tab3:
            st.subheader("AI Response Generator")
            
            sample_complaint = st.selectbox(
                "Select a complaint:",
                data['text_az'].dropna().head(10).tolist() if 'text_az' in data.columns else []
            )
            
            if sample_complaint and st.button("Generate Response"):
                with st.spinner(t('processing')):
                    response = analyzer.generate_response(sample_complaint, st.session_state.language)
                    
                    st.success(t('analysis_complete'))
                    st.write("**Generated Response:**")
                    st.write(response)
        
        with tab4:
            st.subheader("Find Similar Complaints")
            
            query = st.text_input("Enter search query:", placeholder="Describe the issue...")
            
            if query and 'text_az' in data.columns:
                similar_complaints = analyzer.similar_complaints(
                    query, data['text_az'].dropna().tolist(), top_k=5
                )
                
                for i, complaint in enumerate(similar_complaints):
                    st.write(f"**{i+1}. Similarity: {complaint['similarity']:.2f}**")
                    st.write(complaint['text'])
                    st.write("---")

def credit_risk_page(gemini_api: GeminiAPI):
    """Credit Risk and Expected Loss Analysis Page"""
    st.header(t('credit_risk'))
    
    # Initialize risk analyzer
    risk_analyzer = CreditRiskAnalyzer(gemini_api)
    
    # Load sample data
    _, loan_df, _ = generate_sample_data()
    
    # Risk calculation interface
    st.subheader(t('pd_calculation'))
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Customer Information:**")
        age = st.slider("Age", 18, 80, 35)
        income = st.number_input("Monthly Income (AZN)", 300, 10000, 1500)
        employment = st.selectbox("Employment", ['government', 'employed', 'self_employed', 'unemployed'])
        credit_score = st.slider("Credit Score", 300, 850, 650)
    
    with col2:
        st.write("**Loan Details:**")
        loan_amount = st.number_input("Loan Amount (AZN)", 1000, 100000, 25000)
        debt_to_income = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.3)
        collateral_value = st.number_input("Collateral Value (AZN)", 0, 200000, 30000)
        loan_to_value = st.slider("Loan-to-Value Ratio", 0.0, 1.0, 0.8)
    
    # Calculate risk metrics
    if st.button("Calculate Risk Metrics"):
        features = {
            'age': age,
            'income': income,
            'employment': employment,
            'credit_score': credit_score,
            'debt_to_income': debt_to_income,
            'loan_to_value': loan_to_value
        }
        
        # Calculate PD
        pd_result = risk_analyzer.calculate_pd_advanced(features)
        
        # Calculate LGD and EAD
        lgd = risk_analyzer.calculate_lgd(collateral_value, loan_amount)
        ead = risk_analyzer.calculate_ead(loan_amount)
        
        # Calculate Expected Loss
        el_result = risk_analyzer.expected_loss_calculation(pd_result['pd'], lgd, ead)
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Probability of Default (PD)", f"{pd_result['pd']:.2%}")
            risk_level = "High" if pd_result['pd'] > 0.2 else "Medium" if pd_result['pd'] > 0.1 else "Low"
            st.write(f"Risk Level: **{risk_level}**")
        
        with col2:
            st.metric("Loss Given Default (LGD)", f"{lgd:.2%}")
            st.metric("Exposure at Default (EAD)", f"{ead:,.0f} AZN")
        
        with col3:
            st.metric("Expected Loss (EL)", f"{el_result['expected_loss']:,.0f} AZN")
            st.metric("Unexpected Loss (UL)", f"{el_result['unexpected_loss']:,.0f} AZN")
        
        # PD Components breakdown
        st.subheader("PD Components Analysis")
        components = pd.DataFrame({
            'Component': list(pd_result['components'].keys()),
            'Impact': list(pd_result['components'].values())
        })
        
        fig = px.bar(
            components,
            x='Component',
            y='Impact',
            title="PD Risk Factors",
            color='Impact',
            color_continuous_scale='RdYlGn_r'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Scenario Analysis
        st.subheader(t('scenario_analysis'))
        scenarios = risk_analyzer.scenario_analysis(pd_result['pd'], lgd, ead)
        
        scenario_df = pd.DataFrame(scenarios).T
        scenario_df['Expected Loss (AZN)'] = scenario_df['expected_loss']
        
        fig = px.bar(
            x=scenario_df.index,
            y=scenario_df['Expected Loss (AZN)'],
            title="Expected Loss by Scenario",
            color=scenario_df['Expected Loss (AZN)'],
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # AI Explanation
        st.subheader("Decision Explanation")
        with st.expander("AI-Generated Risk Assessment"):
            explanation_prompt = f"""
            Kredit qərarı üçün ətraflı izahat:
            
            Müştəri: {age} yaş, {income} AZN gəlir, {employment} işi
            Kredit məbləği: {loan_amount} AZN
            PD: {pd_result['pd']:.2%}
            Gözlənilən itki: {el_result['expected_loss']:,.0f} AZN
            Risk səviyyəsi: {risk_level}
            
            Bu qərarın əsasları və tövsiyələr.
            """
            
            explanation = gemini_api.generate_response(explanation_prompt, st.session_state.language)
            st.write(explanation)

def product_insights_page(gemini_api: GeminiAPI):
    """Product Insights and Cross-Sell Analysis Page"""
    st.header(t('product_insights'))
    
    # Initialize analyzer
    analyzer = ProductInsightAnalyzer(gemini_api)
    
    # Load sample data
    _, _, customer_df = generate_sample_data()
    
    # Customer segmentation
    st.subheader(t('customer_segments'))
    segmented_df = analyzer.segment_customers_advanced(customer_df)
    
    # Segment overview
    col1, col2 = st.columns(2)
    
    with col1:
        segment_counts = segmented_df['segment'].value_counts()
        fig = px.pie(
            values=segment_counts.values,
            names=segment_counts.index,
            title="Customer Segments"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        value_tier_counts = segmented_df['value_tier'].value_counts()
        fig = px.bar(
            x=value_tier_counts.index,
            y=value_tier_counts.values,
            title="Value Tier Distribution",
            color=value_tier_counts.index
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Cross-sell analysis
    st.subheader(t('cross_sell'))
    
    # Select customer for analysis
    selected_customer = st.selectbox(
        "Select Customer ID:",
        segmented_df['customer_id'].head(20).tolist()
    )
    
    if selected_customer:
        customer_data = segmented_df[segmented_df['customer_id'] == selected_customer].iloc[0].to_dict()
        
        # Product propensity scores
        products = ['credit_card', 'personal_loan', 'mortgage', 'savings_account', 'investment_account', 'insurance']
        propensities = {}
        
        for product in products:
            prop_result = analyzer.calculate_propensity_advanced(customer_data, product)
            propensities[product] = prop_result['propensity']
        
        # Display propensity scores
        prop_df = pd.DataFrame({
            'Product': list(propensities.keys()),
            'Propensity': list(propensities.values())
        }).sort_values('Propensity', ascending=True)
        
        fig = px.bar(
            prop_df,
            x='Propensity',
            y='Product',
            orientation='h',
            title=f"Product Propensity for Customer {selected_customer}",
            color='Propensity',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Top recommendations
        st.write("**Top 3 Product Recommendations:**")
        top_products = prop_df.tail(3)
        for _, row in top_products.iterrows():
            st.write(f"• {row['Product'].title()}: {row['Propensity']:.1%} likelihood")
    
    # Strategy recommendations
    st.subheader("Marketing Strategy Recommendations")
    if st.button("Generate Strategy"):
        with st.spinner("Generating recommendations..."):
            strategy_prompt = f"""
            Bank məhsul strategiyası tövsiyələri:
            
            Müştəri seqmentləri:
            {dict(segmented_df['segment'].value_counts())}
            
            Dəyər səviyyələri:
            {dict(segmented_df['value_tier'].value_counts())}
            
            Hər seqment üçün uyğun məhsul strategiyası və marketinq yanaşması təklif edin.
            """
            
            strategy = gemini_api.generate_response(strategy_prompt, st.session_state.language)
            st.write(strategy)
    
    # Product performance metrics
    st.subheader(t('product_performance'))
    
    # Mock product performance data
    product_metrics = {
        'Product': ['Credit Cards', 'Personal Loans', 'Mortgages', 'Savings', 'Investments'],
        'Revenue (000 AZN)': [1250, 890, 2100, 450, 780],
        'Customers': [3200, 1500, 800, 5500, 1200],
        'Growth Rate': [0.12, 0.08, 0.15, 0.05, 0.22]
    }
    
    metrics_df = pd.DataFrame(product_metrics)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            metrics_df,
            x='Product',
            y='Revenue (000 AZN)',
            title="Product Revenue"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(
            metrics_df,
            x='Customers',
            y='Growth Rate',
            size='Revenue (000 AZN)',
            hover_name='Product',
            title="Growth vs Customer Base"
        )
        st.plotly_chart(fig, use_container_width=True)

def knowledge_search_page(gemini_api: GeminiAPI):
    """Knowledge Search (RAG) Page"""
    st.header(t('knowledge_search'))
    
    # Initialize knowledge base
    if 'knowledge_base' not in st.session_state or not isinstance(st.session_state.knowledge_base, KnowledgeBase):
        kb = KnowledgeBase(gemini_api)

    # Add sample documents
    sample_docs = [
        {
            'title': 'Kredit Kartı Qaydaları',
            'content': 'Kredit kartının istifadə qaydaları, komissiyalar və məhdudiyyətlər haqqında ətraflı məlumat. Aylıq komissiya 2 AZN, nağd pul çıxarma komissiyası 1.5%.',
        },
        {
            'title': 'Mobil Banking',
            'content': 'Mobil tətbiq vasitəsilə bank əməliyyatları, pul köçürmələri və hesabların idarə edilməsi. 24/7 xidmət, biometrik təhlükəsizlik.',
        },
        {
            'title': 'Kredit Şərtləri',
            'content': 'Fərdi kreditlərin şərtləri: minimum gəlir 500 AZN, maksimum məbləğ 50,000 AZN, müddət 5 ilə qədər, faiz dərəcəsi 12-18%.',
        }
    ]

    for doc in sample_docs:
        kb.add_document(doc['title'], doc['content'])

    st.session_state.knowledge_base = kb

    kb = st.session_state.knowledge_base

    
    # Document management
    st.subheader("Knowledge Base Management")
    
    with st.expander("Add New Document"):
        title = st.text_input("Document Title")
        content = st.text_area("Document Content", height=150)
        
        if st.button("Add Document") and title and content:
            kb.add_document(title, content)
            st.success("Document added successfully!")
            st.rerun()
    
    # Current documents
    st.write(f"**Current Documents:** {len(kb.documents)}")
    
    # Search interface
    st.subheader(t('search_knowledge'))
    
    query = st.text_input(
        t('ask_question'),
        placeholder="Kredit kartı komissiyası nə qədərdir?" if st.session_state.language == 'az' 
                   else "What are the credit card fees?"
    )
    
    if query:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Generate answer
            with st.spinner("Searching and generating answer..."):
                answer = kb.generate_answer(query, st.session_state.language)
                st.write("**Answer:**")
                st.write(answer)
        
        with col2:
            # Show relevant documents
            relevant_docs = kb.search(query, top_k=3)
            st.write("**Relevant Documents:**")
            
            for doc in relevant_docs:
                with st.expander(f"{doc['title']} (Score: {doc['similarity']:.2f})"):
                    st.write(doc['content'][:300] + "...")

def main():
    # API açarını əvvəlcə secrets-dən yükləməyə cəhd et
    if not st.session_state.gemini_api_key:
        try:
            st.session_state.gemini_api_key = st.secrets["GEMINI_API_KEY"]
            st.toast("Gemini API açarı secrets-dən yükləndi", icon="🔐")
        except Exception:
            st.warning("Gemini API açarı tapılmadı. Zəhmət olmasa əl ilə daxil edin.")


    # Initialize Gemini API wrapper
    gemini_api = GeminiAPI(api_key=st.session_state.gemini_api_key)


    # Navigation handler
    selected_page = sidebar_navigation()
    
    
    if selected_page == t('home'):
        home_page(gemini_api)
    elif selected_page == t('complaints'):
        complaints_page(gemini_api)
    elif selected_page == t('credit_risk'):
        credit_risk_page(gemini_api)
    elif selected_page == t('product_insights'):
        product_insights_page(gemini_api)
    elif selected_page == t('knowledge_search'):
        knowledge_search_page(gemini_api)




# Run the app
if __name__ == "__main__":
    main()
