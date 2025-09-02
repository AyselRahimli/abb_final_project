# Bank360 - Düzəldilmiş və Azərbaycan dilində
# Bu fayl əsas problemlərin həllini ehtiva edir

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

# Təhlükəsiz import-lar
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
    st.error("scikit-learn quraşdırılmayıb. Çalışdırın: pip install scikit-learn")

try:
    import scipy.stats as stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    st.error("scipy quraşdırılmayıb. Çalışdırın: pip install scipy")

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    st.warning("Google Gemini API mövcud deyil. Quraşdırın: pip install google-generativeai")

# Streamlit səhifəni konfiqurasiya et
st.set_page_config(
    page_title="Bank360 Analitika",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    """Session state-i düzgün default-larla başlat"""
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
    """Funksiyaları təhlükəsiz şəkildə icra et"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        st.error(f"{func.__name__} funksiyasında xəta: {str(e)}")
        return None

@st.cache_data
def generate_sample_data_fixed():
    """Nümunə məlumatları düzgün xəta idarəetməsi və ardıcıllıqla yarad"""
    np.random.seed(42)  # Təkrarlanabilirlik üçün
    
    # Şikayət mətnləri - dəqiq 100 giriş
    base_complaint_texts = [
        "Mobil tətbiqdə problem var, giriş edə bilmirəm",
        "ATM-dən pul çıxarmaq mümkün olmur", 
        "Kart komissiyası çox yüksəkdir",
        "Filial xidməti çox yavaşdır",
        "Kredit məbləği kifayət etmir",
        "İnternet banking işləmir",
        "Hesabımdan səhv məbləğ silinib",
        "Telefon zənglər çox tez-tez gəlir",
        "Online ödəniş sistemi yavaş işləyir",
        "Kart bloklanıb, səbəbi aydın deyil"
    ]
    
    # Dəqiq 100 şikayət mətni yarad
    text_az = [np.random.choice(base_complaint_texts) for _ in range(100)]
    
    # Ardıcıl tiplərlə şikayət məlumatları
    complaint_data = {
        'id': list(range(1, 101)),
        'tarix': pd.date_range(start='2024-01-01', periods=100, freq='D'),
        'musteri_id': np.random.randint(1000, 9999, 100),
        'kanal': np.random.choice(['Mobil Tətbiq', 'Filial', 'Zəng Mərkəzi', 'Veb Sayt'], 100),
        'kateqoriya': np.random.choice(['Kart', 'ATM', 'Mobil', 'Komissiya', 'Filial', 'Kredit'], 100),
        'metn_az': text_az,
        'ciddilik': np.random.choice(['aşağı', 'orta', 'yüksək'], 100, p=[0.4, 0.4, 0.2]),
        'status': np.random.choice(['Açıq', 'Prosesdə', 'Bağlı'], 100, p=[0.2, 0.3, 0.5]),
        'region': np.random.choice(['Bakı', 'Gəncə', 'Sumqayıt', 'Mingəçevir', 'Şəki'], 100)
    }
    
    # Düzgün məlumat tipləri ilə kredit məlumatları
    loan_data = {
        'musteri_id': list(range(1, 201)),
        'yas': np.clip(np.random.normal(40, 12, 200).astype(int), 18, 80),
        'gelir': np.clip(np.random.gamma(2, 1000, 200), 300, 15000),
        'isci_veziyyeti': np.random.choice(['dövlət', 'işçi', 'sərbəst_işçi', 'işsiz'], 200, p=[0.2, 0.5, 0.2, 0.1]),
        'kredit_reytingi': np.clip(np.random.normal(650, 100, 200).astype(int), 300, 850),
        'kredit_meblegi': np.clip(np.random.gamma(2, 5000, 200), 1000, 100000),
        'borc_gelir_nisbeti': np.clip(np.random.beta(2, 3, 200), 0.05, 0.95),
        'teminat_deyeri': np.random.gamma(1.5, 8000, 200),
        'kredit_teminat_nisbeti': np.clip(np.random.beta(3, 2, 200), 0.1, 0.95),
        'muddet_ay': np.random.randint(6, 120, 200),
        'region': np.random.choice(['Bakı', 'Gəncə', 'Sumqayıt', 'Mingəçevir', 'Şəki'], 200)
    }
    
    # Müştəri məlumatları
    customer_data = {
        'musteri_id': list(range(1, 301)),
        'yas': np.clip(np.random.normal(38, 15, 300).astype(int), 18, 80),
        'gelir': np.clip(np.random.gamma(2, 1200, 300), 300, 10000),
        'muddet_ay': np.random.randint(1, 60, 300),
        'mehsul_sayi': np.clip(np.random.poisson(2, 300) + 1, 1, 6),
        'region': np.random.choice(['Bakı', 'Gəncə', 'Sumqayıt', 'Mingəçevir', 'Şəki'], 300),
        'son_tranzaksiya_gunleri': np.random.randint(1, 90, 300),
        'reqemsal_qebul': np.random.choice(['Yüksək', 'Orta', 'Aşağı'], 300, p=[0.3, 0.5, 0.2])
    }
    
    return (
        pd.DataFrame(complaint_data),
        pd.DataFrame(loan_data).head(100),
        pd.DataFrame(customer_data).head(100)
    )

class ImprovedGeminiAPI:
    """Təkmilləşdirilmiş Gemini API wrapper-i"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.model = None
        self.initialized = False
        
        if api_key and GEMINI_AVAILABLE:
            try:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel('gemini-2.5-flash')
                self.initialized = True
                st.success("Gemini API uğurla başladıldı!")
            except Exception as e:
                st.error(f"Gemini API başladılmasında xəta: {str(e)}")
                self.initialized = False
        elif not GEMINI_AVAILABLE:
            st.info("Gemini API mövcud deyil - mock cavablar istifadə edilir")
    
    def generate_response(self, prompt: str, language: str = 'az', max_retries: int = 3) -> str:
        """Təkrar cəhd məntiqi və düzgün xəta idarəetməsi ilə cavab yarad"""
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
                    raise Exception("API-dan boş cavab")
                    
            except Exception as e:
                if attempt == max_retries - 1:
                    st.warning(f"API çağırışı {max_retries} cəhddən sonra uğursuz: {str(e)}")
                    return self._mock_response(prompt, language)
                continue
        
        return self._mock_response(prompt, language)
    
    def _mock_response(self, prompt: str, language: str = 'az') -> str:
        """ABB Bank məlumatları ilə təkmilləşdirilmiş mock cavab sistemi"""
        prompt_lower = prompt.lower()
        
        # Şikayət cavabları
        if any(word in prompt_lower for word in ['complaint', 'şikayət', 'problem']):
            return """Hörmətli müştəri,

ABB Bank olaraq şikayətinizi qəbul edirik və dərhal araşdırmaya başlayırıq. Bizim üçün müştəri məmnuniyyəti prioritetdir.

Əlaqə məlumatlarımız:
• Zəng Mərkəzi: 937
• E-poçt: info@abb-bank.az
• 24/7 online xidmət

2-3 iş günü ərzində sizinlə əlaqə saxlayacağıq. Səbiriniz üçün təşəkkür edirik.

Hörmətlə,
ABB Bank Müştəri Xidmətləri"""
        
        # Kredit analiz cavabları
        elif any(word in prompt_lower for word in ['credit', 'kredit', 'loan', 'risk']):
            return """ABB Bank kredit analizi nəticəsində:

Müştərinin ödəmə qabiliyyəti orta səviyyədə qiymətləndirilir. 

Əlavə məlumatlar:
• Zəng Mərkəzi: 937
• E-poçt: info@abb-bank.az
• Kredit departamenti ilə əlavə məsləhətləşmə tövsiyə olunur

Risk idarəetməsi bölməsi ilə əlaqə saxlayın."""
        
        # Strategiya cavabları
        elif any(word in prompt_lower for word in ['strategy', 'strategiya', 'recommend', 'tövsiyə']):
            return """ABB Bank marketinq strategiyası tövsiyələri:

1) Rəqəmsal platformaları inkişaf etdirin
2) Müştəri seqmentlərinə uyğun məhsullar təklif edin  
3) Müştəri məmnuniyyətini artırmaq üçün xidmət keyfiyyətini yaxşılaşdırın
4) Çarpaz satış imkanlarından istifadə edin

Əlavə məlumat üçün:
• Zəng Mərkəzi: 937
• E-poçt: info@abb-bank.az"""
        
        # Ümumi cavab
        else:
            return """ABB Bank olaraq sorğunuz əsasında analiz aparılmış və müvafiq tövsiyələr hazırlanmışdır.

Əlaqə məlumatlarımız:
• Zəng Mərkəzi: 937  
• E-poçt: info@abb-bank.az

Əlavə məlumat üçün müvafiq departamentlə əlaqə saxlayın."""

def validate_uploaded_file(uploaded_file) -> Optional[pd.DataFrame]:
    """Yüklənən faylları təhlükəsiz şəkildə yoxla və emal et"""
    if uploaded_file is None:
        return None
    
    try:
        file_type = uploaded_file.type
        file_size = uploaded_file.size
        
        # Fayl ölçüsünü yoxla (maksimum 50MB)
        if file_size > 50 * 1024 * 1024:
            st.error("Fayl ölçüsü çox böyükdür. Maksimum 50MB icazə verilir.")
            return None
        
        # Fayl tipinə görə emal et
        if file_type == 'text/csv':
            df = pd.read_csv(uploaded_file, encoding='utf-8')
        elif file_type in ['application/vnd.ms-excel', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet']:
            df = pd.read_excel(uploaded_file)
        elif file_type == 'application/json':
            df = pd.read_json(uploaded_file)
        else:
            st.error(f"Dəstəklənməyən fayl tipi: {file_type}")
            return None
        
        # Əsas yoxlama
        if df.empty:
            st.error("Yüklənən fayl boşdur.")
            return None
        
        if len(df) > 10000:
            st.warning("Böyük fayl aşkar edildi. İlk 10,000 sətir emal edilir.")
            df = df.head(10000)
        
        st.success(f"Fayl uğurla yükləndi! {len(df)} sətir, {len(df.columns)} sütun")
        return df
        
    except Exception as e:
        st.error(f"Fayl emalında xəta: {str(e)}")
        return None

@st.cache_data
def safe_sentiment_analysis(texts: List[str]) -> List[Dict[str, Any]]:
    """Keşləmə ilə təhlükəsiz sentiment analizi"""
    if not texts:
        return []
    
    results = []
    positive_words = ['yaxşı', 'əla', 'mükəmməl', 'razıyam', 'təşəkkür', 'gözəl', 'super']
    negative_words = ['pis', 'səhv', 'problem', 'şikayət', 'narazıyam', 'yavaş', 'dəhşətli', 'çox_pis']
    severity_words = ['təcili', 'dərhal', 'mütləq', 'vacib', 'ciddi', 'mühüm']
    
    for text in texts:
        try:
            text_lower = str(text).lower()
            
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)
            severity_count = sum(1 for word in severity_words if word in text_lower)
            
            if pos_count > neg_count:
                sentiment = 'müsbət'
                score = min(0.9, 0.6 + (pos_count * 0.1))
            elif neg_count > pos_count:
                sentiment = 'mənfi'
                score = max(0.1, 0.4 - (neg_count * 0.1))
            else:
                sentiment = 'neytral'
                score = 0.5
            
            if severity_count >= 2 or neg_count >= 3:
                severity = 'yüksək'
            elif severity_count == 1 or neg_count >= 2:
                severity = 'orta'
            else:
                severity = 'aşağı'
            
            results.append({
                'sentiment': sentiment,
                'score': score,
                'severity': severity,
                'confidence': min(0.95, 0.7 + (pos_count + neg_count) * 0.05)
            })
        except Exception as e:
            # Uğursuz analiz üçün neytral qaytır
            results.append({
                'sentiment': 'neytral',
                'score': 0.5,
                'severity': 'aşağı',
                'confidence': 0.5
            })
    
    return results

def improved_sidebar_navigation():
    """Təkmilləşdirilmiş yan panel"""
    st.sidebar.markdown("### 🏦 Bank360 Analitika")
    
    # Dil seçici
    language_options = {'Azərbaycan': 'az', 'English': 'en'}
    current_lang_key = 'Azərbaycan' if st.session_state.language == 'az' else 'English'
    
    selected_language = st.sidebar.selectbox(
        "Dil / Language",
        list(language_options.keys()),
        index=list(language_options.keys()).index(current_lang_key)
    )
    st.session_state.language = language_options[selected_language]
    
    # API Key idarəetməsi
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Tənzimləmələr")
    
    api_key = st.sidebar.text_input(
        "Gemini API Açarı",
        type="password",
        value=st.session_state.gemini_api_key,
        help="AI xüsusiyyətləri üçün Google Gemini API açarınızı daxil edin",
        placeholder="AIza..."
    )
    
    if api_key != st.session_state.gemini_api_key:
        st.session_state.gemini_api_key = api_key
        if api_key:
            st.sidebar.success("API açarı yeniləndi!")
    
    # Naviqasiya menyusu
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Naviqasiya")
    
    pages = ['Ana Səhifə', 'Şikayətlər', 'Kredit Riski', 'Məhsul Məlumatları', 'Bilik Axtarışı']
    
    selected_page = st.sidebar.radio(
        "Səhifə Seçin",
        pages
    )
    
    # Sistem statusu
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Sistem Statusu")
    
    status_items = [
        ("Gemini API", "✅" if GEMINI_AVAILABLE and st.session_state.gemini_api_key else "❌"),
        ("scikit-learn", "✅" if SKLEARN_AVAILABLE else "❌"),
        ("scipy", "✅" if SCIPY_AVAILABLE else "❌")
    ]
    
    for item, status in status_items:
        st.sidebar.text(f"{item}: {status}")
    
    return selected_page

def main():
    """Təkmilləşdirilmiş xəta idarəetməsi ilə əsas tətbiq"""
    # Session state-i başlat
    initialize_session_state()
    
    # Secrets-dən API açarını yükləməyə çalış
    if not st.session_state.gemini_api_key and not st.session_state.initialized:
        try:
            st.session_state.gemini_api_key = st.secrets.get("GEMINI_API_KEY", "")
            if st.session_state.gemini_api_key:
                st.toast("API açarı secrets-dən yükləndi", icon="🔑")
        except:
            pass  # Secrets faylı yoxdur və ya açar tapılmadı
        
        st.session_state.initialized = True
    
    # API-ni başlat
    gemini_api = safe_execute(ImprovedGeminiAPI, st.session_state.gemini_api_key)
    if not gemini_api:
        gemini_api = ImprovedGeminiAPI()  # Mock rejimə keç
    
    # Naviqasiya
    try:
        selected_page = improved_sidebar_navigation()
        
        # Müvafiq səhifəyə yönləndir
        if selected_page == 'Ana Səhifə':
            home_page_improved(gemini_api)
        elif selected_page == 'Şikayətlər':
            complaints_page_improved(gemini_api)
        elif selected_page == 'Kredit Riski':
            credit_risk_page_improved(gemini_api)
        elif selected_page == 'Məhsul Məlumatları':
            product_insights_page_improved(gemini_api)
        elif selected_page == 'Bilik Axtarışı':
            knowledge_search_page_improved(gemini_api)
            
    except Exception as e:
        st.error(f"Naviqasiya xətası: {str(e)}")
        st.info("Zəhmət olmasa səhifəni yeniləyin və təkrar cəhd edin.")

def home_page_improved(gemini_api):
    """Təkmilləşdirilmiş ana səhifə"""
    st.title("🏦 Bank360 Analitika İdarə Paneli")
    st.markdown("---")
    
    # Məlumatları təhlükəsiz yüklə
    try:
        complaint_df, loan_df, customer_df = generate_sample_data_fixed()
    except Exception as e:
        st.error(f"Nümunə məlumatların yüklənməsində xəta: {str(e)}")
        return
    
    # KPI sətiri
    col1, col2, col3, col4 = st.columns(4)
    
    try:
        with col1:
            st.metric("Ümumi Şikayətlər", len(complaint_df), delta=f"+{np.random.randint(5, 15)}")
        
        with col2:
            csat_score = np.random.uniform(3.8, 4.5)
            st.metric("CSAT Balı", f"{csat_score:.1f}/5.0", delta=f"+{np.random.uniform(0.1, 0.3):.1f}")
        
        with col3:
            high_severity = len(complaint_df[complaint_df['ciddilik'] == 'yüksək']) if 'ciddilik' in complaint_df.columns else 0
            st.metric("Yüksək Ciddiyyət", high_severity, delta=f"-{np.random.randint(1, 3)}")
        
        with col4:
            avg_pd = loan_df['borc_gelir_nisbeti'].mean() * 0.25 if 'borc_gelir_nisbeti' in loan_df.columns else 0.15
            st.metric("Orta PD", f"{avg_pd:.1%}", delta=f"{np.random.uniform(-0.01, 0.01):+.1%}")
    
    except Exception as e:
        st.error(f"Metriklər göstərilməsində xəta: {str(e)}")
    
    st.markdown("---")
    
    # Qrafiklər sətiri
    col1, col2 = st.columns(2)
    
    with col1:
        try:
            if 'kateqoriya' in complaint_df.columns:
                category_counts = complaint_df['kateqoriya'].value_counts()
                fig = px.pie(
                    values=category_counts.values,
                    names=category_counts.index,
                    title="Şikayət Kateqoriyaları"
                )
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Kateqoriya qrafikinin yaradılmasında xəta: {str(e)}")
    
    with col2:
        try:
            if 'tarix' in complaint_df.columns:
                daily_complaints = complaint_df.groupby(complaint_df['tarix'].dt.date).size()
                fig = px.line(
                    x=daily_complaints.index,
                    y=daily_complaints.values,
                    title="Gündəlik Şikayət Tendensiyaları"
                )
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Tendensiya qrafikinin yaradılmasında xəta: {str(e)}")
    
    # AI İntellektual təhlil bölməsi
    st.markdown("---")
    st.subheader("🤖 AI tərəfindən yaradılan təhillər")
    
    if st.button("Biznes Təhlilləri Yarad"):
        with st.spinner("Məlumatlar təhlil edilir və nəticələr yaradılır..."):
            insights_prompt = f"""
            ABB Bank-ın performans məlumatlarını təhlil edin və 3 əsas biznes nəticəsi verin:
            
            ABB Bank məlumatları:
            - Bank adı: ABB Bank  
            - Zəng Mərkəzi: 937
            - E-poçt: info@abb-bank.az
            
            Məlumat Xülasəsi:
            - Ümumi şikayətlər: {len(complaint_df)}
            - Yüksək ciddiyyət şikayətləri: {len(complaint_df[complaint_df['ciddilik'] == 'yüksək']) if 'ciddilik' in complaint_df.columns else 0}
            - Orta risk səviyyəsi: {avg_pd:.1%}
            - Ən çox rastlanan şikayət kateqoriyası: {complaint_df['kateqoriya'].value_counts().index[0] if 'kateqoriya' in complaint_df.columns else 'N/A'}
            
            ABB Bank üçün təkmilləşdirmə tövsiyələrinə diqqət yetirin.
            """
            
            insights = gemini_api.generate_response(insights_prompt, st.session_state.language)
            st.write(insights)

def complaints_page_improved(gemini_api):
    """Təkmilləşdirilmiş şikayətlər səhifəsi"""
    st.title("Şikayətlər və Rəy Təhlili")
    st.markdown("---")
    
    # Fayl yükləmə bölməsi
    st.subheader("Məlumat Yükləyin")
    uploaded_file = st.file_uploader(
        "CSV, Excel və ya JSON fayl seçin",
        type=['csv', 'xlsx', 'json'],
        help="Şikayət məlumatlarını təhlil üçün yükləyin"
    )
    
    # Məlumatları yüklə
    if uploaded_file is not None:
        data = validate_uploaded_file(uploaded_file)
        if data is not None:
            st.session_state.complaint_data = data
    else:
        # Nümunə məlumatlar istifadə et
        try:
            complaint_df, _, _ = generate_sample_data_fixed()
            st.session_state.complaint_data = complaint_df
            st.info("Nümunə məlumatlar istifadə edilir. Həqiqi şikayətləri təhlil etmək üçün öz faylınızı yükləyin.")
        except Exception as e:
            st.error(f"Nümunə məlumatların yüklənməsində xəta: {str(e)}")
            return
    
    data = st.session_state.complaint_data
    
    if data is None or data.empty:
        st.warning("Məlumat yoxdur. Zəhmət olmasa düzgün fayl yükləyin.")
        return
    
    # Məlumat baxışı
    col1, col2, col3, col4 = st.columns(4)
    
    try:
        with col1:
            st.metric("Ümumi Qeydlər", len(data))
        
        with col2:
            high_sev = len(data[data['ciddilik'] == 'yüksək']) if 'ciddilik' in data.columns else 0
            st.metric("Yüksək Ciddiyyət", high_sev)
        
        with col3:
            open_cases = len(data[data['status'] == 'Açıq']) if 'status' in data.columns else 0
            st.metric("Açıq İşlər", open_cases)
        
        with col4:
            avg_days = np.random.randint(2, 7)  # Mock həll vaxtı
            st.metric("Orta Həll (gün)", avg_days)
    
    except Exception as e:
        st.error(f"Metriklər hesablanmasında xəta: {str(e)}")
    
    # Təhlil tab-ları
    tab1, tab2, tab3, tab4 = st.tabs([
        "Sentiment Təhlili", 
        "Kateqoriya Təhlili", 
        "Cavab Yaradıcısı", 
        "Tendensiyalar və Nümunələr"
    ])
    
    with tab1:
        st.subheader("Sentiment Təhlili")
        
        if 'metn_az' in data.columns:
            try:
                sample_size = min(50, len(data))
                sample_texts = data['metn_az'].dropna().head(sample_size).tolist()
                
                if st.button("Sentimentləri Təhlil Et", key="sentiment_btn"):
                    with st.spinner("Sentimentlər təhlil edilir..."):
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
                                    title="Sentiment Paylanması"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                severity_counts = pd.Series(severity_labels).value_counts()
                                colors = {'yüksək': 'red', 'orta': 'orange', 'aşağı': 'green'}
                                fig = px.bar(
                                    x=severity_counts.index,
                                    y=severity_counts.values,
                                    title="Ciddiyyət Paylanması",
                                    color=severity_counts.index,
                                    color_discrete_map=colors
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("Sentimentlər təhlil edilə bilmədi")
            except Exception as e:
                st.error(f"Sentiment təhlilində xəta: {str(e)}")
        else:
            st.warning("Sentiment təhlili üçün mətn sütunu tapılmadı")
    
    with tab2:
        st.subheader("Kateqoriya Təhlili")
        
        if 'kateqoriya' in data.columns:
            try:
                category_counts = data['kateqoriya'].value_counts()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.bar(
                        x=category_counts.values,
                        y=category_counts.index,
                        orientation='h',
                        title="Kateqoriyalara görə Şikayətlər"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    if 'ciddilik' in data.columns:
                        severity_by_cat = pd.crosstab(data['kateqoriya'], data['ciddilik'])
                        fig = px.bar(
                            severity_by_cat,
                            title="Kateqoriyalara görə Ciddiyyət Paylanması",
                            barmode='stack'
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            except Exception as e:
                st.error(f"Kateqoriya təhlilində xəta: {str(e)}")
        else:
            st.warning("Kateqoriya sütunu tapılmadı")
    
    with tab3:
        st.subheader("AI Cavab Yaradıcısı")
        
        if 'metn_az' in data.columns:
            complaint_options = data['metn_az'].dropna().head(10).tolist()
            
            if complaint_options:
                selected_complaint = st.selectbox(
                    "Cavab yaratmaq üçün şikayət seçin:",
                    complaint_options,
                    key="response_complaint"
                )
                
                if st.button("Peşəkar Cavab Yarat", key="generate_response_btn"):
                    with st.spinner("Cavab yaradılır..."):
                        try:
                            response = gemini_api.generate_response(
                                f"""ABB Bank olaraq bu müştəri şikayətinə peşəkar cavab yaradın. 
                                
                                Bank məlumatları:
                                - Bank adı: ABB Bank
                                - Zəng Mərkəzi: 937
                                - E-poçt: info@abb-bank.az
                                
                                Şikayət: {selected_complaint}
                                
                                Cavab hörmətli, peşəkar və həlledici olsun. Bank əlaqə məlumatlarını daxil edin.""",
                                st.session_state.language
                            )
                            
                            st.success("Cavab uğurla yaradıldı!")
                            st.write("**Yaradılan Cavab:**")
                            st.write(response)
                            
                        except Exception as e:
                            st.error(f"Cavab yaratmaqda xəta: {str(e)}")
            else:
                st.warning("Cavab yaratmaq üçün şikayət mövcud deyil")
        else:
            st.warning("Mətn məlumatları mövcud deyil")
    
    with tab4:
        st.subheader("Tendensiyalar və Nümunələr")
        
        try:
            if 'tarix' in data.columns:
                # Gündəlik şikayət tendensiyaları
                data['tarix'] = pd.to_datetime(data['tarix'])
                daily_complaints = data.groupby(data['tarix'].dt.date).size()
                
                fig = px.line(
                    x=daily_complaints.index,
                    y=daily_complaints.values,
                    title="Gündəlik Şikayət Həcmi"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Həftəlik nümunələr
                data['heftenin_gunu'] = data['tarix'].dt.day_name()
                weekly_pattern = data['heftenin_gunu'].value_counts()
                
                fig = px.bar(
                    x=weekly_pattern.index,
                    y=weekly_pattern.values,
                    title="Həftənin Günlərinə görə Şikayətlər"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.info("Tarix sütunu tapılmadı. Zaman tendensiyaları göstərilə bilməz.")
                
        except Exception as e:
            st.error(f"Tendensiya təhlilində xəta: {str(e)}")

def credit_risk_page_improved(gemini_api):
    """Təkmilləşdirilmiş kredit risk səhifəsi"""
    st.title("Kredit Riski və Gözlənilən İtki Təhlili")
    st.markdown("---")
    
    # Giriş bölməsi
    st.subheader("Müştəri Risk Qiymətləndirməsi")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Şəxsi Məlumatlar**")
        age = st.slider("Yaş", 18, 80, 35, key="risk_age")
        income = st.number_input("Aylıq Gəlir (AZN)", 300.0, 15000.0, 1500.0, key="risk_income")
        employment = st.selectbox("İş Vəziyyəti", 
                                ['dövlət', 'işçi', 'sərbəst_işçi', 'işsiz'], 
                                key="risk_employment")
        credit_score = st.slider("Kredit Reytinqi", 300, 850, 650, key="risk_credit_score")
    
    with col2:
        st.write("**Kredit Məlumatları**")
        loan_amount = st.number_input("Kredit Məbləği (AZN)", 1000.0, 100000.0, 25000.0, key="risk_loan_amount")
        debt_to_income = st.slider("Borc-Gəlir Nisbəti", 0.0, 1.0, 0.3, key="risk_dti")
        collateral_value = st.number_input("Təminat Dəyəri (AZN)", 0.0, 200000.0, 30000.0, key="risk_collateral")
        loan_to_value = st.slider("Kredit-Təminat Nisbəti", 0.0, 1.0, 0.8, key="risk_ltv")
    
    # Risk hesablama düyməsi
    if st.button("Risk Metriklər Hesabla", key="calc_risk_btn"):
        try:
            # Sadələşdirilmiş model istifadə edərək PD hesabla
            pd_score = calculate_pd_simple(age, income, employment, credit_score, debt_to_income, loan_to_value)
            
            # LGD hesabla
            if collateral_value >= loan_amount:
                lgd = 0.2  # Kifayət təminatla aşağı LGD
            else:
                collateral_ratio = collateral_value / loan_amount if loan_amount > 0 else 0
                lgd = max(0.3, 0.8 - (collateral_ratio * 0.5))
            
            # EAD hesabla (sadələşdirilmiş)
            ead = loan_amount * 0.85
            
            # Gözlənilən İtki hesabla
            expected_loss = pd_score * lgd * ead
            unexpected_loss = ead * lgd * np.sqrt(pd_score * (1 - pd_score))
            
            # Nəticələri göstər
            st.markdown("---")
            st.subheader("Risk Qiymətləndirmə Nəticələri")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                risk_level = "Yüksək" if pd_score > 0.2 else "Orta" if pd_score > 0.1 else "Aşağı"
                risk_color = "red" if risk_level == "Yüksək" else "orange" if risk_level == "Orta" else "green"
                
                st.metric("Defolt Ehtimalı (PD)", f"{pd_score:.2%}")
                st.markdown(f"**Risk Səviyyəsi:** <span style='color:{risk_color}'>{risk_level}</span>", 
                          unsafe_allow_html=True)
            
            with col2:
                st.metric("Defoltda İtki (LGD)", f"{lgd:.2%}")
                st.metric("Defoltda Məruz Qalma (EAD)", f"{ead:,.0f} AZN")
            
            with col3:
                st.metric("Gözlənilən İtki (EL)", f"{expected_loss:,.0f} AZN")
                st.metric("Gözlənilməz İtki (UL)", f"{unexpected_loss:,.0f} AZN")
            
            # Risk izahı
            st.subheader("Risk Qiymətləndirmə İzahı")
            with st.expander("Ətraflı Təhlili Göstər"):
                explanation_prompt = f"""
                ABB Bank üçün ətraflı kredit risk qiymətləndirmə izahı verin:
                
                ABB Bank məlumatları:
                - Bank adı: ABB Bank
                - Zəng Mərkəzi: 937  
                - E-poçt: info@abb-bank.az
                
                Müştəri Profili:
                - Yaş: {age} il
                - Aylıq Gəlir: {income:,.0f} AZN
                - İş Vəziyyəti: {employment}
                - Kredit Reytinqi: {credit_score}
                
                Kredit Təfərrüatları:
                - Məbləğ: {loan_amount:,.0f} AZN
                - Borc-Gəlir Nisbəti: {debt_to_income:.1%}
                - Kredit-Təminat Nisbəti: {loan_to_value:.1%}
                
                Risk Metriklər:
                - PD: {pd_score:.2%}
                - Gözlənilən İtki: {expected_loss:,.0f} AZN
                - Risk Səviyyəsi: {risk_level}
                
                ABB Bank-ın risk idarəetmə siyasətini nəzərə alaraq əsas qərar verin və izah edin qərarın səbəbini.
                """
                
                try:
                    explanation = gemini_api.generate_response(explanation_prompt, st.session_state.language)
                    st.write(explanation)
                except Exception as e:
                    st.error(f"İzah yaradılmasında xəta: {str(e)}")
            
        except Exception as e:
            st.error(f"Risk hesablanmasında xəta: {str(e)}")

def calculate_pd_simple(age, income, employment, credit_score, debt_to_income, loan_to_value):
    """Sadələşdirilmiş PD hesablaması"""
    base_pd = 0.15
    
    # Yaş faktoru
    if age < 25 or age > 65:
        age_factor = 0.03
    elif 35 <= age <= 50:
        age_factor = -0.02
    else:
        age_factor = 0
    
    # Gəlir faktoru
    income_factor = -0.00002 * income if income > 0 else 0.1
    
    # İş faktoru
    emp_factors = {'dövlət': -0.03, 'işçi': -0.01, 'sərbəst_işçi': 0.02, 'işsiz': 0.15}
    employment_factor = emp_factors.get(employment, 0)
    
    # Kredit reytinq faktoru
    credit_factor = -0.0002 * (credit_score - 600)
    
    # DTI faktoru
    dti_factor = debt_to_income * 0.1
    
    # LTV faktoru
    ltv_factor = loan_to_value * 0.05
    
    pd = base_pd + age_factor + income_factor + employment_factor + credit_factor + dti_factor + ltv_factor
    return max(0.01, min(0.95, pd))

def product_insights_page_improved(gemini_api):
    """Təkmilləşdirilmiş məhsul təhlilləri səhifəsi"""
    st.title("Məhsul Təhlilləri və Çarpaz Satış Analizi")
    st.markdown("---")
    
    # Fayl yükləmə bölməsi
    st.subheader("Məlumat Yükləyin")
    uploaded_file = st.file_uploader(
        "Müştəri məlumatları faylını seçin (CSV, Excel, JSON)",
        type=['csv', 'xlsx', 'json'],
        help="Müştəri məlumatlarını təhlil üçün yükləyin. Gözlənilən sütunlar: müştəri_id, yaş, gəlir, məhsul_sayı, region, vb."
    )
    
    # Məlumatları yüklə və emal et
    if uploaded_file is not None:
        customer_df = validate_uploaded_file(uploaded_file)
        if customer_df is not None:
            st.session_state.customer_data = customer_df
            st.success(f"Fayl uğurla yükləndi! {len(customer_df)} müştəri məlumatı emal ediləcək.")
        else:
            st.error("Fayl emal edilə bilmədi. Zəhmət olmasa düzgün format yoxlayın.")
            return
    else:
        # Nümunə məlumatlar istifadə et
        try:
            _, _, customer_df = generate_sample_data_fixed()
            st.session_state.customer_data = customer_df
            st.info("Nümunə məlumatlar istifadə edilir. Həqiqi təhlil üçün öz faylınızı yükləyin.")
        except Exception as e:
            st.error(f"Nümunə məlumatların yüklənməsində xəta: {str(e)}")
            return
    
    customer_df = st.session_state.customer_data
    
    if customer_df is None or customer_df.empty:
        st.warning("Məlumat yoxdur. Zəhmət olmasa düzgün fayl yükləyin.")
        return
    
    # Məlumat keyfiyyəti yoxlanması
    st.subheader("Məlumat Keyfiyyəti")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Ümumi Qeydlər", len(customer_df))
    with col2:
        missing_pct = (customer_df.isnull().sum().sum() / (len(customer_df) * len(customer_df.columns))) * 100
        st.metric("Çatışmayan Məlumat", f"{missing_pct:.1f}%")
    with col3:
        numeric_cols = customer_df.select_dtypes(include=[np.number]).columns
        st.metric("Rəqəmsal Sütunlar", len(numeric_cols))
    with col4:
        duplicates = customer_df.duplicated().sum()
        st.metric("Təkrar Qeydlər", duplicates)
    
    # Əsas təhlil seçimi
    analysis_type = st.selectbox(
        "Təhlil növünü seçin:",
        ["Müştəri Seqmentasiyası", "Məhsul Meyil Analizi", "Regional Analiz", "Gəlir və Davranış Analizi"]
    )
    
    try:
        if analysis_type == "Müştəri Seqmentasiyası":
            perform_customer_segmentation(customer_df, gemini_api)
        elif analysis_type == "Məhsul Meyil Analizi":
            perform_product_propensity_analysis(customer_df, gemini_api)
        elif analysis_type == "Regional Analiz":
            perform_regional_analysis(customer_df, gemini_api)
        elif analysis_type == "Gəlir və Davranış Analizi":
            perform_income_behavior_analysis(customer_df, gemini_api)
    except Exception as e:
        st.error(f"Təhlildə xəta: {str(e)}")
    
    # Ümumi AI Strategiya Bölməsi
    st.markdown("---")
    st.subheader("🤖 AI tərəfindən Hərtərəfli Məhsul Strategiyası")
    st.info("Yüklənən məlumatlara əsasən ABB Bank üçün ümumi strategiya tövsiyələri")
    
    if st.button("Hərtərəfli Strategiya Yarat", key="comprehensive_strategy", type="primary"):
        with st.spinner("ABB Bank üçün hərtərəfli strategiya yaradılır..."):
            comprehensive_strategy = generate_comprehensive_product_strategy(customer_df, gemini_api)
            st.write(comprehensive_strategy)

def perform_customer_segmentation(customer_df, gemini_api):
    """Müştəri seqmentasiya təhlili"""
    st.subheader("Müştəri Seqmentasiyası")
    
    # Mövcud sütunları yoxla və uyğunlaş
    age_col = find_column(customer_df, ['yas', 'age', 'yaş'])
    income_col = find_column(customer_df, ['gelir', 'income', 'gəlir'])
    tenure_col = find_column(customer_df, ['muddet_ay', 'tenure', 'müddət'])
    
    if not all([age_col, income_col]):
        st.error("Seqmentasiya üçün 'yaş' və 'gəlir' sütunları tələb olunur.")
        return
    
    # Seqment təyin etmə funksiyası
    def assign_segment(row):
        age = row[age_col] if age_col else 35
        income = row[income_col] if income_col else 1000
        tenure = row[tenure_col] if tenure_col else 12
        
        if 25 <= age <= 35 and income >= 1200:
            return 'Gənc Peşəkar'
        elif 35 <= age <= 50 and income >= 1800:
            return 'Sabit'
        elif income >= 3000:
            return 'Premium'
        elif age >= 55:
            return 'Yaşlı'
        elif age <= 25:
            return 'Tələbə/Başlanğıc'
        else:
            return 'Kütləvi Bazar'
    
    customer_df['seqment'] = customer_df.apply(assign_segment, axis=1)
    
    # Vizuallaşdırma
    col1, col2 = st.columns(2)
    
    with col1:
        segment_counts = customer_df['seqment'].value_counts()
        fig = px.pie(values=segment_counts.values, names=segment_counts.index, 
                    title="Müştəri Seqmentləri")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if income_col:
            fig = px.box(customer_df, x='seqment', y=income_col, 
                        title="Seqmentlərə görə Gəlir Paylanması")
            fig.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
    
    # AI Seqment Strategiyası
    st.markdown("---")
    if st.button("Seqment Strategiyası Yarat", key="segment_strategy"):
        with st.spinner("Seqment strategiyası yaradılır..."):
            segment_analysis = analyze_customer_segments(customer_df, segment_counts, gemini_api)
            st.write(segment_analysis)

def perform_product_propensity_analysis(customer_df, gemini_api):
    """Məhsul meyil təhlili"""
    st.subheader("Məhsul Meyil Analizi")
    
    # Müştəri seç
    customer_id_col = find_column(customer_df, ['musteri_id', 'customer_id', 'id'])
    
    if customer_id_col:
        customer_ids = customer_df[customer_id_col].head(20).tolist()
        selected_customer_id = st.selectbox(
            "Təhlil üçün Müştəri Seçin:",
            customer_ids,
            key="product_customer_select"
        )
        
        if selected_customer_id:
            customer_data = customer_df[customer_df[customer_id_col] == selected_customer_id].iloc[0]
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.write("**Müştəri Profili:**")
                # Mövcud sütunları dinamik şəkildə göstər - daha etibarlı yolla
                
                # Yaş sütunu
                age_col = find_column(customer_df, ['yas', 'age', 'yaş'])
                if age_col and age_col in customer_data.index:
                    st.write(f"Yaş: {customer_data[age_col]}")
                
                # Gəlir sütunu
                income_col = find_column(customer_df, ['gelir', 'income', 'gəlir'])
                if income_col and income_col in customer_data.index:
                    st.write(f"Gəlir: {customer_data[income_col]}")
                
                # Müddət sütunu
                tenure_col = find_column(customer_df, ['muddet_ay', 'tenure', 'müddət'])
                if tenure_col and tenure_col in customer_data.index:
                    st.write(f"Müddət (ay): {customer_data[tenure_col]}")
                
                # Məhsul sayı sütunu
                product_col = find_column(customer_df, ['mehsul_sayi', 'products', 'məhsul_sayı'])
                if product_col and product_col in customer_data.index:
                    st.write(f"Məhsul sayı: {customer_data[product_col]}")
                
                # Region sütunu
                region_col = find_column(customer_df, ['region', 'şəhər', 'city'])
                if region_col and region_col in customer_data.index:
                    st.write(f"Region: {customer_data[region_col]}")
            
            with col2:
                # Məhsul meyillərini hesabla
                products = {
                    'Kredit Kartı': calculate_product_propensity_from_data(customer_data, customer_df, 'kredit_kart'),
                    'Şəxsi Kredit': calculate_product_propensity_from_data(customer_data, customer_df, 'sexsi_kredit'),
                    'Mortgage': calculate_product_propensity_from_data(customer_data, customer_df, 'mortgage'),
                    'İnvestisiya Hesabı': calculate_product_propensity_from_data(customer_data, customer_df, 'investisiya'),
                    'Sığorta': calculate_product_propensity_from_data(customer_data, customer_df, 'sigorta')
                }
                
                prop_df = pd.DataFrame(list(products.items()), columns=['Məhsul', 'Meyil'])
                prop_df = prop_df.sort_values('Meyil', ascending=True)
                
                fig = px.bar(prop_df, x='Meyil', y='Məhsul', orientation='h',
                           title=f"Müştəri {selected_customer_id} üçün Məhsul Meyili",
                           color='Meyil', color_continuous_scale='viridis')
                st.plotly_chart(fig, use_container_width=True)
                
                # Üst tövsiyələr
                st.write("**İlk 3 Tövsiyə:**")
                top_3 = prop_df.tail(3)
                for _, row in top_3.iterrows():
                    st.write(f"• {row['Məhsul']}: {row['Meyil']:.1%} ehtimal")
                    
                # AI tövsiyələri
                if st.button("AI Məhsul Tövsiyələri", key="ai_product_rec"):
                    generate_product_recommendations(customer_data, gemini_api)

def perform_regional_analysis(customer_df, gemini_api):
    """Regional təhlil"""
    st.subheader("Regional Analiz")
    
    region_col = find_column(customer_df, ['region', 'şəhər', 'city'])
    income_col = find_column(customer_df, ['gelir', 'income', 'gəlir'])
    
    if region_col:
        col1, col2 = st.columns(2)
        
        with col1:
            region_counts = customer_df[region_col].value_counts()
            fig = px.pie(values=region_counts.values, names=region_counts.index,
                        title="Regional Paylanma")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if income_col:
                avg_income_by_region = customer_df.groupby(region_col)[income_col].mean().sort_values(ascending=True)
                fig = px.bar(x=avg_income_by_region.values, y=avg_income_by_region.index,
                           orientation='h', title="Regiona görə Orta Gəlir")
                st.plotly_chart(fig, use_container_width=True)
        
        # Regional Strategiya
        st.markdown("---")
        if st.button("Regional Strategiya Yarat", key="regional_strategy"):
            with st.spinner("Regional strategiya yaradılır..."):
                regional_analysis = analyze_regional_data(customer_df, region_counts, avg_income_by_region if income_col else None, gemini_api)
                st.write(regional_analysis)
    else:
        st.warning("Regional analiz üçün 'region' sütunu tapılmadı.")

def perform_income_behavior_analysis(customer_df, gemini_api):
    """Gəlir və davranış analizi"""
    st.subheader("Gəlir və Davranış Analizi")
    
    income_col = find_column(customer_df, ['gelir', 'income', 'gəlir'])
    age_col = find_column(customer_df, ['yas', 'age', 'yaş'])
    
    if income_col and age_col:
        # Yaş və gəlir əlaqəsi - statsmodels olmadan sadə scatter plot
        fig = px.scatter(customer_df, x=age_col, y=income_col,
                        title="Yaş və Gəlir Əlaqəsi")
        st.plotly_chart(fig, use_container_width=True)
        
        # Gəlir seqmentləri
        customer_df['gelir_seqment'] = pd.cut(customer_df[income_col], 
                                              bins=3, labels=['Aşağı', 'Orta', 'Yüksək'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            income_seg_counts = customer_df['gelir_seqment'].value_counts()
            fig = px.bar(x=income_seg_counts.index, y=income_seg_counts.values,
                        title="Gəlir Seqmentləri")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Məhsul sayı və gəlir əlaqəsi
            product_col = find_column(customer_df, ['mehsul_sayi', 'products'])
            if product_col:
                fig = px.box(customer_df, x='gelir_seqment', y=product_col,
                           title="Gəlir Seqmentinə görə Məhsul Sayı")
                st.plotly_chart(fig, use_container_width=True)
        
        # AI analiz tövsiyələri
        if st.button("Davranış Analizi Yarat", key="behavior_analysis"):
            generate_behavior_analysis(customer_df, income_col, age_col, gemini_api)
    else:
        st.warning("Gəlir və yaş sütunları tapılmadı. Bu analiz üçün 'gelir' və 'yaş' sütunları tələb olunur.")

def analyze_customer_segments(customer_df, segment_counts, gemini_api):
    """Müştəri seqmentlərini AI ilə analiz et"""
    
    # Seqment statistikalarını hazırla
    age_col = find_column(customer_df, ['yas', 'age', 'yaş'])
    income_col = find_column(customer_df, ['gelir', 'income', 'gəlir'])
    product_col = find_column(customer_df, ['mehsul_sayi', 'products', 'məhsul_sayı'])
    
    segment_stats = {}
    if 'seqment' in customer_df.columns:
        for segment in segment_counts.index:
            segment_data = customer_df[customer_df['seqment'] == segment]
            segment_stats[segment] = {
                'sayı': len(segment_data),
                'orta_yaş': segment_data[age_col].mean() if age_col else 0,
                'orta_gəlir': segment_data[income_col].mean() if income_col else 0,
                'orta_məhsul': segment_data[product_col].mean() if product_col else 0
            }
    
    strategy_prompt = f"""
    ABB Bank üçün müştəri seqment analizi və strategiya tövsiyələri:
    
    ABB Bank məlumatları:
    - Bank adı: ABB Bank
    - Zəng Mərkəzi: 937
    - E-poçt: info@abb-bank.az
    
    Seqment Təhlili:
    {segment_stats}
    
    Ümumi məlumat:
    - Ümumi müştəri sayı: {len(customer_df)}
    - Ən böyük seqment: {segment_counts.index[0]} ({segment_counts.iloc[0]} müştəri)
    
    Hər seqment üçün:
    1. Xüsusi məhsul tövsiyələri
    2. Marketinq strategiyası
    3. Çarpaz satış imkanları
    4. Risk və potensial qiymətləndirmə
    
    ABB Bank-ın məhsul portfelinə uyğun təklif edin.
    """
    
    try:
        return gemini_api.generate_response(strategy_prompt, st.session_state.language)
    except Exception as e:
        return f"Strategiya yaradılmasında xəta: {str(e)}"

def analyze_regional_data(customer_df, region_counts, avg_income_by_region, gemini_api):
    """Regional məlumatları AI ilə analiz et"""
    
    # Regional statistikaları hazırla
    regional_stats = {}
    region_col = find_column(customer_df, ['region', 'şəhər', 'city'])
    
    if region_col:
        for region in region_counts.index:
            region_data = customer_df[customer_df[region_col] == region]
            regional_stats[region] = {
                'müştəri_sayı': len(region_data),
                'orta_gəlir': avg_income_by_region.get(region, 0) if avg_income_by_region is not None else 0,
                'payı': f"{len(region_data)/len(customer_df)*100:.1f}%"
            }
    
    regional_prompt = f"""
    ABB Bank üçün regional analiz və inkişaf strategiyası:
    
    ABB Bank məlumatları:
    - Bank adı: ABB Bank
    - Zəng Mərkəzi: 937
    - E-poçt: info@abb-bank.az
    
    Regional Təhlil:
    {regional_stats}
    
    Ən çox müştəri: {region_counts.index[0]} ({region_counts.iloc[0]} müştəri)
    {f"Ən yüksək gəlir: {avg_income_by_region.index[-1]} ({avg_income_by_region.iloc[-1]:.0f} AZN)" if avg_income_by_region is not None else ""}
    
    Hər region üçün:
    1. Bazar potensialı qiymətləndirmə
    2. Xüsusi məhsul strategiyası
    3. Filial və xidmət tövsiyələri
    4. Rəqabət mövqeyi
    5. Böyümə imkanları
    
    ABB Bank-ın regional inkişaf planını təqdim edin.
    """
    
    try:
        return gemini_api.generate_response(regional_prompt, st.session_state.language)
    except Exception as e:
        return f"Regional analiz yaradılmasında xəta: {str(e)}"

def generate_comprehensive_product_strategy(customer_df, gemini_api):
    """Ümumi məhsul strategiyası yarat"""
    
    # Əsas statistikaları topla
    age_col = find_column(customer_df, ['yas', 'age', 'yaş'])
    income_col = find_column(customer_df, ['gelir', 'income', 'gəlir'])
    tenure_col = find_column(customer_df, ['muddet_ay', 'tenure', 'müddət'])
    product_col = find_column(customer_df, ['mehsul_sayi', 'products', 'məhsul_sayı'])
    region_col = find_column(customer_df, ['region', 'şəhər', 'city'])
    digital_col = find_column(customer_df, ['reqemsal_qebul', 'digital_adoption'])
    
    comprehensive_stats = {
        'ümumi_müştəri': len(customer_df),
        'orta_yaş': customer_df[age_col].mean() if age_col else 0,
        'orta_gəlir': customer_df[income_col].mean() if income_col else 0,
        'orta_məhsul_sayı': customer_df[product_col].mean() if product_col else 0,
        'orta_müddət': customer_df[tenure_col].mean() if tenure_col else 0,
    }
    
    # Rəqəmsal qəbul analizi
    digital_analysis = ""
    if digital_col:
        digital_dist = customer_df[digital_col].value_counts()
        digital_analysis = f"Rəqəmsal Qəbul: {dict(digital_dist)}"
    
    # Regional paylanma
    regional_analysis = ""
    if region_col:
        regional_dist = customer_df[region_col].value_counts()
        regional_analysis = f"Regional Paylanma: {dict(regional_dist.head(3))}"
    
    strategy_prompt = f"""
    ABB Bank üçün hərtərəfli məhsul və çarpaz satış strategiyası yaradın:
    
    ABB Bank məlumatları:
    - Bank adı: ABB Bank
    - Zəng Mərkəzi: 937
    - E-poçt: info@abb-bank.az
    
    Mövcud Müştəri Bazası Analizi:
    {comprehensive_stats}
    
    {digital_analysis}
    {regional_analysis}
    
    Zəhmət olmasa aşağıdakları təqdim edin:
    
    1. **Məhsul Portfel Strategiyası**:
       - Hansı məhsulları prioritet etməli
       - Yeni məhsul imkanları
       - Cross-selling strategiyaları
    
    2. **Müştəri Seqment Tövsiyələri**:
       - Hər seqment üçün uyğun məhsullar
       - Targeting strategiyaları
       - Retention tədbirləri
    
    3. **Rəqəmsal Transformasiya**:
       - Mobil banking təkmilləşdirmə
       - AI və personallaşdırma
       - Customer journey optimizasiyası
    
    4. **Regional İnkişaf Planı**:
       - Bölgələrə görə fərqlənən yanaşmalar
       - Filial şəbəkəsi strategiyası
    
    5. **Performans Göstəriciləri (KPI)**:
       - Hansı metriklər izlənməli
       - Uğur kriteriyaları
    
    ABB Bank-ın mövcud xidmət portfeli və Azərbaycan bank bazarını nəzərə alın.
    """
    
    try:
        return gemini_api.generate_response(strategy_prompt, st.session_state.language)
    except Exception as e:
        return f"Strategiya yaradılmasında xəta: {str(e)}"

def find_column(df, possible_names):
    """Müxtəlif adlarla sütun tap"""
    for name in possible_names:
        if name in df.columns:
            return name
        # Case-insensitive axtarış
        for col in df.columns:
            if col.lower() == name.lower():
                return col
    return None

def calculate_product_propensity_from_data(customer_data, customer_df, product):
    """Həqiqi məlumatlardan məhsul meyili hesabla"""
    age_col = find_column(customer_df, ['yas', 'age', 'yaş'])
    income_col = find_column(customer_df, ['gelir', 'income', 'gəlir'])
    
    
    base_scores = {
        'kredit_kart': 0.4,
        'sexsi_kredit': 0.25,
        'mortgage': 0.15,
        'investisiya': 0.2,
        'sigorta': 0.3
    }
    
    score = base_scores.get(product, 0.25)
    
    # Yaş düzəlişləri
    if product == 'kredit_kart' and 25 <= age <= 45:
        score += 0.15
    elif product == 'mortgage' and 28 <= age <= 45:
        score += 0.2
    elif product == 'investisiya' and age >= 35:
        score += 0.15
    
    # Gəlir düzəlişləri
    if income >= 2500:
        score += 0.1
    elif income >= 1500:
        score += 0.05
    
    return min(0.95, score)

def generate_product_recommendations(customer_data, gemini_api):
    """AI məhsul tövsiyələri yarat"""
    rec_prompt = f"""
    ABB Bank üçün bu müştəri profilinə əsasən məhsul tövsiyələri yaradın:
    
    ABB Bank məlumatları:
    - Bank adı: ABB Bank
    - Zəng Mərkəzi: 937
    - E-poçt: info@abb-bank.az
    
    Müştəri Profili: {customer_data.to_dict()}
    
    3 ən uyğun məhsul tövsiyəsi verin və hər birini izah edin.
    """
    
    try:
        recommendations = gemini_api.generate_response(rec_prompt, st.session_state.language)
        st.write(recommendations)
    except Exception as e:
        st.error(f"Tövsiyələr yaradılmasında xəta: {str(e)}")

def generate_behavior_analysis(customer_df, income_col, age_col, gemini_api):
    """Davranış analizi yarat"""
    analysis_prompt = f"""
    ABB Bank üçün müştəri davranış analizi yaradın:
    
    ABB Bank məlumatları:
    - Bank adı: ABB Bank  
    - Zəng Mərkəzi: 937
    - E-poçt: info@abb-bank.az
    
    Məlumat Xülasəsi:
    - Ümumi müştəri sayı: {len(customer_df)}
    - Orta gəlir: {customer_df[income_col].mean():.0f} AZN
    - Orta yaş: {customer_df[age_col].mean():.0f} il
    - Gəlir diapazon: {customer_df[income_col].min():.0f} - {customer_df[income_col].max():.0f} AZN
    
    3 əsas davranış nümunəsi və marketinq tövsiyələri verin.
    """
    
    try:
        analysis = gemini_api.generate_response(analysis_prompt, st.session_state.language)
        st.write(analysis)
    except Exception as e:
        st.error(f"Analiz yaradılmasında xəta: {str(e)}")

def knowledge_search_page_improved(gemini_api):
    """Təkmilləşdirilmiş bilik axtarış səhifəsi"""
    st.title("Bilik Axtarışı və RAG Sistemi")
    st.markdown("---")
    
    # Bilik bazasını başlat (mövcud deyilsə)
    if 'kb_docs' not in st.session_state:
        st.session_state.kb_docs = [
            {
                'title': 'ABB Bank Kredit Kartı Qaydalari',
                'content': 'ABB Bank kredit kartının istifadə qaydalari: Aylıq komissiya 2 AZN, nağd pul çıxarma 1.5%, minimum ödəniş 5%. 24/7 online idarəetmə. Cashback proqramı mövcuddur. Əlavə məlumat üçün: 937 və ya info@abb-bank.az',
                'category': 'mehsullar'
            },
            {
                'title': 'ABB Mobil Banking Xidmətləri',
                'content': 'ABB mobil tətbiq vasitəsilə: pul köçürmələri, hesab yoxlanması, kommunal ödənişlər, kredit ödənişləri. Biometrik giriş, push bildirişlər. Texniki dəstək: 937, info@abb-bank.az',
                'category': 'reqemsal'
            },
            {
                'title': 'ABB Bank Kredit Şərtləri',
                'content': 'ABB Bank fərdi kreditlər: minimum gəlir 500 AZN, maksimum 50,000 AZN, müddət 60 aya qədər, faiz 12-18%. Zəmanət və ya girov tələb olunur. Məsləhət üçün: 937 və ya info@abb-bank.az',
                'category': 'kreditler'
            },
            {
                'title': 'ABB Bank Əlaqə Məlumatları',
                'content': 'ABB Bank əlaqə məlumatları: Zəng Mərkəzi 937 (24/7), E-poçt info@abb-bank.az, Onlayn banking, mobil tətbiq. Şikayətlər və təkliflər üçün həmçinin yazılı müraciət edə bilərsiniz.',
                'category': 'umumi'
            }
        ]
    
    # Sənəd idarəetməsi
    st.subheader("Bilik Bazası İdarəetməsi")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        with st.expander("Yeni Sənəd Əlavə Et"):
            title = st.text_input("Sənəd Başlığı", key="kb_title")
            category = st.selectbox("Kateqoriya", ['mehsullar', 'reqemsal', 'kreditler', 'umumi'], key="kb_category")
            content = st.text_area("Məzmun", height=100, key="kb_content")
            
            if st.button("Sənəd Əlavə Et", key="add_doc_btn"):
                if title and content:
                    new_doc = {
                        'title': title,
                        'content': content,
                        'category': category
                    }
                    st.session_state.kb_docs.append(new_doc)
                    st.success(f"'{title}' sənədi uğurla əlavə edildi!")
                    st.rerun()
                else:
                    st.warning("Zəhmət olmasa həm başlıq həm də məzmunu doldurun.")
    
    with col2:
        st.metric("Ümumi Sənədlər", len(st.session_state.kb_docs))
        
        categories = [doc['category'] for doc in st.session_state.kb_docs]
        if categories:
            cat_counts = pd.Series(categories).value_counts()
            for cat, count in cat_counts.items():
                st.write(f"{cat}: {count}")
    
    # Axtarış interfeysi
    st.subheader("Bilik Axtarışı")
    
    query = st.text_input(
        "Bank xidmətləri haqqında sual verin:",
        placeholder="Kredit kartının komissiyası nə qədərdir?",
        key="kb_query"
    )
    
    if query:
        try:
            # Sadə axtarış tətbiqi
            relevant_docs = search_documents(st.session_state.kb_docs, query)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write("**AI Cavabı:**")
                
                if relevant_docs:
                    context = " ".join([doc['content'] for doc in relevant_docs[:2]])
                    
                    answer_prompt = f"""
                    ABB Bank bilik bazası əsasında bu suala cavab verin:
                    
                    ABB Bank məlumatları:
                    - Bank adı: ABB Bank
                    - Zəng Mərkəzi: 937
                    - E-poçt: info@abb-bank.az
                    
                    Kontekst: {context}
                    Sual: {query}
                    
                    ABB Bank adından faydalı və dəqiq cavab verin. Cavabın sonunda əlaqə məlumatlarını qeyd edin.
                    """
                    
                    with st.spinner("Cavab yaradılır..."):
                        answer = gemini_api.generate_response(answer_prompt, st.session_state.language)
                        st.write(answer)
                else:
                    st.write("Təəssüf ki, sualınız üçün müvafiq məlumat tapa bilmədim.")
            
            with col2:
                st.write("**Müvafiq Sənədlər:**")
                
                for i, doc in enumerate(relevant_docs[:3]):
                    with st.expander(f"{doc['title']} ({doc.get('score', 0):.2f})"):
                        st.write(doc['content'][:200] + "...")
        
        except Exception as e:
            st.error(f"Axtarışda xəta: {str(e)}")

def search_documents(docs, query):
    """Sadə sənəd axtarış tətbiqi"""
    query_words = query.lower().split()
    
    scored_docs = []
    for doc in docs:
        content_lower = doc['content'].lower()
        title_lower = doc['title'].lower()
        
        # Sadə uyğunluq balı hesabla
        content_score = sum(1 for word in query_words if word in content_lower)
        title_score = sum(2 for word in query_words if word in title_lower)  # Başlıq uyğunluqları daha dəyərli
        
        total_score = content_score + title_score
        
        if total_score > 0:
            doc_copy = doc.copy()
            doc_copy['score'] = total_score / len(query_words)
            scored_docs.append(doc_copy)
    
    # Bal üzrə azalan sırada sıralama
    return sorted(scored_docs, key=lambda x: x['score'], reverse=True)

# Təkmilləşdirilmiş tətbiqi işə sal
if __name__ == "__main__":
    main()
