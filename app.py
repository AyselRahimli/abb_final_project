# Bank360 - 4 Səhifəli Versiya
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

# Təhlükəsiz import-lar
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
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
        'gemini_api_key': "",
        'knowledge_base': None,
        'initialized': False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

@st.cache_data
def generate_sample_data_fixed():
    """Nümunə məlumatları düzgün xəta idarəetməsi və ardıcıllıqla yarad"""
    np.random.seed(42)
    
    # Şikayət mətnləri - dəqiq 100 giriş
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
    
    text_az = [np.random.choice(base_complaint_texts) for _ in range(100)]
    
    # Şikayət məlumatları
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
    
    # Kredit məlumatları
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
    
    return pd.DataFrame(complaint_data), pd.DataFrame(loan_data).head(100)

class ImprovedGeminiAPI:
    """Təkmilləşdirilmiş Gemini API wrapper-i"""
    
    def __init__(self, api_key=None):
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
    
    def generate_response(self, prompt, language='az', max_retries=3):
        """Cavab yarad"""
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
    
    def _mock_response(self, prompt, language='az'):
        """ABB Bank məlumatları ilə mock cavab sistemi"""
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ['complaint', 'şikayət', 'problem']):
            return """Hörmətli müştəri,

ABB Bank olaraq şikayətinizi qəbul edirik və dərhal araşdırmaya başlayırıq. Bizim üçün müştəri məmnuniyyəti prioritetdir.

Əlaqə məlumatlarımız:
• Zəng Mərkəzi: 937
• E-poçt: info@abb-bank.az
• 24/7 online xidmət

2-3 iş günü ərzində sizinlə əlaqə saxlayacağıq.

Hörmətlə,
ABB Bank Müştəri Xidmətləri"""
        
        elif any(word in prompt_lower for word in ['credit', 'kredit', 'loan', 'risk']):
            return """ABB Bank kredit analizi nəticəsində:

Müştərinin ödəmə qabiliyyəti orta səviyyədə qiymətləndirilir. 

Əlavə məlumatlar:
• Zəng Mərkəzi: 937
• E-poçt: info@abb-bank.az
• Kredit departamenti ilə əlavə məsləhətləşmə tövsiyə olunur"""
        
        else:
            return """ABB Bank olaraq sorğunuz əsasında analiz aparılmış və müvafiq tövsiyələr hazırlanmışdır.

Əlaqə məlumatlarımız:
• Zəng Mərkəzi: 937  
• E-poçt: info@abb-bank.az

Əlavə məlumat üçün müvafiq departamentlə əlaqə saxlayın."""

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
        help="AI xüsusiyyətləri üçün Google Gemini API açarınızı daxil edin"
    )
    
    if api_key != st.session_state.gemini_api_key:
        st.session_state.gemini_api_key = api_key
        if api_key:
            st.sidebar.success("API açarı yeniləndi!")
    
    # Naviqasiya menyusu - sadəcə 4 səhifə
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Naviqasiya")
    
    pages = ['Ana Səhifə', 'Şikayətlər', 'Kredit Riski', 'Bilik Axtarışı']
    
    selected_page = st.sidebar.radio(
        "Səhifə Seçin",
        pages
    )
    
    return selected_page

@st.cache_data
def safe_sentiment_analysis(texts):
    """Sentiment analizi"""
    results = []
    positive_words = ['yaxşı', 'əla', 'mükəmməl', 'razıyam', 'təşəkkür', 'gözəl']
    negative_words = ['pis', 'səhv', 'problem', 'şikayət', 'narazıyam', 'yavaş']
    
    for text in texts:
        try:
            text_lower = str(text).lower()
            
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)
            
            if pos_count > neg_count:
                sentiment = 'müsbət'
                score = 0.7
            elif neg_count > pos_count:
                sentiment = 'mənfi'
                score = 0.3
            else:
                sentiment = 'neytral'
                score = 0.5
            
            results.append({
                'sentiment': sentiment,
                'score': score,
                'confidence': 0.8
            })
        except:
            results.append({
                'sentiment': 'neytral',
                'score': 0.5,
                'confidence': 0.5
            })
    
    return results

def home_page(gemini_api):
    """Ana səhifə"""
    st.title("🏦 Bank360 Analitika İdarə Paneli")
    st.markdown("---")
    
    try:
        complaint_df, loan_df = generate_sample_data_fixed()
    except Exception as e:
        st.error(f"Məlumat yüklənməsində xəta: {str(e)}")
        return
    
    # KPI sətriri
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Ümumi Şikayətlər", len(complaint_df), delta=f"+{np.random.randint(5, 15)}")
    
    with col2:
        csat_score = np.random.uniform(3.8, 4.5)
        st.metric("CSAT Balı", f"{csat_score:.1f}/5.0", delta=f"+{np.random.uniform(0.1, 0.3):.1f}")
    
    with col3:
        high_severity = len(complaint_df[complaint_df['ciddilik'] == 'yüksək'])
        st.metric("Yüksək Ciddiyyət", high_severity, delta=f"-{np.random.randint(1, 3)}")
    
    with col4:
        avg_pd = loan_df['borc_gelir_nisbeti'].mean() * 0.25
        st.metric("Orta PD", f"{avg_pd:.1%}", delta=f"{np.random.uniform(-0.01, 0.01):+.1%}")
    
    st.markdown("---")
    
    # Qrafiklər
    col1, col2 = st.columns(2)
    
    with col1:
        category_counts = complaint_df['kateqoriya'].value_counts()
        fig = px.pie(
            values=category_counts.values,
            names=category_counts.index,
            title="Şikayət Kateqoriyaları"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        daily_complaints = complaint_df.groupby(complaint_df['tarix'].dt.date).size()
        fig = px.line(
            x=daily_complaints.index,
            y=daily_complaints.values,
            title="Gündəlik Şikayət Tendensiyaları"
        )
        st.plotly_chart(fig, use_container_width=True)

def complaints_page(gemini_api):
    """Şikayətlər səhifəsi"""
    st.title("Şikayətlər və Rəy Təhlili")
    st.markdown("---")
    
    try:
        complaint_df, _ = generate_sample_data_fixed()
    except Exception as e:
        st.error(f"Məlumat yüklənməsində xəta: {str(e)}")
        return
    
    # Məlumat baxışı
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Ümumi Qeydlər", len(complaint_df))
    with col2:
        high_sev = len(complaint_df[complaint_df['ciddilik'] == 'yüksək'])
        st.metric("Yüksək Ciddiyyət", high_sev)
    with col3:
        open_cases = len(complaint_df[complaint_df['status'] == 'Açıq'])
        st.metric("Açıq İşlər", open_cases)
    with col4:
        avg_days = np.random.randint(2, 7)
        st.metric("Orta Həll (gün)", avg_days)
    
    # Təhlil tabları
    tab1, tab2, tab3 = st.tabs(["Sentiment Təhlili", "Kateqoriya Təhlili", "Cavab Yaradıcısı"])
    
    with tab1:
        st.subheader("Sentiment Təhlili")
        
        if st.button("Sentimentləri Təhlil Et"):
            sample_texts = complaint_df['metn_az'].head(20).tolist()
            sentiments = safe_sentiment_analysis(sample_texts)
            
            if sentiments:
                sentiment_labels = [s['sentiment'] for s in sentiments]
                
                sentiment_counts = pd.Series(sentiment_labels).value_counts()
                fig = px.pie(
                    values=sentiment_counts.values,
                    names=sentiment_counts.index,
                    title="Sentiment Paylanması"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Kateqoriya Təhlili")
        
        category_counts = complaint_df['kateqoriya'].value_counts()
        
        fig = px.bar(
            x=category_counts.values,
            y=category_counts.index,
            orientation='h',
            title="Kateqoriyalara görə Şikayətlər"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("AI Cavab Yaradıcısı")
        
        complaint_options = complaint_df['metn_az'].head(5).tolist()
        
        selected_complaint = st.selectbox(
            "Cavab yaratmaq üçün şikayət seçin:",
            complaint_options
        )
        
        if st.button("Peşəkar Cavab Yarat"):
            response = gemini_api.generate_response(
                f"""ABB Bank olaraq bu müştəri şikayətinə peşəkar cavab yaradın:
                
                Şikayət: {selected_complaint}
                
                Cavab hörmətli, peşəkar və həllledici olsun."""
            )
            st.write("**Yaradılan Cavab:**")
            st.write(response)

def credit_risk_page(gemini_api):
    """Kredit riski səhifəsi"""
    st.title("Kredit Riski və Gözlənilən İtki Təhlili")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Şəxsi Məlumatlar**")
        age = st.slider("Yaş", 18, 80, 35)
        income = st.number_input("Aylıq Gəlir (AZN)", 300.0, 15000.0, 1500.0)
        employment = st.selectbox("İş Vəziyyəti", ['dövlət', 'işçi', 'sərbəst_işçi', 'işsiz'])
        credit_score = st.slider("Kredit Reytinqi", 300, 850, 650)
    
    with col2:
        st.write("**Kredit Məlumatları**")
        loan_amount = st.number_input("Kredit Məbləği (AZN)", 1000.0, 100000.0, 25000.0)
        debt_to_income = st.slider("Borc-Gəlir Nisbəti", 0.0, 1.0, 0.3)
        collateral_value = st.number_input("Təminat Dəyəri (AZN)", 0.0, 200000.0, 30000.0)
        loan_to_value = st.slider("Kredit-Təminat Nisbəti", 0.0, 1.0, 0.8)
    
    if st.button("Risk Metriklər Hesabla"):
        # Sadələşdirilmiş PD hesablama
        base_pd = 0.15
        
        # Yaş faktoru
        age_factor = 0.03 if age < 25 or age > 65 else (-0.02 if 35 <= age <= 50 else 0)
        
        # Gəlir faktoru
        income_factor = -0.00002 * income if income > 0 else 0.1
        
        # İş faktoru
        emp_factors = {'dövlət': -0.03, 'işçi': -0.01, 'sərbəst_işçi': 0.02, 'işsiz': 0.15}
        employment_factor = emp_factors.get(employment, 0)
        
        # Kredit reytinq faktoru
        credit_factor = -0.0002 * (credit_score - 600)
        
        pd_score = max(0.01, min(0.95, base_pd + age_factor + income_factor + employment_factor + credit_factor))
        
        # LGD hesabla
        lgd = 0.2 if collateral_value >= loan_amount else max(0.3, 0.8 - (collateral_value/loan_amount * 0.5))
        
        # EAD və Expected Loss
        ead = loan_amount * 0.85
        expected_loss = pd_score * lgd * ead
        
        # Nəticələri göstər
        st.markdown("---")
        st.subheader("Risk Qiymətləndirmə Nəticələri")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            risk_level = "Yüksək" if pd_score > 0.2 else "Orta" if pd_score > 0.1 else "Aşağı"
            st.metric("Defolt Ehtimalı (PD)", f"{pd_score:.2%}")
            st.write(f"**Risk Səviyyəsi:** {risk_level}")
        
        with col2:
            st.metric("Defoltda İtki (LGD)", f"{lgd:.2%}")
            st.metric("Defoltda Məruz Qalma (EAD)", f"{ead:,.0f} AZN")
        
        with col3:
            st.metric("Gözlənilən İtki (EL)", f"{expected_loss:,.0f} AZN")

def knowledge_search_page(gemini_api):
    """Bilik axtarışı səhifəsi"""
    st.title("Bilik Axtarışı və RAG Sistemi")
    st.markdown("---")
    
    # Bilik bazasını başlat
    if 'kb_docs' not in st.session_state:
        st.session_state.kb_docs = [
            {
                'title': 'ABB Bank Kredit Kartı Qaydalari',
                'content': 'ABB Bank kredit kartının istifadə qaydalari: Aylıq komissiya 2 AZN, nağd pul çıxarma 1.5%, minimum ödəniş 5%. Əlavə məlumat üçün: 937 və ya info@abb-bank.az',
                'category': 'məhsullar'
            },
            {
                'title': 'ABB Mobil Banking Xidmətləri',
                'content': 'ABB mobil tətbiq vasitəsilə: pul köçürmələri, hesab yoxlanması, kommunal ödənişlər, kredit ödənişləri. Texniki dəstək: 937, info@abb-bank.az',
                'category': 'rəqəmsal'
            },
            {
                'title': 'ABB Bank Kredit Şərtləri',
                'content': 'ABB Bank fərdi kreditlər: minimum gəlir 500 AZN, maksimum 50,000 AZN, müddət 60 aya qədər, faiz 12-18%. Məsləhət üçün: 937 və ya info@abb-bank.az',
                'category': 'kreditlər'
            },
            {
                'title': 'ABB Bank Əlaqə Məlumatları',
                'content': 'ABB Bank əlaqə məlumatları: Zəng Mərkəzi 937 (24/7), E-poçt info@abb-bank.az, Online banking, mobil tətbiq.',
                'category': 'ümumi'
            }
        ]
    
    # Axtarış interfeysi
    st.subheader("Bilik Axtarışı")
    
    query = st.text_input(
        "Bank xidmətləri haqqında sual verin:",
        placeholder="Kredit kartının komissiyası nə qədərdir?"
    )
    
    if query:
        # Sadə axtarış
        relevant_docs = search_documents(st.session_state.kb_docs, query)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("**AI Cavabı:**")
            
            if relevant_docs:
                context = " ".join([doc['content'] for doc in relevant_docs[:2]])
                
                answer_prompt = f"""
                ABB Bank bilik bazası əsasında bu suala cavab verin:
                
                Kontekst: {context}
                Sual: {query}
                
                ABB Bank adından faydalı və dəqiq cavab verin.
                """
                
                answer = gemini_api.generate_response(answer_prompt, st.session_state.language)
                st.write(answer)
            else:
                st.write("Təəssüf ki, sualınız üçün müvafiq məlumat tapa bilmədim.")
        
        with col2:
            st.write("**Müvafiq Sənədlər:**")
            
            for doc in relevant_docs[:3]:
                with st.expander(f"{doc['title']}"):
                    st.write(doc['content'][:200] + "...")

def search_documents(docs, query):
    """Sadə sənəd axtarış tətbiqi"""
    query_words = query.lower().split()
    
    scored_docs = []
    for doc in docs:
        content_lower = doc['content'].lower()
        title_lower = doc['title'].lower()
        
        content_score = sum(1 for word in query_words if word in content_lower)
        title_score = sum(2 for word in query_words if word in title_lower)
        
        total_score = content_score + title_score
        
        if total_score > 0:
            doc_copy = doc.copy()
            doc_copy['score'] = total_score / len(query_words)
            scored_docs.append(doc_copy)
    
    return sorted(scored_docs, key=lambda x: x['score'], reverse=True)

def main():
    """Əsas tətbiq"""
    initialize_session_state()
    
    # API açarını yükləməyə çalış
    if not st.session_state.gemini_api_key and not st.session_state.initialized:
        try:
            st.session_state.gemini_api_key = st.secrets.get("GEMINI_API_KEY", "")
        except:
            pass
        st.session_state.initialized = True
    
    # API-ni başlat
    gemini_api = ImprovedGeminiAPI(st.session_state.gemini_api_key)
    
    # Naviqasiya
    selected_page = improved_sidebar_navigation()
    
    # Müvafiq səhifəyə yönləndir
    if selected_page == 'Ana Səhifə':
        home_page(gemini_api)
    elif selected_page == 'Şikayətlər':
        complaints_page(gemini_api)
    elif selected_page == 'Kredit Riski':
        credit_risk_page(gemini_api)
    elif selected_page == 'Bilik Axtarışı':
        knowledge_search_page(gemini_api)

if __name__ == "__main__":
    main()
