# Bank360 - Səliqəli və Xətasız Versiya
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
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Streamlit konfiqurasiyası
st.set_page_config(
    page_title="Bank360 Analitika",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    """Session state başlat"""
    defaults = {
        'language': 'az',
        'complaint_data': None,
        'loan_data': None,
        'customer_data': None,
        'gemini_api_key': ""
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

@st.cache_data
def generate_sample_data():
    """Nümunə məlumatlar yarad"""
    np.random.seed(42)
    
    # Şikayət məlumatları
    complaint_data = {
        'id': list(range(1, 101)),
        'tarix': pd.date_range(start='2024-01-01', periods=100, freq='D'),
        'musteri_id': np.random.randint(1000, 9999, 100),
        'kanal': np.random.choice(['Mobil', 'Filial', 'Zəng', 'Veb'], 100),
        'kateqoriya': np.random.choice(['Kart', 'ATM', 'Mobil', 'Komissiya'], 100),
        'metn_az': ['Problem var'] * 100,
        'ciddilik': np.random.choice(['aşağı', 'orta', 'yüksək'], 100),
        'status': np.random.choice(['Açıq', 'Prosesdə', 'Bağlı'], 100)
    }
    
    # Kredit məlumatları  
    loan_data = {
        'musteri_id': list(range(1, 101)),
        'yas': np.random.randint(18, 80, 100),
        'gelir': np.random.randint(500, 5000, 100),
        'kredit_reytingi': np.random.randint(300, 850, 100),
        'kredit_meblegi': np.random.randint(1000, 50000, 100),
        'muddet_ay': np.random.randint(6, 60, 100)
    }
    
    # Müştəri məlumatları
    customer_data = {
        'musteri_id': list(range(1, 101)),
        'yas': np.random.randint(18, 80, 100),
        'gelir': np.random.randint(500, 5000, 100),
        'muddet_ay': np.random.randint(1, 60, 100),
        'mehsul_sayi': np.random.randint(1, 6, 100),
        'region': np.random.choice(['Bakı', 'Gəncə', 'Sumqayıt'], 100)
    }
    
    return (
        pd.DataFrame(complaint_data),
        pd.DataFrame(loan_data),
        pd.DataFrame(customer_data)
    )

class GeminiAPI:
    """Sadə Gemini API wrapper"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.model = None
        self.initialized = False
        
        if api_key and GEMINI_AVAILABLE:
            try:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel('gemini-2.5-flash')
                self.initialized = True
            except Exception as e:
                st.error(f"API xətası: {str(e)}")
    
    def generate_response(self, prompt, language='az'):
        """Cavab yarat"""
        if not self.initialized:
            return self._mock_response(prompt)
        
        try:
            lang_instruction = "Azərbaycan dilində cavab ver" if language == 'az' else "Respond in English"
            full_prompt = f"{lang_instruction}. {prompt}"
            response = self.model.generate_content(full_prompt)
            return response.text if response.text else self._mock_response(prompt)
        except:
            return self._mock_response(prompt)
    
    def _mock_response(self, prompt):
        """Mock cavab"""
        return """ABB Bank analiz nəticəsi:

Bu məlumatlar əsasında aşağıdakı tövsiyələr verilir:

1. Müştəri bazasını genişləndirin
2. Rəqəmsal xidmətləri inkişaf etdirin  
3. Risk idarəetməsini gücləndirilir
4. Məhsul portfelini təkmilləşdirin
5. Müştəri məmnuniyyətini artırın

Əlavə məlumat üçün:
- Zəng Mərkəzi: 937
- E-poçt: info@abb-bank.az

Hörmətlə, ABB Bank"""

def validate_uploaded_file(uploaded_file):
    """Fayl yoxla"""
    if uploaded_file is None:
        return None
    
    try:
        if uploaded_file.type == 'text/csv':
            df = pd.read_csv(uploaded_file)
        elif 'excel' in uploaded_file.type:
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Yalnız CSV və Excel faylları dəstəklənir")
            return None
        
        if df.empty:
            st.error("Fayl boşdur")
            return None
        
        st.success(f"Fayl yükləndi: {len(df)} sətir, {len(df.columns)} sütun")
        return df
        
    except Exception as e:
        st.error(f"Fayl oxunmadı: {str(e)}")
        return None

def sidebar_navigation():
    """Yan panel naviqasiyası"""
    st.sidebar.title("🏦 Bank360")
    
    # Dil seçimi
    language = st.sidebar.selectbox(
        "Dil / Language",
        ['Azərbaycan', 'English']
    )
    st.session_state.language = 'az' if language == 'Azərbaycan' else 'en'
    
    # API Key
    st.sidebar.subheader("Tənzimlər")
    api_key = st.sidebar.text_input(
        "Gemini API Key",
        type="password",
        value=st.session_state.gemini_api_key
    )
    st.session_state.gemini_api_key = api_key
    
    # Naviqasiya
    st.sidebar.subheader("Səhifələr")
    pages = ['Ana Səhifə', 'Şikayətlər', 'Kredit Riski', 'Məhsul Analizi', 'Bilik Bazası']
    selected_page = st.sidebar.radio("Seçin:", pages)
    
    return selected_page

def home_page(gemini_api):
    """Ana səhifə"""
    st.title("🏦 Bank360 Analitika")
    st.markdown("---")
    
    try:
        complaint_df, loan_df, customer_df = generate_sample_data()
    except Exception as e:
        st.error(f"Məlumat yüklənmə xətası: {str(e)}")
        return
    
    # KPI-lər
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Ümumi Şikayətlər", len(complaint_df))
    with col2:
        st.metric("CSAT Balı", "4.2/5.0")
    with col3:
        high_priority = len(complaint_df[complaint_df['ciddilik'] == 'yüksək'])
        st.metric("Yüksək Prioritet", high_priority)
    with col4:
        st.metric("Orta Risk", "12.5%")
    
    # Qrafiklər
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        # Kateqoriya qrafiği
        try:
            category_counts = complaint_df['kateqoriya'].value_counts()
            fig = px.pie(
                values=category_counts.values,
                names=category_counts.index,
                title="Şikayət Kateqoriyaları"
            )
            st.plotly_chart(fig, use_container_width=True)
        except:
            st.info("Qrafik göstərilə bilmir")
    
    with col2:
        # Tendensiya qrafiği
        try:
            daily_data = complaint_df.groupby(complaint_df['tarix'].dt.date).size()
            fig = px.line(
                x=daily_data.index,
                y=daily_data.values,
                title="Gündəlik Tendensiya"
            )
            st.plotly_chart(fig, use_container_width=True)
        except:
            st.info("Qrafik göstərilə bilmir")
    
    # AI İntelekt
    st.markdown("---")
    st.subheader("🤖 AI Təhlili")
    
    if st.button("Biznes Təhlili Yarat"):
        with st.spinner("Analiz yaradılır..."):
            prompt = f"""
            ABB Bank performans analizi:
            
            ABB Bank məlumatları:
            - Bank adı: ABB Bank
            - Zəng Mərkəzi: 937  
            - E-poçt: info@abb-bank.az
            
            Məlumatlar:
            - Şikayət sayı: {len(complaint_df)}
            - Yüksək prioritet: {high_priority}
            - Müştəri sayı: {len(customer_df)}
            
            3 əsas biznes tövsiyəsi ver.
            """
            
            response = gemini_api.generate_response(prompt)
            st.write(response)

def complaints_page(gemini_api):
    """Şikayətlər səhifəsi"""
    st.title("Şikayətlər və Rəy Analizi")
    st.markdown("---")
    
    # Fayl yükləmə
    st.subheader("Məlumat Yükləmə")
    uploaded_file = st.file_uploader("CSV və ya Excel fayl seçin", type=['csv', 'xlsx'])
    
    if uploaded_file:
        data = validate_uploaded_file(uploaded_file)
        if data is not None:
            st.session_state.complaint_data = data
    else:
        # Nümunə məlumat
        complaint_df, _, _ = generate_sample_data()
        st.session_state.complaint_data = complaint_df
        st.info("Nümunə məlumatlar göstərilir")
    
    data = st.session_state.complaint_data
    
    if data is None:
        st.warning("Məlumat yoxdur")
        return
    
    # Əsas metriklər
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Ümumi Qeydlər", len(data))
    with col2:
        if 'ciddilik' in data.columns:
            high_sev = len(data[data['ciddilik'] == 'yüksək'])
            st.metric("Yüksək Ciddiyyət", high_sev)
    with col3:
        if 'status' in data.columns:
            open_cases = len(data[data['status'] == 'Açıq'])
            st.metric("Açıq İşlər", open_cases)
    
    # Vizuallaşdırma
    if st.button("Qrafiklər Göstər"):
        if 'kateqoriya' in data.columns:
            category_counts = data['kateqoriya'].value_counts()
            fig = px.bar(
                x=category_counts.index,
                y=category_counts.values,
                title="Kateqoriya Paylanması"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # AI cavab generatoru
    st.subheader("AI Cavab Yaradıcısı")
    if 'metn_az' in data.columns:
        complaint_text = st.selectbox(
            "Şikayət seçin:",
            data['metn_az'].head(5).tolist()
        )
        
        if st.button("Cavab Yarat"):
            prompt = f"""
            ABB Bank olaraq bu şikayətə cavab yarat:
            
            Bank: ABB Bank
            Zəng: 937
            E-poçt: info@abb-bank.az
            
            Şikayət: {complaint_text}
            
            Peşəkar və həlledici cavab ver.
            """
            
            response = gemini_api.generate_response(prompt)
            st.write("**Yaradılan Cavab:**")
            st.write(response)

def credit_risk_page(gemini_api):
    """Kredit risk səhifəsi"""
    st.title("Kredit Risk Analizi")
    st.markdown("---")
    
    st.subheader("Risk Hesablaması")
    
    # Input formalar
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Yaş", 18, 80, 35)
        income = st.number_input("Gəlir (AZN)", 300, 10000, 1500)
        employment = st.selectbox("İş", ['dövlət', 'özəl', 'sərbəst'])
        credit_score = st.slider("Kredit Reytinqi", 300, 850, 650)
    
    with col2:
        loan_amount = st.number_input("Kredit Məbləği", 1000, 100000, 25000)
        collateral = st.number_input("Təminat", 0, 200000, 30000)
        debt_ratio = st.slider("Borc/Gəlir", 0.0, 1.0, 0.3)
        term = st.slider("Müddət (ay)", 6, 120, 36)
    
    if st.button("Risk Hesabla"):
        # Sadə risk hesablaması
        base_risk = 0.15
        
        if age < 25 or age > 65:
            base_risk += 0.03
        if income < 1000:
            base_risk += 0.05
        if credit_score < 600:
            base_risk += 0.1
        if debt_ratio > 0.5:
            base_risk += 0.05
        
        risk_score = max(0.01, min(0.95, base_risk))
        
        # Nəticələr
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Default Risk (PD)", f"{risk_score:.2%}")
        with col2:
            lgd = 0.45 if collateral < loan_amount else 0.25
            st.metric("Loss Given Default", f"{lgd:.2%}")
        with col3:
            expected_loss = risk_score * lgd * loan_amount
            st.metric("Expected Loss", f"{expected_loss:,.0f} AZN")
        
        # AI təhlili
        if st.button("Risk Təhlili"):
            prompt = f"""
            ABB Bank kredit risk analizi:
            
            Bank: ABB Bank
            Zəng: 937
            E-poçt: info@abb-bank.az
            
            Müştəri profili:
            - Yaş: {age}
            - Gəlir: {income} AZN
            - Kredit: {loan_amount} AZN
            - Risk: {risk_score:.2%}
            
            Risk qiymətləndirmə və tövsiyə ver.
            """
            
            analysis = gemini_api.generate_response(prompt)
            st.write(analysis)

def product_insights_page(gemini_api):
    """Məhsul analizi səhifəsi"""
    st.title("Məhsul Analizi")
    st.markdown("---")
    
    # Fayl yükləmə
    st.subheader("Məlumat Yükləmə")
    uploaded_file = st.file_uploader("Müştəri məlumatları faylı", type=['csv', 'xlsx'])
    
    if uploaded_file:
        customer_df = validate_uploaded_file(uploaded_file)
        if customer_df is not None:
            st.session_state.customer_data = customer_df
    else:
        _, _, customer_df = generate_sample_data()
        st.session_state.customer_data = customer_df
        st.info("Nümunə məlumatlar istifadə edilir")
    
    data = st.session_state.customer_data
    
    if data is None:
        st.warning("Məlumat yoxdur")
        return
    
    # Məlumat xülasəsi
    st.subheader("Məlumat Xülasəsi")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Müştəri Sayı", len(data))
    with col2:
        st.metric("Sütun Sayı", len(data.columns))
    with col3:
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        st.metric("Rəqəmsal Sütunlar", len(numeric_cols))
    
    # Sütunları göstər
    st.write("**Sütunlar:**", list(data.columns))
    
    # Analiz seçimi
    analysis_type = st.selectbox(
        "Analiz növü:",
        ["Əsas Statistika", "Müştəri Profili", "AI Tövsiyələri", "Vizuallar"]
    )
    
    if analysis_type == "Əsas Statistika":
        show_basic_stats(data)
    elif analysis_type == "Müştəri Profili":
        show_customer_profile(data)
    elif analysis_type == "AI Tövsiyələri":
        show_ai_recommendations(data, gemini_api)
    elif analysis_type == "Vizuallar":
        show_visualizations(data)

def show_basic_stats(df):
    """Əsas statistika göstər"""
    st.subheader("Əsas Statistika")
    
    # Rəqəmsal sütunlar
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        st.write("**Rəqəmsal Sütunlar:**")
        st.dataframe(df[numeric_cols].describe())
    
    # Kateqoriya sütunları
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        unique_vals = df[col].nunique()
        if unique_vals <= 10:
            st.write(f"**{col}:**")
            st.write(df[col].value_counts())

def show_customer_profile(df):
    """Müştəri profili göstər"""
    st.subheader("Müştəri Profili")
    
    # İlk sütunu ID kimi götür
    id_col = df.columns[0]
    customer_ids = df[id_col].head(10).tolist()
    
    selected_id = st.selectbox("Müştəri seçin:", customer_ids)
    
    if selected_id:
        customer = df[df[id_col] == selected_id].iloc[0]
        
        st.write("**Müştəri Məlumatları:**")
        for col in df.columns:
            st.write(f"**{col}:** {customer[col]}")
        
        # Sadə tövsiyələr
        st.write("**Tövsiyələr:**")
        recommendations = []
        
        # Yaş əsaslı
        for age_col in ['age', 'yas']:
            if age_col in df.columns:
                age = customer[age_col]
                if age < 30:
                    recommendations.append("Gənclik məhsulları")
                elif age < 50:
                    recommendations.append("Premium xidmətlər")
                else:
                    recommendations.append("Pensiya planları")
                break
        
        # Gəlir əsaslı
        for income_col in ['income', 'gelir']:
            if income_col in df.columns:
                income = customer[income_col]
                if income > 3000:
                    recommendations.append("Yüksək gəlirli paket")
                elif income > 1500:
                    recommendations.append("Orta səviyyə paket")
                else:
                    recommendations.append("Əsas paket")
                break
        
        if not recommendations:
            recommendations = ["Kredit kartı", "Əmanət hesabı", "Mobil banking"]
        
        for rec in recommendations:
            st.write(f"• {rec}")

def show_ai_recommendations(df, gemini_api):
    """AI tövsiyələri göstər"""
    st.subheader("AI Tövsiyələri")
    
    if st.button("Strategiya Yarat", type="primary"):
        with st.spinner("AI strategiya yaradır..."):
            summary = f"""
            Müştəri bazası analizi:
            - Müştəri sayı: {len(df)}
            - Sütunlar: {list(df.columns)}
            - Rəqəmsal sütunlar: {list(df.select_dtypes(include=[np.number]).columns)}
            """
            
            prompt = f"""
            ABB Bank strategiya tövsiyələri:
            
            ABB Bank:
            - Zəng: 937
            - E-poçt: info@abb-bank.az
            
            Məlumat: {summary}
            
            5 strategiya tövsiyəsi:
            1. Müştəri seqmentasiyası
            2. Məhsul inkişafı  
            3. Rəqəmsal transformasiya
            4. Risk idarəetməsi
            5. Marketinq strategiyası
            """
            
            response = gemini_api.generate_response(prompt)
            st.write(response)

def show_visualizations(df):
    """Vizuallaşdırma göstər"""
    st.subheader("Vizuallaşdırma")
    
    # Rəqəmsal sütunlar üçün histoqram
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        selected_col = st.selectbox("Histoqram üçün sütun:", numeric_cols)
        if selected_col:
            fig = px.histogram(df, x=selected_col, title=f"{selected_col} Paylanması")
            st.plotly_chart(fig, use_container_width=True)
    
    # Kateqoriya sütunları
    cat_cols = df.select_dtypes(include=['object']).columns
    valid_cats = [col for col in cat_cols if df[col].nunique() <= 10]
    
    if valid_cats:
        selected_cat = st.selectbox("Bar chart üçün:", valid_cats)
        if selected_cat:
            counts = df[selected_cat].value_counts()
            fig = px.bar(x=counts.index, y=counts.values, title=f"{selected_cat} Sayları")
            st.plotly_chart(fig, use_container_width=True)

def knowledge_base_page(gemini_api):
    """Bilik bazası səhifəsi"""
    st.title("Bilik Bazası")
    st.markdown("---")
    
    # Sadə bilik bazası
    if 'kb_docs' not in st.session_state:
        st.session_state.kb_docs = [
            {
                'title': 'ABB Bank Kredit Kartı',
                'content': 'ABB Bank kredit kartı: 2 AZN aylıq komissiya, 1.5% nağd çıxarma. 24/7 online. Məlumat: 937, info@abb-bank.az'
            },
            {
                'title': 'ABB Mobil Banking', 
                'content': 'Mobil tətbiq: pul köçürmə, hesab yoxlama, kommunal ödəniş. Biometrik giriş. Dəstək: 937'
            },
            {
                'title': 'ABB Kreditlər',
                'content': 'Fərdi kreditlər: min 500 AZN gəlir, maks 50,000 AZN, 60 ay, faiz 12-18%. Məsləhət: 937'
            }
        ]
    
    # Yeni sənəd əlavə etmə
    st.subheader("Yeni Sənəd")
    with st.expander("Sənəd Əlavə Et"):
        title = st.text_input("Başlıq")
        content = st.text_area("Məzmun")
        
        if st.button("Əlavə Et"):
            if title and content:
                st.session_state.kb_docs.append({
                    'title': title,
                    'content': content
                })
                st.success("Əlavə edildi!")
                st.rerun()
    
    # Sənəd sayı
    st.metric("Sənəd Sayı", len(st.session_state.kb_docs))
    
    # Axtarış
    st.subheader("Axtarış")
    query = st.text_input("Sual verin:", placeholder="Kredit kartı haqqında...")
    
    if query:
        # Sadə axtarış
        results = []
        for doc in st.session_state.kb_docs:
            if query.lower() in doc['content'].lower() or query.lower() in doc['title'].lower():
                results.append(doc)
        
        if results:
            context = " ".join([doc['content'] for doc in results[:2]])
            
            prompt = f"""
            ABB Bank bilik bazası cavabı:
            
            ABB Bank:
            - Zəng: 937
            - E-poçt: info@abb-bank.az
            
            Kontekst: {context}
            Sual: {query}
            
            Dəqiq və faydalı cavab ver.
            """
            
            with st.spinner("Cavab hazırlanır..."):
                answer = gemini_api.generate_response(prompt)
                st.write("**Cavab:**")
                st.write(answer)
            
            st.write("**Əlaqəli Sənədlər:**")
            for doc in results[:3]:
                with st.expander(doc['title']):
                    st.write(doc['content'])
        else:
            st.write("Nəticə tapılmadı.")

def main():
    """Əsas tətbiq"""
    initialize_session_state()
    
    # API açarını yüklə
    try:
        if not st.session_state.gemini_api_key:
            st.session_state.gemini_api_key = st.secrets.get("GEMINI_API_KEY", "")
    except:
        pass
    
    # API başlat
    gemini_api = GeminiAPI(st.session_state.gemini_api_key)
    
    # Naviqasiya
    selected_page = sidebar_navigation()
    
    # Səhifə yönləndirmə
    if selected_page == 'Ana Səhifə':
        home_page(gemini_api)
    elif selected_page == 'Şikayətlər':
        complaints_page(gemini_api)
    elif selected_page == 'Kredit Riski':
        credit_risk_page(gemini_api)
    elif selected_page == 'Məhsul Analizi':
        product_insights_page(gemini_api)
    elif selected_page == 'Bilik Bazası':
        knowledge_base_page(gemini_api)

if __name__ == "__main__":
    main()
