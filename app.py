# ============================================================
# IoT IDS - Streamlit Web Uygulamasi
# CICIoT2023 Trained XGBoost Model
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import tempfile
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from fpdf import FPDF
from datetime import datetime
from utils.pcap_parser import extract_features
from xgboost import XGBClassifier

# ============================================================
# SAYFA AYARLARI
# ============================================================

st.set_page_config(
    page_title="IoT Intrusion Detection System",
    page_icon="🛡️",
    layout="wide"
)

# ============================================================
# MODEL YUKLE
# ============================================================

@st.cache_resource
def load_model():
    xgb = XGBClassifier()
    xgb.load_model('model/xgb_model.json')
    scaler  = joblib.load('model/scaler.pkl')
    encoder = joblib.load('model/label_encoder.pkl')
    return xgb, scaler, encoder

model, scaler, le = load_model()

# ============================================================
# YARDIMCI FONKSIYONLAR
# ============================================================

ATTACK_CATEGORIES = {
    'BenignTraffic'          : ('Benign',     '🟢'),
    'DDoS-ICMP_Flood'        : ('DDoS',       '🔴'),
    'DDoS-UDP_Flood'         : ('DDoS',       '🔴'),
    'DDoS-TCP_Flood'         : ('DDoS',       '🔴'),
    'DDoS-PSHACK_Flood'      : ('DDoS',       '🔴'),
    'DDoS-SYN_Flood'         : ('DDoS',       '🔴'),
    'DDoS-RSTFINFlood'       : ('DDoS',       '🔴'),
    'DDoS-SynonymousIP_Flood': ('DDoS',       '🔴'),
    'DDoS-HTTP_Flood'        : ('DDoS',       '🔴'),
    'DDoS-SlowLoris'         : ('DDoS',       '🔴'),
    'DDoS-ACK_Fragmentation' : ('DDoS',       '🔴'),
    'DDoS-UDP_Fragmentation' : ('DDoS',       '🔴'),
    'DDoS-ICMP_Fragmentation': ('DDoS',       '🔴'),
    'DoS-UDP_Flood'          : ('DoS',        '🟠'),
    'DoS-TCP_Flood'          : ('DoS',        '🟠'),
    'DoS-SYN_Flood'          : ('DoS',        '🟠'),
    'DoS-HTTP_Flood'         : ('DoS',        '🟠'),
    'Mirai-greeth_flood'     : ('Mirai',      '🔴'),
    'Mirai-greip_flood'      : ('Mirai',      '🔴'),
    'Mirai-udpplain'         : ('Mirai',      '🔴'),
    'Recon-HostDiscovery'    : ('Recon',      '🟡'),
    'Recon-OSScan'           : ('Recon',      '🟡'),
    'Recon-PortScan'         : ('Recon',      '🟡'),
    'Recon-PingSweep'        : ('Recon',      '🟡'),
    'VulnerabilityScan'      : ('Recon',      '🟡'),
    'MITM-ArpSpoofing'       : ('Spoofing',   '🟠'),
    'DNS_Spoofing'           : ('Spoofing',   '🟠'),
    'DictionaryBruteForce'   : ('BruteForce', '🟠'),
    'BrowserHijacking'       : ('Web-Based',  '🟣'),
    'SqlInjection'           : ('Web-Based',  '🟣'),
    'CommandInjection'       : ('Web-Based',  '🟣'),
    'XSS'                    : ('Web-Based',  '🟣'),
    'Backdoor_Malware'       : ('Web-Based',  '🟣'),
    'Uploading_Attack'       : ('Web-Based',  '🟣'),
}

DEFENSE_RECOMMENDATIONS = {
    'DDoS'      : 'Apply rate limiting, IP blacklisting and traffic scrubbing.',
    'DoS'       : 'Update firewall rules and enable SYN cookies.',
    'Mirai'     : 'Update IoT device firmware and change default credentials.',
    'Recon'     : 'Add IPS rules for port scan detection.',
    'Spoofing'  : 'Enable ARP inspection and DNS filtering.',
    'BruteForce': 'Apply account lockout policy and enforce MFA.',
    'Web-Based' : 'Deploy WAF (Web Application Firewall) and strengthen input validation.',
    'Benign'    : 'Normal traffic. No action required.',
}


def generate_pdf(results_df, summary):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 16)
    pdf.cell(0, 10, 'IoT IDS - Analysis Report', ln=True, align='C')
    pdf.set_font('Helvetica', '', 10)
    pdf.cell(0, 8, f'Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', ln=True)
    pdf.ln(4)

    # Ozet
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 8, 'Summary', ln=True)
    pdf.set_font('Helvetica', '', 10)
    for k, v in summary.items():
        pdf.cell(0, 7, f'  {k}: {v}', ln=True)
    pdf.ln(4)

    # Kategori dagilimi
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 8, 'Attack Category Distribution', ln=True)
    pdf.set_font('Helvetica', '', 10)
    cat_counts = results_df['Category'].value_counts()
    for cat, cnt in cat_counts.items():
        rec = DEFENSE_RECOMMENDATIONS.get(cat, '')
        pdf.cell(0, 7, f'  {cat}: {cnt} packets', ln=True)
        pdf.set_font('Helvetica', 'I', 9)
        pdf.multi_cell(0, 6, f'    -> {rec}')
        pdf.set_font('Helvetica', '', 10)
    pdf.ln(4)

    # Detay tablosu (ilk 50 satir)
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 8, 'Detailed Results (first 50 rows)', ln=True)
    pdf.set_font('Helvetica', 'B', 9)
    pdf.cell(10,  7, '#',           border=1)
    pdf.cell(80,  7, 'Prediction',  border=1)
    pdf.cell(50,  7, 'Category',    border=1)
    pdf.cell(50,  7, 'Confidence',  border=1)
    pdf.ln()
    pdf.set_font('Helvetica', '', 9)
    for i, row in results_df.head(50).iterrows():
        pdf.cell(10,  6, str(i+1),               border=1)
        pdf.cell(80,  6, str(row['Prediction']),  border=1)
        pdf.cell(50,  6, str(row['Category']),    border=1)
        pdf.cell(50,  6, str(row['Confidence']),  border=1)
        pdf.ln()

    path = tempfile.mktemp(suffix='.pdf')
    pdf.output(path)
    return path

# ============================================================
# ARAYUZ
# ============================================================

st.title('🛡️ IoT Intrusion Detection System')
st.markdown('**CICIoT2023 Dataset | XGBoost Model | 34 Attack Classes**')
st.divider()

# Sol - yukle | Sag - sonuclar
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader('📁 Upload PCAP File')
    uploaded = st.file_uploader(
        'Select a .pcap file', 
        type=['pcap', 'pcapng']
    )
    
    window_size = st.slider(
        'Window Size (packets per row)', 
        min_value=10, 
        max_value=200, 
        value=50, 
        step=10,
        help='Number of packets grouped together for feature extraction'
    )
    
    analyze_btn = st.button('🔍 Analyze', use_container_width=True)

with col2:
    if uploaded and analyze_btn:
        
        # Gecici dosyaya yaz
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pcap') as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        
        with st.spinner('Extracting features from pcap...'):
            try:
                df_features = extract_features(tmp_path, window_size=window_size)
                st.success(f'{len(df_features)} windows extracted from {uploaded.name}')
            except Exception as e:
                st.error(f'Feature extraction failed: {e}')
                st.stop()
        
        with st.spinner('Running model...'):
            X_scaled = scaler.transform(df_features)
            preds    = model.predict(X_scaled)
            probs    = model.predict_proba(X_scaled)
            
            pred_labels       = le.inverse_transform(preds)
            confidence_scores = [f'{probs[i][preds[i]]:.2%}' 
                                 for i in range(len(preds))]
            
            categories = [ATTACK_CATEGORIES.get(p, ('Unknown', '⚪'))[0] 
                         for p in pred_labels]
            
            risk_levels = []
            for p in pred_labels:
                cat = ATTACK_CATEGORIES.get(p, ('Unknown', '⚪'))[0]
                if cat == 'Benign':
                    risk_levels.append('LOW')
                elif cat in ('Recon', 'BruteForce'):
                    risk_levels.append('MEDIUM')
                else:
                    risk_levels.append('HIGH')
        
        # Sonuc DataFrame
        results_df = pd.DataFrame({
            'Window'    : range(1, len(pred_labels)+1),
            'Prediction': pred_labels,
            'Category'  : categories,
            'Confidence': confidence_scores,
            'Risk'      : risk_levels
        })
        
        # Ozet metrikler
        total      = len(results_df)
        n_benign   = (results_df['Category'] == 'Benign').sum()
        n_attack   = total - n_benign
        attack_pct = n_attack / total * 100
        
        summary = {
            'Total Windows' : total,
            'Benign'        : n_benign,
            'Attack'        : n_attack,
            'Attack Rate'   : f'{attack_pct:.1f}%',
            'Unique Threats': results_df[results_df['Category'] != 'Benign']['Prediction'].nunique()
        }
        
        # Metrik kutulari
        m1, m2, m3, m4 = st.columns(4)
        m1.metric('Total Windows', total)
        m2.metric('Benign',  n_benign,  delta=None)
        m3.metric('Attack',  n_attack,  delta=None)
        m4.metric('Attack Rate', f'{attack_pct:.1f}%')
        
        st.divider()
        
        # Kategori dagilimi grafigi
        st.subheader('Attack Category Distribution')
        cat_counts = results_df['Category'].value_counts()
        colors = {
            'Benign'    : '#2ecc71',
            'DDoS'      : '#e74c3c',
            'DoS'       : '#e67e22',
            'Mirai'     : '#c0392b',
            'Recon'     : '#f1c40f',
            'Spoofing'  : '#e67e22',
            'BruteForce': '#e67e22',
            'Web-Based' : '#9b59b6',
            'Unknown'   : '#95a5a6',
        }
        bar_colors = [colors.get(c, '#95a5a6') for c in cat_counts.index]
        
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(cat_counts.index, cat_counts.values, color=bar_colors)
        ax.set_xlabel('Window Count')
        ax.set_title('Detected Attack Categories')
        plt.tight_layout()
        st.pyplot(fig)
        
        st.divider()
        
        # Defense onerileri
        st.subheader('Defense Recommendations')
        detected_cats = results_df['Category'].unique()
        for cat in detected_cats:
            rec = DEFENSE_RECOMMENDATIONS.get(cat, 'No recommendation available.')
            emoji = colors.get(cat, '⚪')
            if cat == 'Benign':
                st.success(f'**{cat}:** {rec}')
            else:
                st.warning(f'**{cat}:** {rec}')
        
        st.divider()
        
        # Detay tablosu
        st.subheader('Detailed Results')
        st.dataframe(results_df, use_container_width=True)
        
        # PDF indir
        st.divider()
        pdf_path = generate_pdf(results_df, summary)
        with open(pdf_path, 'rb') as f:
            st.download_button(
                label='📄 Download PDF Report',
                data=f,
                file_name=f'iot_ids_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf',
                mime='application/pdf',
                use_container_width=True
            )
        
        os.unlink(tmp_path)
    
    elif not uploaded:
        st.info('Upload a .pcap file to start analysis.')