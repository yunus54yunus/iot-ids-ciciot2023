\# IoT Intrusion Detection System

\### CICIoT2023 Dataset | XGBoost | Streamlit



A machine learning-based intrusion detection system (IDS) for IoT networks, trained on the CICIoT2023 dataset. The system detects 33 attack types across 7 categories from network traffic (.pcap files).



\## Results



| Metric | Value |

|--------|-------|

| Accuracy | 0.92 |

| Macro F1-Score | 0.88 |

| Weighted F1-Score | 0.92 |

| Training Time | \~23 seconds (GPU T4) |

| Attack Classes | 34 (33 attacks + benign) |



\## Attack Categories Detected



| Category | Examples |

|----------|---------|

| DDoS | ICMP Flood, SYN Flood, UDP Flood, SlowLoris |

| DoS | TCP Flood, HTTP Flood, SYN Flood |

| Mirai | Greeth Flood, Greip Flood, UDPPlain |

| Recon | Port Scan, OS Scan, Host Discovery |

| Spoofing | ARP Spoofing, DNS Spoofing |

| Brute Force | Dictionary Attack |

| Web-Based | XSS, SQL Injection, Command Injection, Backdoor |



\## Dataset



\*\*CICIoT2023\*\* — Canadian Institute for Cybersecurity

\- 105 real IoT devices

\- 33 attack types

\- 46 network flow features

\- \~46M total records



Stratified sampling applied: 10,000 samples per class, 297,091 total records.



\## How It Works



PCAP File → Feature Extraction (Scapy) → StandardScaler → XGBoost → Results



1\. Upload a `.pcap` file

2\. Scapy parses packets into 46-feature windows

3\. XGBoost classifies each window

4\. Results shown with attack categories and defense recommendations

5\. PDF report download available



\## Installation



```bash

git clone https://github.com/yunus54yunus/iot-ids-ciciot2023.git

cd iot-ids-ciciot2023

pip install -r requirements.txt

streamlit run app.py

```



\## Project Structure



```

iot-ids-ciciot2023/

├── app.py                  # Streamlit web application

├── requirements.txt

├── model/

│   ├── xgb\_model.json      # Trained XGBoost model

│   ├── scaler.pkl          # StandardScaler

│   └── label\_encoder.pkl   # LabelEncoder

└── utils/

&#x20;   └── pcap\_parser.py      # PCAP feature extraction

```



\## Limitations



\- PCAP feature mapping is approximate — not all 46 CICIoT2023 features can be directly extracted from raw packets

\- Confidence scores may be lower than dataset evaluation metrics

\- Web-based attack classes (XSS, SQLi) show lower F1 scores due to limited training samples



\## References



\- Neto et al., "CICIoT2023: A real-time dataset and benchmark for large-scale attacks in IoT environment", Sensors, 2023

\- Chen \& Guestrin, "XGBoost: A Scalable Tree Boosting System", KDD 2016

