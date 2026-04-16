# ============================================================
# PCAP PARSER - CICIoT2023 Feature Extraction
# ============================================================
# CICIoT2023 dataset'indeki 46 feature'i pcap dosyalarindan
# elimizden geldigince extract ediyoruz.
# Bazi feature'lar pcap'tan direkt cikarilabilir,
# bazi statistical feature'lar window-based hesaplanir.
# ============================================================

from scapy.all import rdpcap, IP, TCP, UDP, ICMP, ARP, DNS
import pandas as pd
import numpy as np


# CICIoT2023'un tam feature listesi (model bu siraya gore egitildi)
FEATURE_COLUMNS = [
    'flow_duration', 'Header_Length', 'Protocol Type', 'Duration',
    'Rate', 'Srate', 'Drate', 'fin_flag_number', 'syn_flag_number',
    'rst_flag_number', 'psh_flag_number', 'ack_flag_number',
    'ece_flag_number', 'cwr_flag_number', 'ack_count', 'syn_count',
    'fin_count', 'urg_count', 'rst_count', 'HTTP', 'HTTPS', 'DNS',
    'Telnet', 'SMTP', 'SSH', 'IRC', 'TCP', 'UDP', 'DHCP', 'ARP',
    'ICMP', 'IPv', 'LLC', 'Tot sum', 'Min', 'Max', 'AVG', 'Std',
    'Tot size', 'IAT', 'Number', 'Magnitue', 'Radius', 'Covariance',
    'Variance', 'Weight'
]


def extract_features(pcap_path, window_size=50):
    """
    Pcap dosyasindan CICIoT2023 feature formatinda DataFrame uretir.
    
    - Paketler window_size'lik gruplara bolunur
    - Her gruptan istatistiksel feature'lar hesaplanir
    - Eksik feature'lar 0 ile doldurulur (mapping limitation)
    
    Args:
        pcap_path: .pcap dosyasinin yolu
        window_size: kac paketten bir satir uretilecegi
    
    Returns:
        pd.DataFrame: model'e girebilecek formatta feature matrisi
    """
    
    try:
        packets = rdpcap(pcap_path)
    except Exception as e:
        raise ValueError(f"Pcap okunamadi: {e}")
    
    if len(packets) == 0:
        raise ValueError("Pcap dosyasi bos.")
    
    rows = []
    
    # Paketleri window'lara bol
    for i in range(0, len(packets), window_size):
        window = packets[i:i+window_size]
        row = _extract_window_features(window)
        rows.append(row)
    
    df = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    return df


def _extract_window_features(window):
    """
    Bir paket grubundan feature vektoru cikarir.
    """
    
    pkt_lengths = []
    iats = []
    prev_time = None
    
    # Flag sayaclari
    fin_count = syn_count = rst_count = psh_count = 0
    ack_count = ece_count = cwr_count = urg_count = 0
    
    # Protokol sayaclari
    tcp_count = udp_count = icmp_count = arp_count = 0
    dns_count = http_count = https_count = ssh_count = 0
    telnet_count = smtp_count = irc_count = dhcp_count = 0
    ipv_count = llc_count = 0
    
    header_lengths = []
    
    for pkt in window:
        # Paket boyutu
        pkt_len = len(pkt)
        pkt_lengths.append(pkt_len)
        
        # IAT (inter-arrival time)
        curr_time = float(pkt.time)
        if prev_time is not None:
            iats.append(curr_time - prev_time)
        prev_time = curr_time
        
        # IP layer
        if pkt.haslayer(IP):
            ipv_count += 1
            header_lengths.append(pkt[IP].ihl * 4)
        
        # TCP flags
        if pkt.haslayer(TCP):
            tcp_count += 1
            flags = pkt[TCP].flags
            if flags & 0x01: fin_count += 1
            if flags & 0x02: syn_count += 1
            if flags & 0x04: rst_count += 1
            if flags & 0x08: psh_count += 1
            if flags & 0x10: ack_count += 1
            if flags & 0x20: urg_count += 1
            if flags & 0x40: ece_count += 1
            if flags & 0x80: cwr_count += 1
            
            # Port bazli protokol tespiti
            dport = pkt[TCP].dport
            sport = pkt[TCP].sport
            if 80 in (dport, sport):   http_count += 1
            if 443 in (dport, sport):  https_count += 1
            if 22 in (dport, sport):   ssh_count += 1
            if 23 in (dport, sport):   telnet_count += 1
            if 25 in (dport, sport):   smtp_count += 1
            if 194 in (dport, sport):  irc_count += 1
        
        # UDP
        if pkt.haslayer(UDP):
            udp_count += 1
            dport = pkt[UDP].dport
            sport = pkt[UDP].sport
            if 53 in (dport, sport):   dns_count += 1
            if 67 in (dport, sport):   dhcp_count += 1
            if 68 in (dport, sport):   dhcp_count += 1
        
        # ICMP
        if pkt.haslayer(ICMP):
            icmp_count += 1
        
        # ARP
        if pkt.haslayer(ARP):
            arp_count += 1
    
    # Istatistiksel hesaplamalar
    n = len(window)
    lengths = np.array(pkt_lengths) if pkt_lengths else np.array([0])
    iat_arr = np.array(iats) if iats else np.array([0])
    
    tot_sum  = float(np.sum(lengths))
    min_len  = float(np.min(lengths))
    max_len  = float(np.max(lengths))
    avg_len  = float(np.mean(lengths))
    std_len  = float(np.std(lengths))
    var_len  = float(np.var(lengths))
    iat_mean = float(np.mean(iat_arr))
    
    # Flow duration
    if len(window) > 1:
        flow_dur = float(window[-1].time - window[0].time)
    else:
        flow_dur = 0.0
    
    # Rate (paket/saniye)
    rate  = n / flow_dur if flow_dur > 0 else 0.0
    srate = tcp_count / flow_dur if flow_dur > 0 else 0.0
    drate = udp_count / flow_dur if flow_dur > 0 else 0.0
    
    # Header length ortalama
    header_len = float(np.mean(header_lengths)) if header_lengths else 0.0
    
    # Magnitude, Radius, Covariance, Weight (istatistiksel)
    magnitude  = float(np.sqrt(np.sum(lengths**2)))
    radius     = float(np.sqrt(np.sum((lengths - avg_len)**2) / n))
    covariance = float(np.cov(lengths, lengths)[0][1]) if n > 1 else 0.0
    weight     = float(n)
    
    # Normalize flag sayilari (oran olarak)
    fin_flag = fin_count / n
    syn_flag = syn_count / n
    rst_flag = rst_count / n
    psh_flag = psh_count / n
    ack_flag = ack_count / n
    ece_flag = ece_count / n
    cwr_flag = cwr_count / n
    
    row = [
        flow_dur, header_len, tcp_count + udp_count + icmp_count, flow_dur,
        rate, srate, drate,
        fin_flag, syn_flag, rst_flag, psh_flag, ack_flag, ece_flag, cwr_flag,
        ack_count, syn_count, fin_count, urg_count, rst_count,
        http_count, https_count, dns_count, telnet_count, smtp_count,
        ssh_count, irc_count, tcp_count, udp_count, dhcp_count,
        arp_count, icmp_count, ipv_count, llc_count,
        tot_sum, min_len, max_len, avg_len, std_len,
        tot_sum, iat_mean, n,
        magnitude, radius, covariance, var_len, weight
    ]
    
    return row