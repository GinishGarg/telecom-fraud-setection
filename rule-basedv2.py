"""
Rule-Based Fraud Detection Agent for Telecom Datasets
Separate detectors for IPDR (Internet Session) and CDR (Call Detail Records)
With comprehensive accuracy evaluation and performance metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from collections import defaultdict
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    roc_curve, auc, precision_recall_curve,
    f1_score, precision_score, recall_score, accuracy_score
)
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

class IPDRFraudDetector:
    """Rule-Based Fraud Detector for IPDR Data"""
    
    def __init__(self, duration_threshold=7200, byte_spike_std=3, 
                 cell_hop_window=3600, cell_hop_count=5):
        """
        Initialize IPDR detector with configurable thresholds
        
        Args:
            duration_threshold: Sessions longer than this (seconds) are flagged
            byte_spike_std: Standard deviations for byte spike detection
            cell_hop_window: Time window for cell tower hopping (seconds)
            cell_hop_count: Number of hops to flag as suspicious
        """
        self.duration_threshold = duration_threshold
        self.byte_spike_std = byte_spike_std
        self.cell_hop_window = cell_hop_window
        self.cell_hop_count = cell_hop_count
        
        # High-risk domains and ports
        self.high_risk_domains = [
            'tor', 'onion', 'darknet', 'proxy', 'anonymizer', 
            'malware', 'phishing', 'spam', 'botnet'
        ]
        self.suspicious_ports = [
            22, 23, 25, 445, 1433, 3306, 3389, 5900, 8080, 
            31337, 12345, 54321  # Known malware ports
        ]
        
    def detect_anomalies(self, df):
        """
        Apply all IPDR fraud detection rules
        
        Args:
            df: DataFrame with IPDR data
            
        Returns:
            DataFrame with anomaly flags and reasons
        """
        print("🔍 Starting IPDR Anomaly Detection...")
        
        # Create output dataframe
        result_df = df.copy()
        result_df['anomaly_flag'] = False
        result_df['anomaly_reason'] = ''
        result_df['anomaly_score'] = 0
        
        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            result_df['timestamp'] = pd.to_datetime(result_df['timestamp'])
        
        # Rule 1: VPN Usage Detection
        print("  ✓ Checking VPN usage patterns...")
        vpn_mask = result_df['vpn_usage'] == True
        result_df.loc[vpn_mask, 'anomaly_flag'] = True
        result_df.loc[vpn_mask, 'anomaly_reason'] += 'VPN_USAGE; '
        result_df.loc[vpn_mask, 'anomaly_score'] += 2
        
        # Rule 2: Extreme Duration (Very Long Sessions)
        print("  ✓ Detecting abnormal session durations...")
        long_duration = result_df['duration'] > self.duration_threshold
        result_df.loc[long_duration, 'anomaly_flag'] = True
        result_df.loc[long_duration, 'anomaly_reason'] += 'LONG_SESSION; '
        result_df.loc[long_duration, 'anomaly_score'] += 3
        
        # Very short sessions (potential scanning)
        short_duration = result_df['duration'] < 10
        result_df.loc[short_duration, 'anomaly_flag'] = True
        result_df.loc[short_duration, 'anomaly_reason'] += 'SHORT_SESSION; '
        result_df.loc[short_duration, 'anomaly_score'] += 1
        
        # Rule 3: Byte Spike Detection
        print("  ✓ Analyzing data transfer spikes...")
        if result_df['bytes_sent'].std() > 0:
            byte_sent_threshold = result_df['bytes_sent'].mean() + \
                                  self.byte_spike_std * result_df['bytes_sent'].std()
            byte_spike_sent = result_df['bytes_sent'] > byte_sent_threshold
            result_df.loc[byte_spike_sent, 'anomaly_flag'] = True
            result_df.loc[byte_spike_sent, 'anomaly_reason'] += 'HIGH_BYTES_SENT; '
            result_df.loc[byte_spike_sent, 'anomaly_score'] += 4
        
        if result_df['bytes_received'].std() > 0:
            byte_recv_threshold = result_df['bytes_received'].mean() + \
                                  self.byte_spike_std * result_df['bytes_received'].std()
            byte_spike_recv = result_df['bytes_received'] > byte_recv_threshold
            result_df.loc[byte_spike_recv, 'anomaly_flag'] = True
            result_df.loc[byte_spike_recv, 'anomaly_reason'] += 'HIGH_BYTES_RECEIVED; '
            result_df.loc[byte_spike_recv, 'anomaly_score'] += 4
        
        # Rule 4: High-Risk Domain Detection
        print("  ✓ Scanning for high-risk domains...")
        if 'domain' in result_df.columns:
            risk_domain_mask = result_df['domain'].str.lower().apply(
                lambda x: any(risk in str(x).lower() for risk in self.high_risk_domains)
            )
            result_df.loc[risk_domain_mask, 'anomaly_flag'] = True
            result_df.loc[risk_domain_mask, 'anomaly_reason'] += 'HIGH_RISK_DOMAIN; '
            result_df.loc[risk_domain_mask, 'anomaly_score'] += 5
        
        # Rule 5: Suspicious Port Usage
        print("  ✓ Identifying suspicious port activity...")
        if 'port' in result_df.columns:
            susp_port_mask = result_df['port'].isin(self.suspicious_ports)
            result_df.loc[susp_port_mask, 'anomaly_flag'] = True
            result_df.loc[susp_port_mask, 'anomaly_reason'] += 'SUSPICIOUS_PORT; '
            result_df.loc[susp_port_mask, 'anomaly_score'] += 4
        
        # Rule 6: Rapid Cell Tower Hopping
        print("  ✓ Detecting cell tower hopping...")
        if 'user_id' in result_df.columns and 'cell_id' in result_df.columns:
            result_df = result_df.sort_values(['user_id', 'timestamp'])
            
            for user in result_df['user_id'].unique():
                user_mask = result_df['user_id'] == user
                user_data = result_df[user_mask].copy()
                
                if len(user_data) < self.cell_hop_count:
                    continue
                
                # Check cell changes within time window
                user_data['cell_change'] = user_data['cell_id'] != user_data['cell_id'].shift(1)
                user_data['time_diff'] = user_data['timestamp'].diff().dt.total_seconds()
                
                # Count changes in rolling window
                for idx in user_data.index:
                    window_start = user_data.loc[idx, 'timestamp'] - timedelta(seconds=self.cell_hop_window)
                    window_data = user_data[
                        (user_data['timestamp'] >= window_start) & 
                        (user_data['timestamp'] <= user_data.loc[idx, 'timestamp'])
                    ]
                    
                    if window_data['cell_change'].sum() >= self.cell_hop_count:
                        result_df.loc[idx, 'anomaly_flag'] = True
                        result_df.loc[idx, 'anomaly_reason'] += 'CELL_TOWER_HOPPING; '
                        result_df.loc[idx, 'anomaly_score'] += 3
        
        # Rule 7: Repeated IP Pair Connections (Persistence)
        print("  ✓ Checking for suspicious IP persistence...")
        if 'ip_src' in result_df.columns and 'ip_dst' in result_df.columns:
            result_df['ip_pair'] = result_df['ip_src'].astype(str) + '_' + result_df['ip_dst'].astype(str)
            ip_pair_counts = result_df.groupby(['user_id', 'ip_pair']).size()
            suspicious_pairs = ip_pair_counts[ip_pair_counts > 10].index
            
            for user, ip_pair in suspicious_pairs:
                mask = (result_df['user_id'] == user) & (result_df['ip_pair'] == ip_pair)
                result_df.loc[mask, 'anomaly_flag'] = True
                result_df.loc[mask, 'anomaly_reason'] += 'REPEATED_IP_PAIR; '
                result_df.loc[mask, 'anomaly_score'] += 2
        
        # Rule 8: Unusual Data Transfer Ratio
        print("  ✓ Analyzing upload/download ratios...")
        result_df['transfer_ratio'] = result_df['bytes_sent'] / (result_df['bytes_received'] + 1)
        high_upload = result_df['transfer_ratio'] > 10  # Much more upload than download
        result_df.loc[high_upload, 'anomaly_flag'] = True
        result_df.loc[high_upload, 'anomaly_reason'] += 'HIGH_UPLOAD_RATIO; '
        result_df.loc[high_upload, 'anomaly_score'] += 3
        
        # Clean up anomaly reasons
        result_df['anomaly_reason'] = result_df['anomaly_reason'].str.rstrip('; ')
        
        # Summary statistics
        total_events = len(result_df)
        flagged_events = result_df['anomaly_flag'].sum()
        print(f"\n📊 IPDR Detection Summary:")
        print(f"   Total Events: {total_events:,}")
        print(f"   Flagged Events: {flagged_events:,} ({flagged_events/total_events*100:.2f}%)")
        
        return result_df


class CDRFraudDetector:
    """Rule-Based Fraud Detector for CDR Data"""
    
    def __init__(self, sim_swap_window=24, sim_swap_threshold=2,
                 vishing_duration=300, vishing_count=5, 
                 short_call_duration=10, short_call_count=10):
        """
        Initialize CDR detector with configurable thresholds
        
        Args:
            sim_swap_window: Hours to check for SIM swap
            sim_swap_threshold: Number of different IMEIs to flag
            vishing_duration: Call duration threshold for vishing (seconds)
            vishing_count: Number of calls to same number
            short_call_duration: Threshold for short calls
            short_call_count: Number of short calls to flag
        """
        self.sim_swap_window = sim_swap_window
        self.sim_swap_threshold = sim_swap_threshold
        self.vishing_duration = vishing_duration
        self.vishing_count = vishing_count
        self.short_call_duration = short_call_duration
        self.short_call_count = short_call_count
        
    def detect_anomalies(self, df):
        """
        Apply all CDR fraud detection rules
        
        Args:
            df: DataFrame with CDR data
            
        Returns:
            DataFrame with anomaly flags and reasons
        """
        print("🔍 Starting CDR Anomaly Detection...")
        
        # Create output dataframe
        result_df = df.copy()
        result_df['anomaly_flag'] = False
        result_df['anomaly_reason'] = ''
        result_df['anomaly_score'] = 0
        
        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            result_df['timestamp'] = pd.to_datetime(result_df['timestamp'])
        
        # Rule 1: SIM Swap Detection
        print("  ✓ Detecting SIM swap patterns...")
        result_df = result_df.sort_values(['imsi', 'timestamp'])
        
        for imsi in result_df['imsi'].unique():
            imsi_mask = result_df['imsi'] == imsi
            imsi_data = result_df[imsi_mask].copy()
            
            # Check for multiple IMEIs in time window
            for idx in imsi_data.index:
                window_start = imsi_data.loc[idx, 'timestamp'] - timedelta(hours=self.sim_swap_window)
                window_data = imsi_data[
                    (imsi_data['timestamp'] >= window_start) & 
                    (imsi_data['timestamp'] <= imsi_data.loc[idx, 'timestamp'])
                ]
                
                unique_imeis = window_data['imei'].nunique()
                if unique_imeis > self.sim_swap_threshold:
                    result_df.loc[idx, 'anomaly_flag'] = True
                    result_df.loc[idx, 'anomaly_reason'] += f'SIM_SWAP({unique_imeis}_DEVICES); '
                    result_df.loc[idx, 'anomaly_score'] += 8
        
        # Rule 2: Vishing Call Detection (Long Repeated Calls)
        print("  ✓ Identifying vishing patterns...")
        if 'ip_dst' in result_df.columns:  # Using ip_dst as callee identifier
            long_calls = result_df[result_df['duration'] > self.vishing_duration]
            call_counts = long_calls.groupby(['user_id', 'ip_dst']).size()
            vishing_pairs = call_counts[call_counts >= self.vishing_count].index
            
            for user, callee in vishing_pairs:
                mask = (result_df['user_id'] == user) & (result_df['ip_dst'] == callee)
                result_df.loc[mask, 'anomaly_flag'] = True
                result_df.loc[mask, 'anomaly_reason'] += 'VISHING_PATTERN; '
                result_df.loc[mask, 'anomaly_score'] += 6
        
        # Rule 3: Short Duration Repeated Calls (Wangiri Scam)
        print("  ✓ Detecting short-duration call patterns...")
        short_calls = result_df[result_df['duration'] < self.short_call_duration]
        short_call_counts = short_calls.groupby(['user_id', 'ip_dst']).size()
        wangiri_pairs = short_call_counts[short_call_counts >= self.short_call_count].index
        
        for user, callee in wangiri_pairs:
            mask = (result_df['user_id'] == user) & (result_df['ip_dst'] == callee)
            result_df.loc[mask, 'anomaly_flag'] = True
            result_df.loc[mask, 'anomaly_reason'] += 'WANGIRI_SCAM; '
            result_df.loc[mask, 'anomaly_score'] += 5
        
        # Rule 4: Night-time Calling Patterns
        print("  ✓ Analyzing night-time activity...")
        if 'hour' in result_df.columns:
            night_calls = (result_df['hour'] >= 0) & (result_df['hour'] <= 5)
        else:
            result_df['hour'] = result_df['timestamp'].dt.hour
            night_calls = (result_df['hour'] >= 0) & (result_df['hour'] <= 5)
        
        # Count night calls per user
        night_call_counts = result_df[night_calls].groupby('user_id').size()
        suspicious_night_users = night_call_counts[night_call_counts > 10].index
        
        for user in suspicious_night_users:
            mask = (result_df['user_id'] == user) & night_calls
            result_df.loc[mask, 'anomaly_flag'] = True
            result_df.loc[mask, 'anomaly_reason'] += 'NIGHT_CALLING; '
            result_df.loc[mask, 'anomaly_score'] += 3
        
        # Rule 5: Device Change Anomaly (IMEI shifts)
        print("  ✓ Tracking device changes...")
        result_df = result_df.sort_values(['user_id', 'timestamp'])
        result_df['imei_change'] = result_df.groupby('user_id')['imei'].shift(1) != result_df['imei']
        
        # Flag frequent device changes
        device_changes = result_df.groupby('user_id')['imei_change'].sum()
        frequent_changers = device_changes[device_changes > 5].index
        
        for user in frequent_changers:
            mask = (result_df['user_id'] == user) & result_df['imei_change']
            result_df.loc[mask, 'anomaly_flag'] = True
            result_df.loc[mask, 'anomaly_reason'] += 'FREQUENT_DEVICE_CHANGE; '
            result_df.loc[mask, 'anomaly_score'] += 4
        
        # Rule 6: High Call Volume
        print("  ✓ Detecting high call volumes...")
        call_volumes = result_df.groupby('user_id').size()
        high_volume_threshold = call_volumes.mean() + 3 * call_volumes.std()
        high_volume_users = call_volumes[call_volumes > high_volume_threshold].index
        
        for user in high_volume_users:
            mask = result_df['user_id'] == user
            result_df.loc[mask, 'anomaly_flag'] = True
            result_df.loc[mask, 'anomaly_reason'] += 'HIGH_CALL_VOLUME; '
            result_df.loc[mask, 'anomaly_score'] += 2
        
        # Rule 7: Weekend/Holiday Activity Spike
        print("  ✓ Checking weekend activity patterns...")
        if 'is_weekend' in result_df.columns:
            weekend_activity = result_df[result_df['is_weekend'] == 1].groupby('user_id').size()
            weekday_activity = result_df[result_df['is_weekend'] == 0].groupby('user_id').size()
            
            for user in weekend_activity.index:
                if user in weekday_activity.index:
                    ratio = weekend_activity[user] / (weekday_activity[user] + 1)
                    if ratio > 3:  # Much more active on weekends
                        mask = (result_df['user_id'] == user) & (result_df['is_weekend'] == 1)
                        result_df.loc[mask, 'anomaly_flag'] = True
                        result_df.loc[mask, 'anomaly_reason'] += 'WEEKEND_SPIKE; '
                        result_df.loc[mask, 'anomaly_score'] += 2
        
        # Rule 8: Roaming Abuse (Location jumps)
        print("  ✓ Detecting roaming abuse...")
        if 'location_lat' in result_df.columns and 'location_lon' in result_df.columns:
            result_df = result_df.sort_values(['user_id', 'timestamp'])
            
            for user in result_df['user_id'].unique():
                user_mask = result_df['user_id'] == user
                user_data = result_df[user_mask].copy()
                
                if len(user_data) < 2:
                    continue
                
                # Calculate distance between consecutive locations (simplified)
                user_data['lat_diff'] = user_data['location_lat'].diff().abs()
                user_data['lon_diff'] = user_data['location_lon'].diff().abs()
                user_data['time_diff'] = user_data['timestamp'].diff().dt.total_seconds()
                
                # Flag impossible travel (large distance in short time)
                impossible_travel = (
                    ((user_data['lat_diff'] > 5) | (user_data['lon_diff'] > 5)) & 
                    (user_data['time_diff'] < 3600)
                )
                
                for idx in user_data[impossible_travel].index:
                    result_df.loc[idx, 'anomaly_flag'] = True
                    result_df.loc[idx, 'anomaly_reason'] += 'ROAMING_ABUSE; '
                    result_df.loc[idx, 'anomaly_score'] += 7
        
        # Clean up anomaly reasons
        result_df['anomaly_reason'] = result_df['anomaly_reason'].str.rstrip('; ')
        
        # Summary statistics
        total_calls = len(result_df)
        flagged_calls = result_df['anomaly_flag'].sum()
        print(f"\n📊 CDR Detection Summary:")
        print(f"   Total Calls: {total_calls:,}")
        print(f"   Flagged Calls: {flagged_calls:,} ({flagged_calls/total_calls*100:.2f}%)")
        
        return result_df


def visualize_ipdr_results(df, output_prefix='ipdr'):
    """Generate comprehensive visualizations for IPDR analysis"""
    print("\n📈 Generating IPDR visualizations...")
    
    # ========== FIGURE 1: Main Overview (6 plots) ==========
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Anomaly Count by Type
    plt.subplot(2, 3, 1)
    anomaly_reasons = df[df['anomaly_flag']]['anomaly_reason'].str.split('; ', expand=True).stack()
    anomaly_counts = anomaly_reasons.value_counts()
    anomaly_counts.plot(kind='barh', color='coral')
    plt.title('IPDR: Anomaly Types Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Count')
    plt.ylabel('Anomaly Type')
    
    # 2. Anomaly Distribution Pie Chart
    plt.subplot(2, 3, 2)
    labels = ['Anomalous', 'Normal']
    sizes = [df['anomaly_flag'].sum(), (~df['anomaly_flag']).sum()]
    colors = ['#ff6b6b', '#51cf66']
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    plt.title('IPDR: Overall Anomaly Distribution', fontsize=14, fontweight='bold')
    
    # 3. Anomalies Over Time
    if 'timestamp' in df.columns:
        plt.subplot(2, 3, 3)
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        hourly_anomalies = df[df['anomaly_flag']].groupby('hour').size()
        hourly_anomalies.plot(kind='line', marker='o', color='red', linewidth=2)
        plt.title('IPDR: Anomalies by Hour', fontsize=14, fontweight='bold')
        plt.xlabel('Hour of Day')
        plt.ylabel('Anomaly Count')
        plt.grid(True, alpha=0.3)
    
    # 4. Anomaly Score Distribution
    plt.subplot(2, 3, 4)
    df[df['anomaly_flag']]['anomaly_score'].hist(bins=20, color='skyblue', edgecolor='black')
    plt.title('IPDR: Anomaly Score Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Frequency')
    
    # 5. Bytes Sent vs Received (Anomalous vs Normal)
    plt.subplot(2, 3, 5)
    normal = df[~df['anomaly_flag']].sample(min(1000, len(df[~df['anomaly_flag']])))
    anomalous = df[df['anomaly_flag']].sample(min(1000, len(df[df['anomaly_flag']])))
    plt.scatter(normal['bytes_sent'], normal['bytes_received'], alpha=0.3, label='Normal', s=10)
    plt.scatter(anomalous['bytes_sent'], anomalous['bytes_received'], 
                alpha=0.6, label='Anomalous', s=20, color='red')
    plt.xlabel('Bytes Sent')
    plt.ylabel('Bytes Received')
    plt.title('IPDR: Data Transfer Patterns', fontsize=14, fontweight='bold')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    
    # 6. VPN Usage Analysis
    plt.subplot(2, 3, 6)
    vpn_stats = df.groupby('vpn_usage')['anomaly_flag'].value_counts().unstack(fill_value=0)
    vpn_stats.plot(kind='bar', stacked=True, color=['#51cf66', '#ff6b6b'])
    plt.title('IPDR: VPN Usage vs Anomalies', fontsize=14, fontweight='bold')
    plt.xlabel('VPN Usage')
    plt.ylabel('Count')
    plt.legend(['Normal', 'Anomalous'])
    plt.xticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_analysis_1.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {output_prefix}_analysis_1.png")
    plt.close()
    
    # ========== FIGURE 2: Protocol and Port Analysis (6 plots) ==========
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Protocol Distribution
    plt.subplot(2, 3, 1)
    if 'protocol' in df.columns:
        protocol_anomalies = df.groupby('protocol')['anomaly_flag'].sum().sort_values(ascending=False).head(10)
        protocol_anomalies.plot(kind='bar', color='indianred')
        plt.title('IPDR: Top Protocols with Anomalies', fontsize=14, fontweight='bold')
        plt.xlabel('Protocol')
        plt.ylabel('Anomaly Count')
        plt.xticks(rotation=45)
    
    # 2. Port Analysis
    plt.subplot(2, 3, 2)
    if 'port' in df.columns:
        port_anomalies = df[df['anomaly_flag']]['port'].value_counts().head(15)
        port_anomalies.plot(kind='barh', color='darkred')
        plt.title('IPDR: Top Anomalous Ports', fontsize=14, fontweight='bold')
        plt.xlabel('Count')
        plt.ylabel('Port')
    
    # 3. Duration Distribution Comparison
    plt.subplot(2, 3, 3)
    plt.hist([df[~df['anomaly_flag']]['duration'], df[df['anomaly_flag']]['duration']], 
             bins=50, label=['Normal', 'Anomalous'], color=['green', 'red'], alpha=0.6)
    plt.title('IPDR: Session Duration Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.yscale('log')
    
    # 4. Bytes Per Second Analysis
    plt.subplot(2, 3, 4)
    df['bps'] = df['bytes_sent'] / (df['duration'] + 1)
    df['bps_anomaly'] = df['bps'] > df['bps'].quantile(0.95)
    df.boxplot(column='bps', by='anomaly_flag', ax=plt.gca())
    plt.title('IPDR: Bytes Per Second by Anomaly Status', fontsize=14, fontweight='bold')
    plt.xlabel('Anomaly Flag')
    plt.ylabel('Bytes Per Second')
    plt.yscale('log')
    plt.suptitle('')
    
    # 5. Cell ID Distribution
    plt.subplot(2, 3, 5)
    if 'cell_id' in df.columns:
        top_cells = df[df['anomaly_flag']]['cell_id'].value_counts().head(15)
        top_cells.plot(kind='barh', color='orange')
        plt.title('IPDR: Top Cell Towers with Anomalies', fontsize=14, fontweight='bold')
        plt.xlabel('Anomaly Count')
        plt.ylabel('Cell ID')
    
    # 6. Domain Analysis
    plt.subplot(2, 3, 6)
    if 'domain' in df.columns:
        domain_stats = df[df['anomaly_flag']]['domain'].value_counts().head(10)
        domain_stats.plot(kind='bar', color='purple')
        plt.title('IPDR: Top Anomalous Domains', fontsize=14, fontweight='bold')
        plt.xlabel('Domain')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_analysis_2.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {output_prefix}_analysis_2.png")
    plt.close()
    
    # ========== FIGURE 3: User Behavior and Temporal Analysis (6 plots) ==========
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Top Anomalous Users
    plt.subplot(2, 3, 1)
    top_users = df[df['anomaly_flag']].groupby('user_id').size().sort_values(ascending=False).head(20)
    top_users.plot(kind='barh', color='crimson')
    plt.title('IPDR: Top 20 Users with Anomalies', fontsize=14, fontweight='bold')
    plt.xlabel('Anomaly Count')
    plt.ylabel('User ID')
    
    # 2. Anomaly Score by User (Box Plot)
    plt.subplot(2, 3, 2)
    top_users_list = df[df['anomaly_flag']].groupby('user_id')['anomaly_score'].sum().sort_values(ascending=False).head(10).index
    df[df['user_id'].isin(top_users_list)].boxplot(column='anomaly_score', by='user_id', ax=plt.gca(), rot=45)
    plt.title('IPDR: Anomaly Score Distribution by Top Users', fontsize=14, fontweight='bold')
    plt.xlabel('User ID')
    plt.ylabel('Anomaly Score')
    plt.suptitle('')
    
    # 3. Heatmap: Anomalies by Hour and Day of Week
    plt.subplot(2, 3, 3)
    if 'timestamp' in df.columns:
        df['dayofweek'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        heatmap_data = df[df['anomaly_flag']].groupby(['dayofweek', 'hour']).size().unstack(fill_value=0)
        sns.heatmap(heatmap_data, cmap='Reds', annot=False, fmt='d', cbar_kws={'label': 'Count'})
        plt.title('IPDR: Anomaly Heatmap (Day vs Hour)', fontsize=14, fontweight='bold')
        plt.xlabel('Hour of Day')
        plt.ylabel('Day of Week (0=Mon)')
    
    # 4. Cumulative Anomalies Over Time
    plt.subplot(2, 3, 4)
    if 'timestamp' in df.columns:
        df_sorted = df[df['anomaly_flag']].sort_values('timestamp')
        df_sorted['cumulative'] = range(1, len(df_sorted) + 1)
        plt.plot(df_sorted['timestamp'], df_sorted['cumulative'], color='darkred', linewidth=2)
        plt.title('IPDR: Cumulative Anomalies Over Time', fontsize=14, fontweight='bold')
        plt.xlabel('Timestamp')
        plt.ylabel('Cumulative Anomaly Count')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
    
    # 5. Transfer Ratio Distribution
    plt.subplot(2, 3, 5)
    if 'transfer_ratio' in df.columns:
        df_filtered = df[df['transfer_ratio'] < 100]  # Filter outliers for visibility
        plt.hist([df_filtered[~df_filtered['anomaly_flag']]['transfer_ratio'], 
                  df_filtered[df_filtered['anomaly_flag']]['transfer_ratio']], 
                 bins=50, label=['Normal', 'Anomalous'], color=['green', 'red'], alpha=0.6)
        plt.title('IPDR: Upload/Download Ratio Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Upload/Download Ratio')
        plt.ylabel('Frequency')
        plt.legend()
        plt.yscale('log')
    
    # 6. Location-based Analysis
    plt.subplot(2, 3, 6)
    if 'location_lat' in df.columns and 'location_lon' in df.columns:
        normal_sample = df[~df['anomaly_flag']].sample(min(500, len(df[~df['anomaly_flag']])))
        anomaly_sample = df[df['anomaly_flag']].sample(min(500, len(df[df['anomaly_flag']])))
        plt.scatter(normal_sample['location_lon'], normal_sample['location_lat'], 
                   alpha=0.3, s=10, label='Normal', c='green')
        plt.scatter(anomaly_sample['location_lon'], anomaly_sample['location_lat'], 
                   alpha=0.6, s=30, label='Anomalous', c='red', marker='x')
        plt.title('IPDR: Geographic Distribution of Anomalies', fontsize=14, fontweight='bold')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_analysis_3.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {output_prefix}_analysis_3.png")
    plt.close()
    
    # ========== FIGURE 4: Advanced Statistical Analysis (6 plots) ==========
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Anomaly Score vs Duration
    plt.subplot(2, 3, 1)
    anomaly_data = df[df['anomaly_flag']]
    plt.scatter(anomaly_data['duration'], anomaly_data['anomaly_score'], alpha=0.5, c='red', s=20)
    plt.title('IPDR: Anomaly Score vs Session Duration', fontsize=14, fontweight='bold')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Anomaly Score')
    plt.xscale('log')
    
    # 2. Multiple Anomaly Types per Event
    plt.subplot(2, 3, 2)
    df['num_anomaly_types'] = df['anomaly_reason'].apply(lambda x: len(str(x).split('; ')) if pd.notna(x) and x != '' else 0)
    anomaly_type_counts = df[df['anomaly_flag']]['num_anomaly_types'].value_counts().sort_index()
    anomaly_type_counts.plot(kind='bar', color='darkorange')
    plt.title('IPDR: Number of Anomaly Types per Event', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Anomaly Types')
    plt.ylabel('Event Count')
    plt.xticks(rotation=0)
    
    # 3. Anomaly Correlation Matrix
    plt.subplot(2, 3, 3)
    if len(df[df['anomaly_flag']]) > 0:
        corr_features = ['duration', 'bytes_sent', 'bytes_received', 'anomaly_score']
        corr_features = [f for f in corr_features if f in df.columns]
        if len(corr_features) > 1:
            corr_matrix = df[df['anomaly_flag']][corr_features].corr()
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)
            plt.title('IPDR: Feature Correlation (Anomalous Events)', fontsize=14, fontweight='bold')
    
    # 4. Time Between Anomalies (per User)
    plt.subplot(2, 3, 4)
    if 'timestamp' in df.columns:
        df_sorted = df[df['anomaly_flag']].sort_values(['user_id', 'timestamp'])
        df_sorted['time_between'] = df_sorted.groupby('user_id')['timestamp'].diff().dt.total_seconds()
        time_between = df_sorted['time_between'].dropna()
        if len(time_between) > 0:
            plt.hist(time_between, bins=50, color='teal', edgecolor='black')
            plt.title('IPDR: Time Between Consecutive Anomalies', fontsize=14, fontweight='bold')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Frequency')
            plt.xscale('log')
    
    # 5. Anomaly Distribution by IMSI/IMEI
    plt.subplot(2, 3, 5)
    if 'imsi' in df.columns:
        imsi_anomalies = df[df['anomaly_flag']].groupby('imsi').size().sort_values(ascending=False).head(15)
        imsi_anomalies.plot(kind='barh', color='darkviolet')
        plt.title('IPDR: Top IMSIs with Anomalies', fontsize=14, fontweight='bold')
        plt.xlabel('Anomaly Count')
        plt.ylabel('IMSI')
    
    # 6. Rolling Average of Anomalies
    plt.subplot(2, 3, 6)
    if 'timestamp' in df.columns:
        df_sorted = df.sort_values('timestamp')
        df_sorted['anomaly_int'] = df_sorted['anomaly_flag'].astype(int)
        df_sorted['rolling_anomaly'] = df_sorted['anomaly_int'].rolling(window=100, min_periods=1).mean()
        plt.plot(df_sorted['timestamp'], df_sorted['rolling_anomaly'], color='darkred', linewidth=2)
        plt.title('IPDR: Rolling Anomaly Rate (Window=100)', fontsize=14, fontweight='bold')
        plt.xlabel('Timestamp')
        plt.ylabel('Anomaly Rate')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_analysis_4.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {output_prefix}_analysis_4.png")
    plt.close()


def evaluate_model_performance(df, ground_truth_col='is_fraud', prediction_col='anomaly_flag', 
                              score_col='anomaly_score', output_prefix='model'):
    """
    Comprehensive evaluation of fraud detection model performance
    
    Args:
        df: DataFrame with predictions and ground truth
        ground_truth_col: Column name for actual fraud labels
        prediction_col: Column name for model predictions
        score_col: Column name for anomaly scores
        output_prefix: Prefix for output files
    
    Returns:
        Dictionary with all performance metrics
    """
    print(f"\n{'='*70}")
    print(f"🎯 EVALUATING {output_prefix.upper()} MODEL PERFORMANCE")
    print(f"{'='*70}")
    
    # Check if ground truth exists
    if ground_truth_col not in df.columns:
        print(f"⚠️  Warning: Ground truth column '{ground_truth_col}' not found in dataset")
        print("   Skipping accuracy evaluation. Cannot validate without labels.")
        return None
    
    # Extract predictions and ground truth
    y_true = df[ground_truth_col].astype(int)
    y_pred = df[prediction_col].astype(int)
    
    # Handle anomaly scores (might not exist for all rows)
    if score_col in df.columns:
        y_score = df[score_col].fillna(0)
    else:
        y_score = y_pred.astype(float)
    
    # ========== BASIC METRICS ==========
    print("\n📊 CLASSIFICATION METRICS:")
    print("-" * 70)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Precision: {precision:.4f} (of flagged events, {precision*100:.2f}% are truly fraudulent)")
    print(f"   Recall:    {recall:.4f} (detected {recall*100:.2f}% of all fraud cases)")
    print(f"   F1-Score:  {f1:.4f} (harmonic mean of precision and recall)")
    
    # ========== CONFUSION MATRIX ==========
    print("\n📋 CONFUSION MATRIX:")
    print("-" * 70)
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"   True Negatives (TN):  {tn:,} (Correctly identified normal)")
    print(f"   False Positives (FP): {fp:,} (Normal flagged as fraud)")
    print(f"   False Negatives (FN): {fn:,} (Fraud missed by detector)")
    print(f"   True Positives (TP):  {tp:,} (Correctly identified fraud)")
    
    # ========== ERROR ANALYSIS ==========
    print("\n🔍 ERROR ANALYSIS:")
    print("-" * 70)
    
    total_fraud = y_true.sum()
    total_normal = len(y_true) - total_fraud
    
    false_positive_rate = fp / total_normal if total_normal > 0 else 0
    false_negative_rate = fn / total_fraud if total_fraud > 0 else 0
    
    print(f"   False Positive Rate: {false_positive_rate:.4f} ({false_positive_rate*100:.2f}% of normal flagged)")
    print(f"   False Negative Rate: {false_negative_rate:.4f} ({false_negative_rate*100:.2f}% of fraud missed)")
    print(f"   Specificity (TNR):   {tn/total_normal if total_normal > 0 else 0:.4f}")
    print(f"   Sensitivity (TPR):   {recall:.4f}")
    
    # ========== DETAILED CLASSIFICATION REPORT ==========
    print("\n📈 DETAILED CLASSIFICATION REPORT:")
    print("-" * 70)
    print(classification_report(y_true, y_pred, 
                               target_names=['Normal', 'Fraudulent'],
                               digits=4))
    
    # ========== VISUALIZATIONS ==========
    print("\n🎨 Generating performance visualizations...")
    
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Confusion Matrix Heatmap
    plt.subplot(2, 3, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Fraud'],
                yticklabels=['Normal', 'Fraud'])
    plt.title(f'{output_prefix.upper()}: Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    # 2. Normalized Confusion Matrix
    plt.subplot(2, 3, 2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='RdYlGn', 
                xticklabels=['Normal', 'Fraud'],
                yticklabels=['Normal', 'Fraud'])
    plt.title(f'{output_prefix.upper()}: Normalized Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    # 3. ROC Curve
    plt.subplot(2, 3, 3)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title(f'{output_prefix.upper()}: ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    # 4. Precision-Recall Curve
    plt.subplot(2, 3, 4)
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall_curve, precision_curve)
    
    plt.plot(recall_curve, precision_curve, color='blue', lw=2, 
             label=f'PR curve (AUC = {pr_auc:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{output_prefix.upper()}: Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    
    # 5. Metric Comparison Bar Chart
    plt.subplot(2, 3, 5)
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [accuracy, precision, recall, f1]
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    bars = plt.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.3f}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.ylim([0, 1.1])
    plt.title(f'{output_prefix.upper()}: Performance Metrics', fontsize=14, fontweight='bold')
    plt.ylabel('Score')
    plt.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Good (0.8+)')
    plt.axhline(y=0.6, color='orange', linestyle='--', alpha=0.5, label='Fair (0.6+)')
    plt.legend()
    
    # 6. Score Distribution by Class
    plt.subplot(2, 3, 6)
    fraud_scores = y_score[y_true == 1]
    normal_scores = y_score[y_true == 0]
    
    plt.hist(normal_scores, bins=30, alpha=0.6, label='Normal', color='green', edgecolor='black')
    plt.hist(fraud_scores, bins=30, alpha=0.6, label='Fraudulent', color='red', edgecolor='black')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Frequency')
    plt.title(f'{output_prefix.upper()}: Score Distribution by Class', fontsize=14, fontweight='bold')
    plt.legend()
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_performance_metrics.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {output_prefix}_performance_metrics.png")
    plt.close()
    
    # ========== THRESHOLD ANALYSIS ==========
    print("\n⚙️  THRESHOLD OPTIMIZATION:")
    print("-" * 70)
    
    # Find optimal threshold based on F1-score
    thresholds_to_test = np.linspace(y_score.min(), y_score.max(), 100)
    f1_scores = []
    
    for thresh in thresholds_to_test:
        y_pred_thresh = (y_score >= thresh).astype(int)
        f1_scores.append(f1_score(y_true, y_pred_thresh, zero_division=0))
    
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds_to_test[optimal_idx]
    optimal_f1 = f1_scores[optimal_idx]
    
    print(f"   Current Threshold: Binary (0/1)")
    print(f"   Optimal Threshold: {optimal_threshold:.2f}")
    print(f"   Optimal F1-Score: {optimal_f1:.4f}")
    print(f"   Improvement: {(optimal_f1 - f1)*100:.2f}%")
    
    # ========== ERROR CASE ANALYSIS ==========
    print("\n❌ FALSE POSITIVE ANALYSIS:")
    print("-" * 70)
    
    fp_mask = (y_true == 0) & (y_pred == 1)
    if fp_mask.sum() > 0:
        fp_data = df[fp_mask]
        print(f"   Total False Positives: {fp_mask.sum():,}")
        
        if 'anomaly_reason' in fp_data.columns:
            fp_reasons = fp_data['anomaly_reason'].str.split('; ', expand=True).stack().value_counts()
            print(f"\n   Top reasons for false positives:")
            for reason, count in fp_reasons.head(5).items():
                print(f"     - {reason}: {count:,} ({count/fp_mask.sum()*100:.1f}%)")
    
    print("\n❌ FALSE NEGATIVE ANALYSIS:")
    print("-" * 70)
    
    fn_mask = (y_true == 1) & (y_pred == 0)
    if fn_mask.sum() > 0:
        fn_data = df[fn_mask]
        print(f"   Total False Negatives: {fn_mask.sum():,}")
        print(f"   Missed fraud cases: {fn_mask.sum()/y_true.sum()*100:.2f}% of all fraud")
        
        if 'anomaly_type' in fn_data.columns:
            fn_types = fn_data['anomaly_type'].value_counts()
            print(f"\n   Types of fraud missed:")
            for fraud_type, count in fn_types.head(5).items():
                print(f"     - {fraud_type}: {count:,}")
    
    # ========== PERFORMANCE BY ANOMALY TYPE ==========
    if 'anomaly_reason' in df.columns and 'anomaly_type' in df.columns:
        print("\n📊 PERFORMANCE BY FRAUD TYPE:")
        print("-" * 70)
        
        # Get unique fraud types from ground truth
        fraud_types = df[df[ground_truth_col] == 1]['anomaly_type'].value_counts()
        
        for fraud_type in fraud_types.head(5).index:
            if pd.isna(fraud_type):
                continue
            
            type_mask = df['anomaly_type'] == fraud_type
            if type_mask.sum() == 0:
                continue
            
            y_true_type = df[type_mask][ground_truth_col].astype(int)
            y_pred_type = df[type_mask][prediction_col].astype(int)
            
            if len(y_true_type) > 0:
                type_precision = precision_score(y_true_type, y_pred_type, zero_division=0)
                type_recall = recall_score(y_true_type, y_pred_type, zero_division=0)
                type_f1 = f1_score(y_true_type, y_pred_type, zero_division=0)
                
                print(f"\n   {fraud_type}:")
                print(f"     Cases: {type_mask.sum():,}")
                print(f"     Precision: {type_precision:.3f} | Recall: {type_recall:.3f} | F1: {type_f1:.3f}")
    
    # ========== RETURN METRICS DICTIONARY ==========
    metrics_dict = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp),
        'false_positive_rate': false_positive_rate,
        'false_negative_rate': false_negative_rate,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'optimal_threshold': optimal_threshold,
        'optimal_f1': optimal_f1,
        'confusion_matrix': cm
    }
    
    # Save metrics to file
    metrics_df = pd.DataFrame([metrics_dict])
    metrics_df.to_csv(f'{output_prefix}_performance_metrics.csv', index=False)
    print(f"\n✓ Saved metrics to: {output_prefix}_performance_metrics.csv")
    
    return metrics_dict


def compare_models(ipdr_metrics, cdr_metrics):
    """Compare performance between IPDR and CDR models"""
    print("\n" + "="*70)
    print("⚖️  MODEL COMPARISON: IPDR vs CDR")
    print("="*70)
    
    if ipdr_metrics is None or cdr_metrics is None:
        print("⚠️  Cannot compare - one or both models lack ground truth labels")
        return
    
    comparison_data = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC'],
        'IPDR': [
            ipdr_metrics['accuracy'],
            ipdr_metrics['precision'],
            ipdr_metrics['recall'],
            ipdr_metrics['f1_score'],
            ipdr_metrics['roc_auc']
        ],
        'CDR': [
            cdr_metrics['accuracy'],
            cdr_metrics['precision'],
            cdr_metrics['recall'],
            cdr_metrics['f1_score'],
            cdr_metrics['roc_auc']
        ]
    }
    
    comp_df = pd.DataFrame(comparison_data)
    comp_df['Difference'] = comp_df['IPDR'] - comp_df['CDR']
    comp_df['Better Model'] = comp_df['Difference'].apply(lambda x: 'IPDR' if x > 0 else ('CDR' if x < 0 else 'Equal'))
    
    print("\n" + comp_df.to_string(index=False))
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Side-by-side comparison
    x = np.arange(len(comparison_data['Metric']))
    width = 0.35
    
    axes[0].bar(x - width/2, comparison_data['IPDR'], width, label='IPDR', alpha=0.8, color='coral')
    axes[0].bar(x + width/2, comparison_data['CDR'], width, label='CDR', alpha=0.8, color='salmon')
    axes[0].set_xlabel('Metrics')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(comparison_data['Metric'], rotation=45)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1.1])
    
    # Error comparison
    error_data = {
        'False Positives': [ipdr_metrics['false_positives'], cdr_metrics['false_positives']],
        'False Negatives': [ipdr_metrics['false_negatives'], cdr_metrics['false_negatives']]
    }
    error_df = pd.DataFrame(error_data, index=['IPDR', 'CDR'])
    error_df.plot(kind='bar', ax=axes[1], color=['orange', 'red'], alpha=0.7)
    axes[1].set_title('Error Comparison', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Count')
    axes[1].set_xticklabels(['IPDR', 'CDR'], rotation=0)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved comparison visualization: model_comparison.png")
    plt.close()
    
    comp_df.to_csv('model_comparison.csv', index=False)
    print("✓ Saved comparison data: model_comparison.csv")
    """Generate comprehensive visualizations for CDR analysis"""
    print("\n📈 Generating CDR visualizations...")
    
    # ========== FIGURE 1: Main Overview (6 plots) ==========
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Anomaly Count by Type
    plt.subplot(2, 3, 1)
    anomaly_reasons = df[df['anomaly_flag']]['anomaly_reason'].str.split('; ', expand=True).stack()
    anomaly_counts = anomaly_reasons.value_counts()
    anomaly_counts.plot(kind='barh', color='salmon')
    plt.title('CDR: Anomaly Types Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Count')
    plt.ylabel('Anomaly Type')
    
    # 2. Anomaly Distribution Pie Chart
    plt.subplot(2, 3, 2)
    labels = ['Anomalous', 'Normal']
    sizes = [df['anomaly_flag'].sum(), (~df['anomaly_flag']).sum()]
    colors = ['#ff6b6b', '#51cf66']
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    plt.title('CDR: Overall Anomaly Distribution', fontsize=14, fontweight='bold')
    
    # 3. Anomalies Over Time
    if 'timestamp' in df.columns:
        plt.subplot(2, 3, 3)
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        hourly_anomalies = df[df['anomaly_flag']].groupby('hour').size()
        hourly_anomalies.plot(kind='line', marker='o', color='darkred', linewidth=2)
        plt.title('CDR: Anomalies by Hour', fontsize=14, fontweight='bold')
        plt.xlabel('Hour of Day')
        plt.ylabel('Anomaly Count')
        plt.grid(True, alpha=0.3)
    
    # 4. SIM Swap Activity
    plt.subplot(2, 3, 4)
    sim_swap_data = df[df['anomaly_reason'].str.contains('SIM_SWAP', na=False)]
    if len(sim_swap_data) > 0:
        sim_swap_daily = sim_swap_data.groupby(pd.to_datetime(sim_swap_data['timestamp']).dt.date).size()
        sim_swap_daily.plot(kind='bar', color='crimson')
        plt.title('CDR: SIM Swap Activity Over Time', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('SIM Swap Events')
        plt.xticks(rotation=45)
    else:
        plt.text(0.5, 0.5, 'No SIM Swap Events Detected', 
                ha='center', va='center', fontsize=12)
        plt.title('CDR: SIM Swap Activity', fontsize=14, fontweight='bold')
    
    # 5. Call Duration Distribution
    plt.subplot(2, 3, 5)
    df[df['anomaly_flag']]['duration'].hist(bins=30, alpha=0.7, label='Anomalous', color='red')
    df[~df['anomaly_flag']]['duration'].hist(bins=30, alpha=0.5, label='Normal', color='green')
    plt.title('CDR: Call Duration Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.yscale('log')
    
    # 6. Device Changes Per User
    plt.subplot(2, 3, 6)
    device_changes = df.groupby('user_id')['imei'].nunique()
    device_changes.hist(bins=20, color='purple', edgecolor='black')
    plt.title('CDR: Device Changes per User', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Unique IMEIs')
    plt.ylabel('Number of Users')
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_analysis_1.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {output_prefix}_analysis_1.png")
    plt.close()
    
    # ========== FIGURE 2: SIM Swap and Fraud Patterns (6 plots) ==========
    fig = plt.figure(figsize=(20, 12))
    
    # 1. SIM Swap Heatmap by Hour and Day
    plt.subplot(2, 3, 1)
    if 'timestamp' in df.columns:
        sim_swap_data = df[df['anomaly_reason'].str.contains('SIM_SWAP', na=False)]
        if len(sim_swap_data) > 0:
            sim_swap_data['dayofweek'] = pd.to_datetime(sim_swap_data['timestamp']).dt.dayofweek
            sim_swap_data['hour'] = pd.to_datetime(sim_swap_data['timestamp']).dt.hour
            heatmap_data = sim_swap_data.groupby(['dayofweek', 'hour']).size().unstack(fill_value=0)
            sns.heatmap(heatmap_data, cmap='Reds', annot=False, cbar_kws={'label': 'Count'})
            plt.title('CDR: SIM Swap Heatmap (Day vs Hour)', fontsize=14, fontweight='bold')
            plt.xlabel('Hour of Day')
            plt.ylabel('Day of Week (0=Mon)')
        else:
            plt.text(0.5, 0.5, 'No SIM Swap Data', ha='center', va='center')
            plt.title('CDR: SIM Swap Heatmap', fontsize=14, fontweight='bold')
    
    # 2. Vishing Pattern Analysis
    plt.subplot(2, 3, 2)
    vishing_data = df[df['anomaly_reason'].str.contains('VISHING', na=False)]
    if len(vishing_data) > 0:
        vishing_duration = vishing_data['duration']
        plt.hist(vishing_duration, bins=30, color='orangered', edgecolor='black')
        plt.title('CDR: Vishing Call Duration Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Duration (seconds)')
        plt.ylabel('Count')
    else:
        plt.text(0.5, 0.5, 'No Vishing Patterns Detected', ha='center', va='center')
        plt.title('CDR: Vishing Analysis', fontsize=14, fontweight='bold')
    
    # 3. Wangiri Scam Detection
    plt.subplot(2, 3, 3)
    wangiri_data = df[df['anomaly_reason'].str.contains('WANGIRI', na=False)]
    if len(wangiri_data) > 0:
        wangiri_users = wangiri_data.groupby('user_id').size().sort_values(ascending=False).head(15)
        wangiri_users.plot(kind='barh', color='darkred')
        plt.title('CDR: Top Users with Wangiri Activity', fontsize=14, fontweight='bold')
        plt.xlabel('Wangiri Call Count')
        plt.ylabel('User ID')
    else:
        plt.text(0.5, 0.5, 'No Wangiri Patterns Detected', ha='center', va='center')
        plt.title('CDR: Wangiri Analysis', fontsize=14, fontweight='bold')
    
    # 4. Night Calling Patterns
    plt.subplot(2, 3, 4)
    night_data = df[df['anomaly_reason'].str.contains('NIGHT_CALLING', na=False)]
    if len(night_data) > 0:
        night_hourly = night_data.groupby('hour').size()
        night_hourly.plot(kind='bar', color='midnightblue')
        plt.title('CDR: Night Calling Distribution by Hour', fontsize=14, fontweight='bold')
        plt.xlabel('Hour')
        plt.ylabel('Call Count')
        plt.xticks(rotation=0)
    else:
        plt.text(0.5, 0.5, 'No Night Calling Patterns', ha='center', va='center')
        plt.title('CDR: Night Calling Analysis', fontsize=14, fontweight='bold')
    
    # 5. Device Change Timeline
    plt.subplot(2, 3, 5)
    if 'imei_change' in df.columns:
        device_change_data = df[df['imei_change'] == True]
        if len(device_change_data) > 0 and 'timestamp' in df.columns:
            device_change_daily = device_change_data.groupby(pd.to_datetime(device_change_data['timestamp']).dt.date).size()
            device_change_daily.plot(kind='line', marker='o', color='purple', linewidth=2)
            plt.title('CDR: Device Changes Over Time', fontsize=14, fontweight='bold')
            plt.xlabel('Date')
            plt.ylabel('Device Changes')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
    
    # 6. Roaming Abuse Detection
    plt.subplot(2, 3, 6)
    roaming_data = df[df['anomaly_reason'].str.contains('ROAMING_ABUSE', na=False)]
    if len(roaming_data) > 0:
        roaming_users = roaming_data.groupby('user_id').size().sort_values(ascending=False).head(15)
        roaming_users.plot(kind='barh', color='teal')
        plt.title('CDR: Top Users with Roaming Abuse', fontsize=14, fontweight='bold')
        plt.xlabel('Roaming Abuse Events')
        plt.ylabel('User ID')
    else:
        plt.text(0.5, 0.5, 'No Roaming Abuse Detected', ha='center', va='center')
        plt.title('CDR: Roaming Abuse Analysis', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_analysis_2.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {output_prefix}_analysis_2.png")
    plt.close()
    
    # ========== FIGURE 3: User Behavior and Temporal Analysis (6 plots) ==========
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Top Anomalous Users
    plt.subplot(2, 3, 1)
    top_users = df[df['anomaly_flag']].groupby('user_id').size().sort_values(ascending=False).head(20)
    top_users.plot(kind='barh', color='crimson')
    plt.title('CDR: Top 20 Users with Anomalies', fontsize=14, fontweight='bold')
    plt.xlabel('Anomaly Count')
    plt.ylabel('User ID')
    
    # 2. Anomaly Score Distribution
    plt.subplot(2, 3, 2)
    df[df['anomaly_flag']]['anomaly_score'].hist(bins=25, color='coral', edgecolor='black')
    plt.title('CDR: Anomaly Score Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Frequency')
    
    # 3. Call Volume Analysis
    plt.subplot(2, 3, 3)
    call_volumes = df.groupby('user_id').size()
    plt.hist([call_volumes[call_volumes.index.isin(df[df['anomaly_flag']]['user_id'])],
              call_volumes[~call_volumes.index.isin(df[df['anomaly_flag']]['user_id'])]],
             bins=30, label=['Users with Anomalies', 'Normal Users'], 
             color=['red', 'green'], alpha=0.6)
    plt.title('CDR: Call Volume Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Calls')
    plt.ylabel('Number of Users')
    plt.legend()
    plt.yscale('log')
    
    # 4. Weekend vs Weekday Activity
    plt.subplot(2, 3, 4)
    if 'is_weekend' in df.columns:
        weekend_anomalies = df[df['is_weekend'] == 1]['anomaly_flag'].value_counts()
        weekday_anomalies = df[df['is_weekend'] == 0]['anomaly_flag'].value_counts()
        
        x = ['Weekday', 'Weekend']
        normal = [weekday_anomalies.get(False, 0), weekend_anomalies.get(False, 0)]
        anomalous = [weekday_anomalies.get(True, 0), weekend_anomalies.get(True, 0)]
        
        width = 0.35
        x_pos = np.arange(len(x))
        plt.bar(x_pos - width/2, normal, width, label='Normal', color='green')
        plt.bar(x_pos + width/2, anomalous, width, label='Anomalous', color='red')
        plt.xticks(x_pos, x)
        plt.title('CDR: Weekend vs Weekday Anomalies', fontsize=14, fontweight='bold')
        plt.ylabel('Count')
        plt.legend()
    
    # 5. Cumulative Anomalies Over Time
    plt.subplot(2, 3, 5)
    if 'timestamp' in df.columns:
        df_sorted = df[df['anomaly_flag']].sort_values('timestamp')
        df_sorted['cumulative'] = range(1, len(df_sorted) + 1)
        plt.plot(df_sorted['timestamp'], df_sorted['cumulative'], color='darkred', linewidth=2)
        plt.title('CDR: Cumulative Anomalies Over Time', fontsize=14, fontweight='bold')
        plt.xlabel('Timestamp')
        plt.ylabel('Cumulative Anomaly Count')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
    
    # 6. IMSI-IMEI Relationship
    plt.subplot(2, 3, 6)
    imsi_imei_counts = df.groupby('imsi')['imei'].nunique().sort_values(ascending=False).head(20)
    imsi_imei_counts.plot(kind='barh', color='indigo')
    plt.title('CDR: Top IMSIs by Device Count', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Unique IMEIs')
    plt.ylabel('IMSI')
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_analysis_3.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {output_prefix}_analysis_3.png")
    plt.close()
    
    # ========== FIGURE 4: Advanced Analysis (6 plots) ==========
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Multiple Anomaly Types per Call
    plt.subplot(2, 3, 1)
    df['num_anomaly_types'] = df['anomaly_reason'].apply(lambda x: len(str(x).split('; ')) if pd.notna(x) and x != '' else 0)
    anomaly_type_counts = df[df['anomaly_flag']]['num_anomaly_types'].value_counts().sort_index()
    anomaly_type_counts.plot(kind='bar', color='darkorange')
    plt.title('CDR: Number of Anomaly Types per Call', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Anomaly Types')
    plt.ylabel('Call Count')
    plt.xticks(rotation=0)
    
    # 2. Anomaly Heatmap: Day vs Hour
    plt.subplot(2, 3, 2)
    if 'timestamp' in df.columns:
        df['dayofweek'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        heatmap_data = df[df['anomaly_flag']].groupby(['dayofweek', 'hour']).size().unstack(fill_value=0)
        sns.heatmap(heatmap_data, cmap='YlOrRd', annot=False, fmt='d', cbar_kws={'label': 'Count'})
        plt.title('CDR: Anomaly Heatmap (Day vs Hour)', fontsize=14, fontweight='bold')
        plt.xlabel('Hour of Day')
        plt.ylabel('Day of Week (0=Mon)')
    
    # 3. Call Pattern by Top Destination
    plt.subplot(2, 3, 3)
    if 'ip_dst' in df.columns:
        top_destinations = df[df['anomaly_flag']]['ip_dst'].value_counts().head(15)
        top_destinations.plot(kind='barh', color='steelblue')
        plt.title('CDR: Top Destinations in Anomalous Calls', fontsize=14, fontweight='bold')
        plt.xlabel('Call Count')
        plt.ylabel('Destination')
    
    # 4. Time Between Anomalies
    plt.subplot(2, 3, 4)
    if 'timestamp' in df.columns:
        df_sorted = df[df['anomaly_flag']].sort_values(['user_id', 'timestamp'])
        df_sorted['time_between'] = df_sorted.groupby('user_id')['timestamp'].diff().dt.total_seconds()
        time_between = df_sorted['time_between'].dropna()
        if len(time_between) > 0:
            plt.hist(time_between, bins=50, color='maroon', edgecolor='black')
            plt.title('CDR: Time Between Consecutive Anomalies', fontsize=14, fontweight='bold')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Frequency')
            plt.xscale('log')
    
    # 5. Anomaly Score vs Call Duration
    plt.subplot(2, 3, 5)
    anomaly_data = df[df['anomaly_flag']]
    plt.scatter(anomaly_data['duration'], anomaly_data['anomaly_score'], alpha=0.5, c='red', s=20)
    plt.title('CDR: Anomaly Score vs Call Duration', fontsize=14, fontweight='bold')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Anomaly Score')
    plt.xscale('log')
    
    # 6. Geographic Distribution (if available)
    plt.subplot(2, 3, 6)
    if 'location_lat' in df.columns and 'location_lon' in df.columns:
        normal_sample = df[~df['anomaly_flag']].sample(min(500, len(df[~df['anomaly_flag']])))
        anomaly_sample = df[df['anomaly_flag']].sample(min(500, len(df[df['anomaly_flag']])))
        plt.scatter(normal_sample['location_lon'], normal_sample['location_lat'], 
                   alpha=0.3, s=10, label='Normal', c='green')
        plt.scatter(anomaly_sample['location_lon'], anomaly_sample['location_lat'], 
                   alpha=0.6, s=30, label='Anomalous', c='red', marker='x')
        plt.title('CDR: Geographic Distribution of Anomalies', fontsize=14, fontweight='bold')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.legend()
    else:
        plt.text(0.5, 0.5, 'Location Data Not Available', ha='center', va='center')
        plt.title('CDR: Geographic Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_analysis_4.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {output_prefix}_analysis_4.png")
    plt.close()


def main():
    """Main execution function"""
    print("="*70)
    print("🚀 TELECOM FRAUD DETECTION AGENT")
    print("="*70)
    
    # File paths (modify these to your actual file paths)
    ipdr_file = 'ipdr.csv'
    cdr_file = 'cdrv2.csv'
    
    # ========== IPDR PROCESSING ==========
    print("\n" + "="*70)
    print("📡 PROCESSING IPDR DATA")
    print("="*70)
    
    try:
        ipdr_df = pd.read_csv(ipdr_file)
        print(f"✓ Loaded IPDR data: {len(ipdr_df):,} records")
        print(f"  Columns: {list(ipdr_df.columns)}")
        
        # Initialize and run IPDR detector
        ipdr_detector = IPDRFraudDetector(
            duration_threshold=7200,  # 2 hours
            byte_spike_std=3,
            cell_hop_window=3600,  # 1 hour
            cell_hop_count=5
        )
        
        ipdr_results = ipdr_detector.detect_anomalies(ipdr_df)
        
        # Save results
        ipdr_results.to_csv('ipdr_rules_output.csv', index=False)
        print(f"\n✓ Saved: ipdr_rules_output.csv")
        
        # Generate visualizations
        visualize_ipdr_results(ipdr_results, 'ipdr')
        
        # Evaluate model performance if ground truth exists
        ipdr_metrics = evaluate_model_performance(
            ipdr_results, 
            ground_truth_col='is_fraud',
            prediction_col='anomaly_flag',
            score_col='anomaly_score',
            output_prefix='ipdr'
        )
        
        # Print detailed statistics
        print("\n📊 IPDR Detailed Statistics:")
        print(f"   Total Events: {len(ipdr_results):,}")
        print(f"   Flagged Events: {ipdr_results['anomaly_flag'].sum():,}")
        print(f"   Unique Users Affected: {ipdr_results[ipdr_results['anomaly_flag']]['user_id'].nunique():,}")
        print(f"\n   Top Anomaly Types:")
        anomaly_reasons = ipdr_results[ipdr_results['anomaly_flag']]['anomaly_reason'].str.split('; ', expand=True).stack()
        for reason, count in anomaly_reasons.value_counts().head(10).items():
            print(f"     - {reason}: {count:,}")
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find {ipdr_file}")
        print("   Please ensure the IPDR CSV file is in the current directory")
    except Exception as e:
        print(f"❌ Error processing IPDR data: {str(e)}")
    
    # ========== CDR PROCESSING ==========
    print("\n" + "="*70)
    print("📞 PROCESSING CDR DATA")
    print("="*70)
    
    try:
        cdr_df = pd.read_csv(cdr_file)
        print(f"✓ Loaded CDR data: {len(cdr_df):,} records")
        print(f"  Columns: {list(cdr_df.columns)}")
        
        # Initialize and run CDR detector
        cdr_detector = CDRFraudDetector(
            sim_swap_window=24,  # 24 hours
            sim_swap_threshold=2,  # 2+ different devices
            vishing_duration=300,  # 5 minutes
            vishing_count=5,
            short_call_duration=10,  # 10 seconds
            short_call_count=10
        )
        
        cdr_results = cdr_detector.detect_anomalies(cdr_df)
        
        # Save results
        cdr_results.to_csv('cdr_rules_output.csv', index=False)
        print(f"\n✓ Saved: cdr_rules_output.csv")
        
        # Generate visualizations
        visualize_cdr_results(cdr_results, 'cdr')
        
        # Evaluate model performance if ground truth exists
        cdr_metrics = evaluate_model_performance(
            cdr_results,
            ground_truth_col='is_fraud',
            prediction_col='anomaly_flag',
            score_col='anomaly_score',
            output_prefix='cdr'
        )
        
        # Print detailed statistics
        print("\n📊 CDR Detailed Statistics:")
        print(f"   Total Calls: {len(cdr_results):,}")
        print(f"   Flagged Calls: {cdr_results['anomaly_flag'].sum():,}")
        print(f"   Unique Users Affected: {cdr_results[cdr_results['anomaly_flag']]['user_id'].nunique():,}")
        print(f"\n   Top Anomaly Types:")
        anomaly_reasons = cdr_results[cdr_results['anomaly_flag']]['anomaly_reason'].str.split('; ', expand=True).stack()
        for reason, count in anomaly_reasons.value_counts().head(10).items():
            print(f"     - {reason}: {count:,}")
        
        # SIM Swap specific analysis
        sim_swap_events = cdr_results[cdr_results['anomaly_reason'].str.contains('SIM_SWAP', na=False)]
        if len(sim_swap_events) > 0:
            print(f"\n⚠️  SIM Swap Analysis:")
            print(f"   Total SIM Swap Events: {len(sim_swap_events):,}")
            print(f"   Affected IMSIs: {sim_swap_events['imsi'].nunique():,}")
            print(f"   Average Anomaly Score: {sim_swap_events['anomaly_score'].mean():.2f}")
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find {cdr_file}")
        print("   Please ensure the CDR CSV file is in the current directory")
        cdr_metrics = None
    except Exception as e:
        print(f"❌ Error processing CDR data: {str(e)}")
        cdr_metrics = None
    
    # ========== MODEL COMPARISON ==========
    if 'ipdr_metrics' in locals() and 'cdr_metrics' in locals():
        compare_models(ipdr_metrics, cdr_metrics)
    
    # ========== FINAL SUMMARY ==========
    print("\n" + "="*70)
    print("✅ FRAUD DETECTION COMPLETE")
    print("="*70)
    print("\n📁 Output Files Generated:")
    print("   CSV Results:")
    print("   - ipdr_rules_output.csv (IPDR anomaly results)")
    print("   - cdr_rules_output.csv (CDR anomaly results)")
    print("\n   IPDR Visualizations (24 plots total):")
    print("   - ipdr_analysis_1.png (Overview: 6 plots)")
    print("   - ipdr_analysis_2.png (Protocol & Port Analysis: 6 plots)")
    print("   - ipdr_analysis_3.png (User Behavior & Temporal: 6 plots)")
    print("   - ipdr_analysis_4.png (Advanced Statistics: 6 plots)")
    print("\n   CDR Visualizations (24 plots total):")
    print("   - cdr_analysis_1.png (Overview: 6 plots)")
    print("   - cdr_analysis_2.png (Fraud Patterns: 6 plots)")
    print("   - cdr_analysis_3.png (User Behavior & Temporal: 6 plots)")
    print("   - cdr_analysis_4.png (Advanced Analysis: 6 plots)")
    print("\n   Performance Evaluation (if ground truth available):")
    print("   - ipdr_performance_metrics.png (6 performance plots)")
    print("   - cdr_performance_metrics.png (6 performance plots)")
    print("   - ipdr_performance_metrics.csv (detailed metrics)")
    print("   - cdr_performance_metrics.csv (detailed metrics)")
    print("   - model_comparison.png (IPDR vs CDR comparison)")
    print("   - model_comparison.csv (comparison data)")
    print("\n💡 Notes on False Positives/Negatives:")
    print("   • VPN usage may be legitimate for privacy-conscious users")
    print("   • High data transfer could be legitimate streaming/downloads")
    print("   • Night calling may be normal for shift workers or international calls")
    print("   • Device changes may be legitimate upgrades")
    print("   • Short calls could be legitimate quick check-ins")
    print("   • Cell tower hopping may occur during commutes")
    print("   • Weekend spikes could be normal for personal users")
    print("\n🔧 Recommendations:")
    print("   • Adjust thresholds based on baseline user behavior")
    print("   • Use optimal threshold from threshold analysis")
    print("   • Combine with ML models for better accuracy")
    print("   • Implement user feedback loop for false positive reduction")
    print("   • Add whitelisting for known legitimate patterns")
    print("   • Monitor anomaly score distribution for threshold tuning")
    print("   • Consider contextual factors (time zones, user profiles)")
    print("   • Set up alerts for high-severity anomalies (score > 7)")
    print("   • Review false negative cases to improve rule coverage")
    print("\n📊 Performance Metrics Explained:")
    print("   • Accuracy: Overall correctness (TP+TN)/(TP+TN+FP+FN)")
    print("   • Precision: Of flagged cases, how many are truly fraud (TP/(TP+FP))")
    print("   • Recall: Of all fraud cases, how many we detected (TP/(TP+FN))")
    print("   • F1-Score: Balance between precision and recall")
    print("   • ROC AUC: Model's ability to distinguish classes (higher is better)")
    print("   • False Positive Rate: Normal cases incorrectly flagged")
    print("   • False Negative Rate: Fraud cases we missed")
    print("\n🎯 Model Quality Guidelines:")
    print("   • Excellent: F1 > 0.9, Accuracy > 0.95")
    print("   • Good: F1 > 0.8, Accuracy > 0.90")
    print("   • Fair: F1 > 0.6, Accuracy > 0.80")
    print("   • Needs Improvement: F1 < 0.6")
    print("\n📊 Visualization Categories:")
    print("   IPDR: Overview, Protocol Analysis, User Patterns, Statistics")
    print("   CDR: Overview, Fraud Patterns, User Behavior, Advanced Metrics")
    print("="*70)


if __name__ == "__main__":
    main()