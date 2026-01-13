"""
Rule-Based Fraud Detection Agent for Telecom Systems
Analyzes IPDR and CDR data to detect fraud using deterministic rules
"""

import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
from typing import List, Dict, Any
import json

class RuleBasedFraudAgent:
    """
    Rule-Based Fraud Detection Agent
    
    Purpose: Detect telecom fraud using deterministic rules without ML
    Inputs: IPDR and CDR records (pandas DataFrames or individual rows)
    Outputs: Fraud alerts with type, confidence, and evidence
    """
    
    def __init__(self):
        # Time windows for pattern detection (in seconds)
        self.TIME_WINDOWS = {
            'short': 300,      # 5 minutes
            'medium': 3600,    # 1 hour
            'long': 86400      # 24 hours
        }
        
        # Rule thresholds
        self.THRESHOLDS = {
            'sim_swap_imei_changes': 3,           # IMEI changes per day
            'sim_swap_cell_changes': 5,           # Rapid cell tower changes
            'cloning_concurrent_calls': 2,        # Simultaneous calls
            'mass_sms_count': 100,                # SMS in 5 minutes
            'mass_sms_unique_recipients': 50,     # Unique callees in hour
            'voip_sip_ports': [5060, 5061],       # SIP protocol ports
            'voip_excessive_duration': 10800,     # 3 hours continuous
            'vpn_high_traffic_mb': 1000,          # 1GB in hour
            'domain_frequency': 500,              # Requests in 5 minutes
            'call_duration_short': 3,             # Seconds (flash calls)
            'call_duration_long': 14400,          # 4 hours
            'tower_hopping_count': 10,            # Cell changes in hour
            'excessive_calls_count': 200,         # Calls per hour
            'data_volume_anomaly_gb': 5,          # GB per hour
            'night_activity_hours': (23, 6),      # 11 PM to 6 AM
        }
        
        # State tracking for pattern detection
        self.user_state = defaultdict(lambda: {
            'imei_history': [],
            'cell_history': [],
            'call_history': [],
            'sms_history': [],
            'domain_access': defaultdict(list),
            'data_usage': [],
            'vpn_sessions': []
        })
        
    def analyze_ipdr(self, ipdr_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze IPDR records for fraud patterns"""
        alerts = []
        
        for _, row in ipdr_data.iterrows():
            # Convert row to dict for easier access
            record = row.to_dict()
            
            # Run all IPDR-based rules
            alerts.extend(self._check_vpn_abuse(record))
            alerts.extend(self._check_high_frequency_domain_access(record))
            alerts.extend(self._check_data_volume_anomaly(record))
            alerts.extend(self._check_voip_spoofing(record))
            alerts.extend(self._check_night_activity(record))
            
        return alerts
    
    def analyze_cdr(self, cdr_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze CDR records for fraud patterns"""
        alerts = []
        
        for _, row in cdr_data.iterrows():
            record = row.to_dict()
            
            # Run all CDR-based rules
            alerts.extend(self._check_sim_swap(record))
            alerts.extend(self._check_sim_cloning(record))
            alerts.extend(self._check_mass_sms_spam(record))
            alerts.extend(self._check_call_anomalies(record))
            alerts.extend(self._check_tower_hopping(record))
            alerts.extend(self._check_excessive_calls(record))
            
        return alerts
    
    # ==================== SIM SWAP DETECTION ====================
    def _check_sim_swap(self, record: Dict) -> List[Dict]:
        """
        Rule: Detect SIM swap by monitoring IMEI changes
        Condition: More than 3 different IMEIs for same IMSI in 24 hours
        Rationale: Legitimate users rarely change devices multiple times daily
        """
        alerts = []
        user_id = record.get('imsi') or record.get('user_id')
        imei = record.get('imei')
        timestamp = self._parse_timestamp(record.get('call_start') or record.get('timestamp'))
        
        if not all([user_id, imei, timestamp]):
            return alerts
        
        # Track IMEI history
        state = self.user_state[user_id]
        state['imei_history'].append({
            'imei': imei,
            'timestamp': timestamp
        })
        
        # Clean old entries (24 hour window)
        cutoff = timestamp - timedelta(seconds=self.TIME_WINDOWS['long'])
        state['imei_history'] = [
            h for h in state['imei_history'] 
            if h['timestamp'] > cutoff
        ]
        
        # Count unique IMEIs
        unique_imeis = len(set(h['imei'] for h in state['imei_history']))
        
        if unique_imeis >= self.THRESHOLDS['sim_swap_imei_changes']:
            alerts.append({
                'alert_type': 'SIM_SWAP_DETECTED',
                'confidence': 0.85,
                'user_id': user_id,
                'timestamp': timestamp.isoformat(),
                'evidence': {
                    'unique_imeis': unique_imeis,
                    'imei_list': list(set(h['imei'] for h in state['imei_history'])),
                    'time_window': '24_hours',
                    'threshold_exceeded': unique_imeis - self.THRESHOLDS['sim_swap_imei_changes']
                },
                'severity': 'HIGH'
            })
        
        return alerts
    
    # ==================== SIM CLONING DETECTION ====================
    def _check_sim_cloning(self, record: Dict) -> List[Dict]:
        """
        Rule: Detect SIM cloning via concurrent call activity
        Condition: Same IMSI making calls from 2+ different IMEIs simultaneously
        Rationale: One SIM cannot be in multiple devices at once physically
        """
        alerts = []
        user_id = record.get('imsi')
        imei = record.get('imei')
        call_start = self._parse_timestamp(record.get('call_start'))
        call_end = self._parse_timestamp(record.get('call_end'))
        
        if not all([user_id, imei, call_start]):
            return alerts
        
        state = self.user_state[user_id]
        
        # Track active calls
        state['call_history'].append({
            'imei': imei,
            'start': call_start,
            'end': call_end or call_start + timedelta(seconds=record.get('duration', 0)),
            'cdr_id': record.get('cdr_id')
        })
        
        # Clean old calls
        cutoff = call_start - timedelta(seconds=self.TIME_WINDOWS['medium'])
        state['call_history'] = [
            c for c in state['call_history']
            if c['end'] > cutoff
        ]
        
        # Check for overlapping calls with different IMEIs
        concurrent_imeis = set()
        for call in state['call_history']:
            # Check if this call overlaps with current call
            if (call['start'] <= call_start <= call['end']) or \
               (call_start <= call['start'] <= call_end):
                concurrent_imeis.add(call['imei'])
        
        if len(concurrent_imeis) >= self.THRESHOLDS['cloning_concurrent_calls']:
            alerts.append({
                'alert_type': 'SIM_CLONING_DETECTED',
                'confidence': 0.95,
                'user_id': user_id,
                'timestamp': call_start.isoformat(),
                'evidence': {
                    'concurrent_imeis': list(concurrent_imeis),
                    'concurrent_count': len(concurrent_imeis),
                    'call_details': [
                        {'imei': c['imei'], 'start': c['start'].isoformat()} 
                        for c in state['call_history'][-5:]
                    ]
                },
                'severity': 'CRITICAL'
            })
        
        return alerts
    
    # ==================== MASS SMS SPAM DETECTION ====================
    def _check_mass_sms_spam(self, record: Dict) -> List[Dict]:
        """
        Rule: Detect mass SMS spam campaigns
        Condition: >100 SMS in 5 minutes OR >50 unique recipients in 1 hour
        Rationale: Normal users don't send bulk SMS; indicates spam/phishing
        """
        alerts = []
        
        if record.get('call_type') != 'SMS' and record.get('service_type') != 'SMS':
            return alerts
        
        user_id = record.get('caller') or record.get('imsi')
        timestamp = self._parse_timestamp(record.get('call_start') or record.get('timestamp'))
        recipient = record.get('callee')
        
        if not all([user_id, timestamp, recipient]):
            return alerts
        
        state = self.user_state[user_id]
        state['sms_history'].append({
            'timestamp': timestamp,
            'recipient': recipient
        })
        
        # Check 5-minute window for volume
        cutoff_short = timestamp - timedelta(seconds=self.TIME_WINDOWS['short'])
        recent_sms = [s for s in state['sms_history'] if s['timestamp'] > cutoff_short]
        
        if len(recent_sms) >= self.THRESHOLDS['mass_sms_count']:
            alerts.append({
                'alert_type': 'MASS_SMS_SPAM',
                'confidence': 0.90,
                'user_id': user_id,
                'timestamp': timestamp.isoformat(),
                'evidence': {
                    'sms_count_5min': len(recent_sms),
                    'threshold': self.THRESHOLDS['mass_sms_count'],
                    'time_window': '5_minutes'
                },
                'severity': 'HIGH'
            })
        
        # Check 1-hour window for unique recipients
        cutoff_medium = timestamp - timedelta(seconds=self.TIME_WINDOWS['medium'])
        hourly_sms = [s for s in state['sms_history'] if s['timestamp'] > cutoff_medium]
        unique_recipients = len(set(s['recipient'] for s in hourly_sms))
        
        if unique_recipients >= self.THRESHOLDS['mass_sms_unique_recipients']:
            alerts.append({
                'alert_type': 'MASS_SMS_BROADCAST',
                'confidence': 0.88,
                'user_id': user_id,
                'timestamp': timestamp.isoformat(),
                'evidence': {
                    'unique_recipients': unique_recipients,
                    'total_sms': len(hourly_sms),
                    'threshold': self.THRESHOLDS['mass_sms_unique_recipients'],
                    'time_window': '1_hour'
                },
                'severity': 'HIGH'
            })
        
        # Clean old SMS records
        state['sms_history'] = [s for s in state['sms_history'] if s['timestamp'] > cutoff_medium]
        
        return alerts
    
    # ==================== VOIP SPOOFING DETECTION ====================
    def _check_voip_spoofing(self, record: Dict) -> List[Dict]:
        """
        Rule: Detect VOIP/SIP spoofing attempts
        Condition: SIP protocol traffic (ports 5060/5061) with excessive duration
        Rationale: Spoofed calls often use VOIP; long sessions indicate toll fraud
        """
        alerts = []
        
        port = record.get('port')
        protocol = record.get('protocol', '').upper()
        duration = record.get('duration', 0)
        user_id = record.get('user_id') or record.get('imsi')
        timestamp = self._parse_timestamp(record.get('timestamp'))
        
        if not all([port, user_id, timestamp]):
            return alerts
        
        # Check for SIP ports
        is_sip = (port in self.THRESHOLDS['voip_sip_ports']) or (protocol == 'SIP')
        
        if is_sip and duration > self.THRESHOLDS['voip_excessive_duration']:
            alerts.append({
                'alert_type': 'VOIP_SPOOFING_SUSPECTED',
                'confidence': 0.75,
                'user_id': user_id,
                'timestamp': timestamp.isoformat(),
                'evidence': {
                    'port': port,
                    'protocol': protocol,
                    'duration_seconds': duration,
                    'duration_hours': round(duration / 3600, 2),
                    'threshold_hours': self.THRESHOLDS['voip_excessive_duration'] / 3600,
                    'ip_src': record.get('ip_src'),
                    'ip_dst': record.get('ip_dst')
                },
                'severity': 'MEDIUM'
            })
        
        return alerts
    
    # ==================== VPN/PROXY ABUSE DETECTION ====================
    def _check_vpn_abuse(self, record: Dict) -> List[Dict]:
        """
        Rule: Detect VPN/Proxy abuse for anonymization
        Condition: VPN usage with high data transfer (>1GB/hour)
        Rationale: Fraudsters use VPNs to hide location; high traffic indicates data theft
        """
        alerts = []
        
        vpn_usage = record.get('vpn_usage')
        bytes_total = (record.get('bytes_sent', 0) + record.get('bytes_received', 0))
        user_id = record.get('user_id') or record.get('imsi')
        timestamp = self._parse_timestamp(record.get('timestamp'))
        
        if not all([user_id, timestamp]):
            return alerts
        
        if vpn_usage:
            state = self.user_state[user_id]
            state['vpn_sessions'].append({
                'timestamp': timestamp,
                'bytes': bytes_total
            })
            
            # Check hourly VPN traffic
            cutoff = timestamp - timedelta(seconds=self.TIME_WINDOWS['medium'])
            hourly_vpn = [v for v in state['vpn_sessions'] if v['timestamp'] > cutoff]
            total_mb = sum(v['bytes'] for v in hourly_vpn) / (1024 * 1024)
            
            if total_mb >= self.THRESHOLDS['vpn_high_traffic_mb']:
                alerts.append({
                    'alert_type': 'VPN_ABUSE_DETECTED',
                    'confidence': 0.70,
                    'user_id': user_id,
                    'timestamp': timestamp.isoformat(),
                    'evidence': {
                        'traffic_mb': round(total_mb, 2),
                        'traffic_gb': round(total_mb / 1024, 2),
                        'threshold_mb': self.THRESHOLDS['vpn_high_traffic_mb'],
                        'vpn_sessions_count': len(hourly_vpn),
                        'time_window': '1_hour'
                    },
                    'severity': 'MEDIUM'
                })
            
            state['vpn_sessions'] = hourly_vpn
        
        return alerts
    
    # ==================== HIGH-FREQUENCY DOMAIN ACCESS ====================
    def _check_high_frequency_domain_access(self, record: Dict) -> List[Dict]:
        """
        Rule: Detect automated/bot-like domain access patterns
        Condition: >500 requests to same domain in 5 minutes
        Rationale: Indicates scraping, DDoS, or automated fraud attempts
        """
        alerts = []
        
        domain = record.get('domain')
        user_id = record.get('user_id') or record.get('imsi')
        timestamp = self._parse_timestamp(record.get('timestamp'))
        
        if not all([domain, user_id, timestamp]):
            return alerts
        
        state = self.user_state[user_id]
        state['domain_access'][domain].append(timestamp)
        
        # Check 5-minute window
        cutoff = timestamp - timedelta(seconds=self.TIME_WINDOWS['short'])
        recent_access = [t for t in state['domain_access'][domain] if t > cutoff]
        
        if len(recent_access) >= self.THRESHOLDS['domain_frequency']:
            alerts.append({
                'alert_type': 'HIGH_FREQUENCY_DOMAIN_ACCESS',
                'confidence': 0.80,
                'user_id': user_id,
                'timestamp': timestamp.isoformat(),
                'evidence': {
                    'domain': domain,
                    'request_count': len(recent_access),
                    'threshold': self.THRESHOLDS['domain_frequency'],
                    'time_window': '5_minutes',
                    'requests_per_second': round(len(recent_access) / 300, 2)
                },
                'severity': 'MEDIUM'
            })
        
        # Cleanup
        state['domain_access'][domain] = recent_access
        
        return alerts
    
    # ==================== CALL ANOMALY DETECTION ====================
    def _check_call_anomalies(self, record: Dict) -> List[Dict]:
        """
        Rule: Detect abnormal call durations
        Condition: Flash calls (<3 sec) or extremely long calls (>4 hours)
        Rationale: Flash calls used for verification fraud; long calls indicate toll fraud
        """
        alerts = []
        
        duration = record.get('duration', 0)
        call_type = record.get('call_type')
        user_id = record.get('caller') or record.get('imsi')
        timestamp = self._parse_timestamp(record.get('call_start'))
        
        if call_type == 'SMS' or not all([user_id, timestamp]):
            return alerts
        
        # Flash call detection
        if 0 < duration <= self.THRESHOLDS['call_duration_short']:
            alerts.append({
                'alert_type': 'FLASH_CALL_DETECTED',
                'confidence': 0.65,
                'user_id': user_id,
                'timestamp': timestamp.isoformat(),
                'evidence': {
                    'duration_seconds': duration,
                    'threshold': self.THRESHOLDS['call_duration_short'],
                    'callee': record.get('callee'),
                    'call_type': call_type
                },
                'severity': 'LOW'
            })
        
        # Extremely long call detection
        if duration >= self.THRESHOLDS['call_duration_long']:
            alerts.append({
                'alert_type': 'EXCESSIVE_CALL_DURATION',
                'confidence': 0.78,
                'user_id': user_id,
                'timestamp': timestamp.isoformat(),
                'evidence': {
                    'duration_seconds': duration,
                    'duration_hours': round(duration / 3600, 2),
                    'threshold_hours': self.THRESHOLDS['call_duration_long'] / 3600,
                    'callee': record.get('callee'),
                    'potential_toll_fraud': True
                },
                'severity': 'HIGH'
            })
        
        return alerts
    
    # ==================== TOWER HOPPING DETECTION ====================
    def _check_tower_hopping(self, record: Dict) -> List[Dict]:
        """
        Rule: Detect rapid cell tower changes
        Condition: >10 different cell towers in 1 hour
        Rationale: Indicates device cloning or location spoofing
        """
        alerts = []
        
        cell_id = record.get('cell_id')
        user_id = record.get('caller') or record.get('imsi')
        timestamp = self._parse_timestamp(record.get('call_start') or record.get('timestamp'))
        
        if not all([cell_id, user_id, timestamp]):
            return alerts
        
        state = self.user_state[user_id]
        state['cell_history'].append({
            'cell_id': cell_id,
            'timestamp': timestamp
        })
        
        # Check 1-hour window
        cutoff = timestamp - timedelta(seconds=self.TIME_WINDOWS['medium'])
        recent_cells = [c for c in state['cell_history'] if c['timestamp'] > cutoff]
        unique_cells = len(set(c['cell_id'] for c in recent_cells))
        
        if unique_cells >= self.THRESHOLDS['tower_hopping_count']:
            alerts.append({
                'alert_type': 'TOWER_HOPPING_DETECTED',
                'confidence': 0.82,
                'user_id': user_id,
                'timestamp': timestamp.isoformat(),
                'evidence': {
                    'unique_cell_towers': unique_cells,
                    'threshold': self.THRESHOLDS['tower_hopping_count'],
                    'time_window': '1_hour',
                    'cell_ids': list(set(c['cell_id'] for c in recent_cells))
                },
                'severity': 'HIGH'
            })
        
        state['cell_history'] = recent_cells
        
        return alerts
    
    # ==================== EXCESSIVE CALLS DETECTION ====================
    def _check_excessive_calls(self, record: Dict) -> List[Dict]:
        """
        Rule: Detect excessive call volume
        Condition: >200 calls per hour
        Rationale: Indicates robocalling, spam, or account compromise
        """
        alerts = []
        
        user_id = record.get('caller') or record.get('imsi')
        timestamp = self._parse_timestamp(record.get('call_start'))
        call_type = record.get('call_type')
        
        if call_type == 'SMS' or not all([user_id, timestamp]):
            return alerts
        
        state = self.user_state[user_id]
        
        # Count calls in last hour (already tracked in call_history)
        cutoff = timestamp - timedelta(seconds=self.TIME_WINDOWS['medium'])
        recent_calls = [c for c in state['call_history'] if c['start'] > cutoff]
        
        if len(recent_calls) >= self.THRESHOLDS['excessive_calls_count']:
            alerts.append({
                'alert_type': 'EXCESSIVE_CALL_VOLUME',
                'confidence': 0.85,
                'user_id': user_id,
                'timestamp': timestamp.isoformat(),
                'evidence': {
                    'call_count': len(recent_calls),
                    'threshold': self.THRESHOLDS['excessive_calls_count'],
                    'time_window': '1_hour',
                    'calls_per_minute': round(len(recent_calls) / 60, 2)
                },
                'severity': 'HIGH'
            })
        
        return alerts
    
    # ==================== DATA VOLUME ANOMALY ====================
    def _check_data_volume_anomaly(self, record: Dict) -> List[Dict]:
        """
        Rule: Detect abnormal data usage
        Condition: >5GB data transfer in 1 hour
        Rationale: Indicates data theft or account compromise
        """
        alerts = []
        
        bytes_total = (record.get('bytes_sent', 0) + record.get('bytes_received', 0))
        user_id = record.get('user_id') or record.get('imsi')
        timestamp = self._parse_timestamp(record.get('timestamp'))
        
        if not all([user_id, timestamp]) or bytes_total == 0:
            return alerts
        
        state = self.user_state[user_id]
        state['data_usage'].append({
            'timestamp': timestamp,
            'bytes': bytes_total
        })
        
        # Check hourly data usage
        cutoff = timestamp - timedelta(seconds=self.TIME_WINDOWS['medium'])
        hourly_data = [d for d in state['data_usage'] if d['timestamp'] > cutoff]
        total_gb = sum(d['bytes'] for d in hourly_data) / (1024 ** 3)
        
        if total_gb >= self.THRESHOLDS['data_volume_anomaly_gb']:
            alerts.append({
                'alert_type': 'DATA_VOLUME_ANOMALY',
                'confidence': 0.75,
                'user_id': user_id,
                'timestamp': timestamp.isoformat(),
                'evidence': {
                    'data_usage_gb': round(total_gb, 2),
                    'threshold_gb': self.THRESHOLDS['data_volume_anomaly_gb'],
                    'time_window': '1_hour',
                    'session_count': len(hourly_data)
                },
                'severity': 'MEDIUM'
            })
        
        state['data_usage'] = hourly_data
        
        return alerts
    
    # ==================== NIGHT ACTIVITY DETECTION ====================
    def _check_night_activity(self, record: Dict) -> List[Dict]:
        """
        Rule: Detect suspicious night-time activity
        Condition: High data usage during 11 PM - 6 AM
        Rationale: Unusual for normal users; indicates automated fraud or data exfiltration
        """
        alerts = []
        
        timestamp = self._parse_timestamp(record.get('timestamp'))
        bytes_total = (record.get('bytes_sent', 0) + record.get('bytes_received', 0))
        user_id = record.get('user_id') or record.get('imsi')
        
        if not all([timestamp, user_id]) or bytes_total < 100 * 1024 * 1024:  # >100MB
            return alerts
        
        hour = timestamp.hour
        night_start, night_end = self.THRESHOLDS['night_activity_hours']
        
        if hour >= night_start or hour < night_end:
            alerts.append({
                'alert_type': 'SUSPICIOUS_NIGHT_ACTIVITY',
                'confidence': 0.60,
                'user_id': user_id,
                'timestamp': timestamp.isoformat(),
                'evidence': {
                    'hour': hour,
                    'data_usage_mb': round(bytes_total / (1024 * 1024), 2),
                    'night_window': f'{night_start}:00-{night_end}:00'
                },
                'severity': 'LOW'
            })
        
        return alerts
    
    # ==================== UTILITY METHODS ====================
    def _parse_timestamp(self, ts) -> datetime:
        """Parse timestamp to datetime object"""
        if pd.isna(ts) or ts is None:
            return datetime.now()
        if isinstance(ts, datetime):
            return ts
        if isinstance(ts, str):
            try:
                return datetime.fromisoformat(ts.replace('Z', '+00:00'))
            except:
                return datetime.now()
        return datetime.now()
    
    def get_alert_summary(self, alerts: List[Dict]) -> Dict:
        """Generate summary statistics of alerts"""
        if not alerts:
            return {'total_alerts': 0}
        
        summary = {
            'total_alerts': len(alerts),
            'by_type': defaultdict(int),
            'by_severity': defaultdict(int),
            'avg_confidence': 0,
            'high_severity_count': 0
        }
        
        for alert in alerts:
            summary['by_type'][alert['alert_type']] += 1
            summary['by_severity'][alert['severity']] += 1
            summary['avg_confidence'] += alert['confidence']
        
        summary['avg_confidence'] = round(summary['avg_confidence'] / len(alerts), 3)
        summary['high_severity_count'] = sum(
            count for sev, count in summary['by_severity'].items() 
            if sev in ['HIGH', 'CRITICAL']
        )
        
        return dict(summary)
    
    def export_alerts_for_correlation(self, alerts: List[Dict]) -> str:
        """
        Export alerts in JSON format for Correlation Agent
        Format: {user_id, alert_types[], max_confidence, evidence_summary}
        """
        if not alerts:
            return json.dumps([])
        
        # Group alerts by user
        user_alerts = defaultdict(list)
        for alert in alerts:
            user_alerts[alert['user_id']].append(alert)
        
        correlation_data = []
        for user_id, user_alert_list in user_alerts.items():
            correlation_data.append({
                'user_id': user_id,
                'alert_types': [a['alert_type'] for a in user_alert_list],
                'alert_count': len(user_alert_list),
                'max_confidence': max(a['confidence'] for a in user_alert_list),
                'max_severity': self._get_max_severity(user_alert_list),
                'timestamp': user_alert_list[-1]['timestamp'],
                'evidence_summary': [a['evidence'] for a in user_alert_list]
            })
        
        return json.dumps(correlation_data, indent=2)
    
    def _get_max_severity(self, alerts: List[Dict]) -> str:
        """Get maximum severity level from alerts"""
        severity_order = {'CRITICAL': 4, 'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
        max_sev = max(alerts, key=lambda x: severity_order.get(x['severity'], 0))
        return max_sev['severity']


# ==================== DEMO USAGE ====================
def main():
    print("=" * 70)
    print("RULE-BASED TELECOM FRAUD DETECTION AGENT")
    print("=" * 70)
    
    # Initialize agent
    agent = RuleBasedFraudAgent()
    
    # Get manual input for IPDR data
    print("\n📊 IPDR DATA ENTRY")
    print("-" * 70)
    print("Enter IPDR records (comma-separated values)")
    print("Format: event_id,user_id,imsi,imei,timestamp,domain,ip_src,ip_dst,port,protocol,duration,bytes_sent,bytes_received,vpn_usage,cell_id")
    print("Example: E001,U001,460001234567890,123456789012345,2024-12-08T10:30:00,facebook.com,192.168.1.1,8.8.8.8,443,HTTPS,300,1048576,2097152,True,CELL001")
    print("Enter 'done' when finished\n")
    
    ipdr_records = []
    while True:
        entry = input("IPDR record: ").strip()
        if entry.lower() == 'done':
            break
        if entry:
            ipdr_records.append(entry)
    
    # Get manual input for CDR data
    print("\n📞 CDR DATA ENTRY")
    print("-" * 70)
    print("Enter CDR records (comma-separated values)")
    print("Format: cdr_id,caller,callee,call_start,call_end,duration,call_type,cell_id,imei,imsi,status,service_type")
    print("Example: C001,919876543210,919123456789,2024-12-08T10:30:00,2024-12-08T10:35:00,300,VOICE,CELL001,123456789012345,460001234567890,COMPLETED,VOICE")
    print("Enter 'done' when finished\n")
    
    cdr_records = []
    while True:
        entry = input("CDR record: ").strip()
        if entry.lower() == 'done':
            break
        if entry:
            cdr_records.append(entry)
    
    # Parse IPDR data
    if ipdr_records:
        print("\n🔍 Processing IPDR Data...")
        ipdr_data = []
        ipdr_columns = ['event_id', 'user_id', 'imsi', 'imei', 'timestamp', 'domain', 
                       'ip_src', 'ip_dst', 'port', 'protocol', 'duration', 'bytes_sent', 
                       'bytes_received', 'vpn_usage', 'cell_id']
        
        for record in ipdr_records:
            values = record.split(',')
            if len(values) == len(ipdr_columns):
                row = {}
                for i, col in enumerate(ipdr_columns):
                    val = values[i].strip()
                    # Type conversion
                    if col in ['port', 'duration', 'bytes_sent', 'bytes_received']:
                        row[col] = int(val) if val.isdigit() else 0
                    elif col == 'vpn_usage':
                        row[col] = val.lower() in ['true', '1', 'yes']
                    else:
                        row[col] = val
                ipdr_data.append(row)
        
        if ipdr_data:
            ipdr_df = pd.DataFrame(ipdr_data)
            ipdr_alerts = agent.analyze_ipdr(ipdr_df)
            
            print(f"✅ Analyzed {len(ipdr_data)} IPDR records")
            print(f"🚨 Generated {len(ipdr_alerts)} alerts\n")
            
            if ipdr_alerts:
                print("=" * 70)
                print("IPDR FRAUD ALERTS")
                print("=" * 70)
                for i, alert in enumerate(ipdr_alerts, 1):
                    print(f"\n🔴 Alert #{i}: {alert['alert_type']}")
                    print(f"   Severity: {alert['severity']}")
                    print(f"   Confidence: {alert['confidence']:.2%}")
                    print(f"   User: {alert['user_id']}")
                    print(f"   Time: {alert['timestamp']}")
                    print(f"   Evidence: {json.dumps(alert['evidence'], indent=6)}")
        else:
            print("⚠️  No valid IPDR records parsed")
    
    # Parse CDR data
    if cdr_records:
        print("\n🔍 Processing CDR Data...")
        cdr_data = []
        cdr_columns = ['cdr_id', 'caller', 'callee', 'call_start', 'call_end', 
                      'duration', 'call_type', 'cell_id', 'imei', 'imsi', 'status', 'service_type']
        
        for record in cdr_records:
            values = record.split(',')
            if len(values) == len(cdr_columns):
                row = {}
                for i, col in enumerate(cdr_columns):
                    val = values[i].strip()
                    # Type conversion
                    if col == 'duration':
                        row[col] = int(val) if val.isdigit() else 0
                    else:
                        row[col] = val
                cdr_data.append(row)
        
        if cdr_data:
            cdr_df = pd.DataFrame(cdr_data)
            cdr_alerts = agent.analyze_cdr(cdr_df)
            
            print(f"✅ Analyzed {len(cdr_data)} CDR records")
            print(f"🚨 Generated {len(cdr_alerts)} alerts\n")
            
            if cdr_alerts:
                print("=" * 70)
                print("CDR FRAUD ALERTS")
                print("=" * 70)
                for i, alert in enumerate(cdr_alerts, 1):
                    print(f"\n🔴 Alert #{i}: {alert['alert_type']}")
                    print(f"   Severity: {alert['severity']}")
                    print(f"   Confidence: {alert['confidence']:.2%}")
                    print(f"   User: {alert['user_id']}")
                    print(f"   Time: {alert['timestamp']}")
                    print(f"   Evidence: {json.dumps(alert['evidence'], indent=6)}")
        else:
            print("⚠️  No valid CDR records parsed")
    
    # Generate combined summary
    all_alerts = []
    if ipdr_records:
        ipdr_df = pd.DataFrame([dict(zip(['event_id', 'user_id', 'imsi', 'imei', 'timestamp', 'domain', 
                                          'ip_src', 'ip_dst', 'port', 'protocol', 'duration', 'bytes_sent', 
                                          'bytes_received', 'vpn_usage', 'cell_id'], 
                                         r.split(','))) for r in ipdr_records])
        all_alerts.extend(agent.analyze_ipdr(ipdr_df))
    
    if cdr_records:
        cdr_df = pd.DataFrame([dict(zip(['cdr_id', 'caller', 'callee', 'call_start', 'call_end', 
                                         'duration', 'call_type', 'cell_id', 'imei', 'imsi', 'status', 'service_type'],
                                        r.split(','))) for r in cdr_records])
        all_alerts.extend(agent.analyze_cdr(cdr_df))
    
    if all_alerts:
        print("\n" + "=" * 70)
        print("ALERT SUMMARY")
        print("=" * 70)
        summary = agent.get_alert_summary(all_alerts)
        print(f"Total Alerts: {summary['total_alerts']}")
        print(f"Average Confidence: {summary['avg_confidence']:.2%}")
        print(f"High/Critical Severity: {summary['high_severity_count']}")
        print(f"\nBy Type:")
        for alert_type, count in summary['by_type'].items():
            print(f"  - {alert_type}: {count}")
        print(f"\nBy Severity:")
        for severity, count in summary['by_severity'].items():
            print(f"  - {severity}: {count}")
        
        # Export for correlation
        print("\n" + "=" * 70)
        print("CORRELATION AGENT EXPORT")
        print("=" * 70)
        correlation_export = agent.export_alerts_for_correlation(all_alerts)
        print(correlation_export)
    else:
        print("\n✅ No fraud detected in the analyzed data")
    
    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)


# ==================== EXAMPLE DATA FOR TESTING ====================
def run_demo_with_sample_data():
    """Run demo with pre-populated sample data"""
    print("=" * 70)
    print("DEMO MODE: Using Sample Data")
    print("=" * 70)
    
    agent = RuleBasedFraudAgent()
    
    # Sample IPDR data demonstrating various fraud patterns
    ipdr_sample = pd.DataFrame([
        # VPN abuse with high traffic
        {
            'event_id': 'E001', 'user_id': 'U001', 'imsi': '460001234567890',
            'imei': '123456789012345', 'timestamp': '2024-12-08T10:30:00',
            'domain': 'suspicious-site.com', 'ip_src': '192.168.1.1',
            'ip_dst': '8.8.8.8', 'port': 443, 'protocol': 'HTTPS',
            'duration': 3600, 'bytes_sent': 536870912, 'bytes_received': 536870912,
            'vpn_usage': True, 'cell_id': 'CELL001'
        },
        # High frequency domain access
        *[{
            'event_id': f'E00{i}', 'user_id': 'U002', 'imsi': '460009876543210',
            'imei': '987654321098765', 'timestamp': f'2024-12-08T11:00:{str(i%60).zfill(2)}',
            'domain': 'api.target.com', 'ip_src': '192.168.1.2',
            'ip_dst': '1.2.3.4', 'port': 443, 'protocol': 'HTTPS',
            'duration': 1, 'bytes_sent': 1024, 'bytes_received': 2048,
            'vpn_usage': False, 'cell_id': 'CELL002'
        } for i in range(2, 600)],  # 598 requests
        # VOIP spoofing
        {
            'event_id': 'E600', 'user_id': 'U003', 'imsi': '460005555555555',
            'imei': '555555555555555', 'timestamp': '2024-12-08T12:00:00',
            'domain': 'voip.server.com', 'ip_src': '192.168.1.3',
            'ip_dst': '5.6.7.8', 'port': 5060, 'protocol': 'SIP',
            'duration': 12000, 'bytes_sent': 10485760, 'bytes_received': 10485760,
            'vpn_usage': False, 'cell_id': 'CELL003'
        }
    ])
    
    # Sample CDR data demonstrating various fraud patterns
    cdr_sample = pd.DataFrame([
        # SIM swap - multiple IMEI changes
        {
            'cdr_id': 'C001', 'caller': '919876543210', 'callee': '919123456789',
            'call_start': '2024-12-08T08:00:00', 'call_end': '2024-12-08T08:05:00',
            'duration': 300, 'call_type': 'VOICE', 'cell_id': 'CELL001',
            'imei': '111111111111111', 'imsi': '460001111111111',
            'status': 'COMPLETED', 'service_type': 'VOICE'
        },
        {
            'cdr_id': 'C002', 'caller': '919876543210', 'callee': '919123456780',
            'call_start': '2024-12-08T10:00:00', 'call_end': '2024-12-08T10:03:00',
            'duration': 180, 'call_type': 'VOICE', 'cell_id': 'CELL002',
            'imei': '222222222222222', 'imsi': '460001111111111',
            'status': 'COMPLETED', 'service_type': 'VOICE'
        },
        {
            'cdr_id': 'C003', 'caller': '919876543210', 'callee': '919123456781',
            'call_start': '2024-12-08T14:00:00', 'call_end': '2024-12-08T14:02:00',
            'duration': 120, 'call_type': 'VOICE', 'cell_id': 'CELL003',
            'imei': '333333333333333', 'imsi': '460001111111111',
            'status': 'COMPLETED', 'service_type': 'VOICE'
        },
        {
            'cdr_id': 'C004', 'caller': '919876543210', 'callee': '919123456782',
            'call_start': '2024-12-08T16:00:00', 'call_end': '2024-12-08T16:01:00',
            'duration': 60, 'call_type': 'VOICE', 'cell_id': 'CELL004',
            'imei': '444444444444444', 'imsi': '460001111111111',
            'status': 'COMPLETED', 'service_type': 'VOICE'
        },
        # Mass SMS spam
        *[{
            'cdr_id': f'SMS{str(i).zfill(3)}', 'caller': '919999999999',
            'callee': f'9191234567{str(i).zfill(2)}',
            'call_start': f'2024-12-08T13:00:{str(i%60).zfill(2)}',
            'call_end': f'2024-12-08T13:00:{str(i%60).zfill(2)}',
            'duration': 0, 'call_type': 'SMS', 'cell_id': 'CELL005',
            'imei': '999999999999999', 'imsi': '460009999999999',
            'status': 'COMPLETED', 'service_type': 'SMS'
        } for i in range(1, 151)],  # 150 SMS messages
        # Flash calls
        *[{
            'cdr_id': f'FLASH{i}', 'caller': '918888888888',
            'callee': f'9198765432{i%10}',
            'call_start': f'2024-12-08T15:00:{str(i).zfill(2)}',
            'call_end': f'2024-12-08T15:00:{str(i).zfill(2)}',
            'duration': 2, 'call_type': 'VOICE', 'cell_id': 'CELL006',
            'imei': '888888888888888', 'imsi': '460008888888888',
            'status': 'COMPLETED', 'service_type': 'VOICE'
        } for i in range(1, 11)]
    ])
    
    print("\n📊 Analyzing IPDR Data...")
    ipdr_alerts = agent.analyze_ipdr(ipdr_sample)
    print(f"✅ Analyzed {len(ipdr_sample)} IPDR records")
    print(f"🚨 Generated {len(ipdr_alerts)} alerts")
    
    print("\n📞 Analyzing CDR Data...")
    cdr_alerts = agent.analyze_cdr(cdr_sample)
    print(f"✅ Analyzed {len(cdr_sample)} CDR records")
    print(f"🚨 Generated {len(cdr_alerts)} alerts")
    
    all_alerts = ipdr_alerts + cdr_alerts
    
    if all_alerts:
        print("\n" + "=" * 70)
        print("DETECTED FRAUD ALERTS")
        print("=" * 70)
        
        for i, alert in enumerate(all_alerts[:10], 1):  # Show first 10
            print(f"\n🔴 Alert #{i}: {alert['alert_type']}")
            print(f"   Severity: {alert['severity']}")
            print(f"   Confidence: {alert['confidence']:.2%}")
            print(f"   User: {alert['user_id']}")
            print(f"   Evidence: {json.dumps(alert['evidence'], indent=6)}")
        
        if len(all_alerts) > 10:
            print(f"\n... and {len(all_alerts) - 10} more alerts")
        
        print("\n" + "=" * 70)
        print("SUMMARY STATISTICS")
        print("=" * 70)
        summary = agent.get_alert_summary(all_alerts)
        print(json.dumps(summary, indent=2, default=str))
        
        print("\n" + "=" * 70)
        print("EXPORT FOR CORRELATION AGENT")
        print("=" * 70)
        print(agent.export_alerts_for_correlation(all_alerts))


if __name__ == "__main__":
    print("\n🎯 Choose mode:")
    print("1. Manual data entry")
    print("2. Run demo with sample data")
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == '2':
        run_demo_with_sample_data()
    else:
        main()