import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import json
import requests
from typing import Dict, List, Tuple, Optional

# ==================== AUTOENCODER MODEL ====================
class TelecomAutoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=8):
        super(TelecomAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, encoding_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

# ==================== OLLAMA INTEGRATION ====================
class OllamaExplainer:
    def __init__(self, base_url="http://10.10.10.7:11434"):
        self.base_url = base_url
    
    def explain_anomaly(self, anomaly_data: Dict) -> str:
        """Use Ollama LLM to generate human-readable explanations"""
        prompt = f"""Analyze this telecom fraud anomaly detection result:

Anomaly Score: {anomaly_data['anomaly_score']:.3f}
Top Contributing Features:
{json.dumps(anomaly_data['top_features'], indent=2)}

Model Evidence:
{json.dumps(anomaly_data['model_evidence'], indent=2)}

Provide a concise explanation of why this behavior is anomalous and what type of fraud it might indicate."""

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": "llama2",
                    "prompt": prompt,
                    "stream": False
                },
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json().get('response', 'Explanation unavailable')
            else:
                return f"Ollama error: {response.status_code}"
        except Exception as e:
            return f"LLM explanation unavailable: {str(e)}"

# ==================== MAIN ANOMALY DETECTION AGENT ====================
class UnsupervisedAnomalyAgent:
    def __init__(self, n_features=17):
        self.n_features = n_features
        self.scaler = StandardScaler()
        
        # Model initialization
        self.isolation_forest = IsolationForest(
            n_estimators=100,
            contamination=0.05,
            random_state=42,
            n_jobs=-1
        )
        
        self.lof = LocalOutlierFactor(
            n_neighbors=20,
            contamination=0.05,
            novelty=True
        )
        
        self.autoencoder = TelecomAutoencoder(n_features, encoding_dim=8)
        self.ae_optimizer = optim.Adam(self.autoencoder.parameters(), lr=0.001)
        self.ae_criterion = nn.MSELoss()
        
        self.kmeans = KMeans(n_clusters=5, random_state=42)
        self.dbscan = DBSCAN(eps=0.5, min_samples=5)
        
        # Streaming components
        self.window_buffer = deque(maxlen=300)  # 5-minute window
        self.trained = False
        
        # Ollama explainer
        self.explainer = OllamaExplainer()
        
        # Feature names
        self.feature_names = [
            # IPDR (7)
            'bytes_sent', 'bytes_received', 'domain_entropy', 
            'port_variety', 'protocol_frequency', 'session_duration', 'vpn_usage',
            # CDR (5)
            'call_frequency', 'distinct_callees', 'repeat_call_ratio',
            'tower_movement_speed', 'call_duration_std',
            # Fused (5)
            'imei_imsi_uniqueness', 'device_sharing_score', 
            'geo_velocity_anomaly', 'behavioral_drift', 'cross_domain_corr'
        ]
    
    def train_models(self, baseline_data: np.ndarray, epochs=50):
        """Initial training on baseline normal data"""
        print(f"Training models on {len(baseline_data)} baseline samples...")
        
        # Normalize data
        X_scaled = self.scaler.fit_transform(baseline_data)
        
        # Train Isolation Forest
        print("Training Isolation Forest...")
        self.isolation_forest.fit(X_scaled)
        
        # Train LOF
        print("Training LOF...")
        self.lof.fit(X_scaled)
        
        # Train Autoencoder
        print("Training Autoencoder...")
        X_tensor = torch.FloatTensor(X_scaled)
        
        for epoch in range(epochs):
            self.autoencoder.train()
            reconstructed, _ = self.autoencoder(X_tensor)
            loss = self.ae_criterion(reconstructed, X_tensor)
            
            self.ae_optimizer.zero_grad()
            loss.backward()
            self.ae_optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
        
        # Train clustering
        print("Training KMeans...")
        self.kmeans.fit(X_scaled)
        
        self.trained = True
        print("✓ All models trained successfully")
    
    def _get_isolation_forest_score(self, X: np.ndarray) -> np.ndarray:
        """Isolation Forest anomaly scores"""
        scores = self.isolation_forest.decision_function(X)
        # Convert to [0, 1] where 1 is most anomalous
        return 1 - (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    
    def _get_lof_score(self, X: np.ndarray) -> np.ndarray:
        """LOF anomaly scores"""
        scores = -self.lof.decision_function(X)
        scores = np.clip(scores, 0, None)
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    
    def _get_autoencoder_score(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Autoencoder reconstruction error"""
        self.autoencoder.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            reconstructed, encoded = self.autoencoder(X_tensor)
            errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1).numpy()
        
        # Normalize reconstruction errors
        errors_norm = (errors - errors.min()) / (errors.max() - errors.min() + 1e-8)
        return errors_norm, encoded.numpy()
    
    def _get_cluster_distance_score(self, X: np.ndarray) -> np.ndarray:
        """Distance from nearest cluster centroid"""
        distances = np.min(self.kmeans.transform(X), axis=1)
        return (distances - distances.min()) / (distances.max() - distances.min() + 1e-8)
    
    def detect_anomalies(self, features: np.ndarray, entity_ids: List[str]) -> List[Dict]:
        """Main detection pipeline"""
        if not self.trained:
            raise RuntimeError("Models must be trained before detection")
        
        # Scale features
        X_scaled = self.scaler.transform(features)
        
        # Get individual model scores
        if_scores = self._get_isolation_forest_score(X_scaled)
        lof_scores = self._get_lof_score(X_scaled)
        ae_scores, embeddings = self._get_autoencoder_score(X_scaled)
        cluster_scores = self._get_cluster_distance_score(X_scaled)
        
        # Ensemble scoring
        final_scores = (
            0.30 * if_scores +
            0.25 * lof_scores +
            0.30 * ae_scores +
            0.15 * cluster_scores
        )
        
        # Generate results
        results = []
        for i, entity_id in enumerate(entity_ids):
            # Feature contributions (using autoencoder gradients)
            feature_contributions = self._get_feature_contributions(
                features[i], embeddings[i]
            )
            
            # Determine anomaly type
            anomaly_type, subtype = self._classify_anomaly_type(
                features[i], final_scores[i]
            )
            
            result = {
                "entity_id": entity_id,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "anomaly_score": float(final_scores[i]),
                "anomaly_type": anomaly_type,
                "confidence": float(np.std([if_scores[i], lof_scores[i], 
                                            ae_scores[i], cluster_scores[i]])),
                "model_evidence": {
                    "isolation_forest": float(if_scores[i]),
                    "lof": float(lof_scores[i]),
                    "autoencoder": float(ae_scores[i]),
                    "cluster_distance": float(cluster_scores[i])
                },
                "top_features": feature_contributions,
                "anomaly_subtype": subtype,
                "cluster_id": int(self.kmeans.predict(X_scaled[i:i+1])[0]),
                "embedding": embeddings[i].tolist()
            }
            
            results.append(result)
        
        return results
    
    def _get_feature_contributions(self, features: np.ndarray, 
                                   embedding: np.ndarray) -> List[Dict]:
        """Calculate feature importance using embedding-based analysis"""
        # Simple approach: use absolute feature values weighted by their variance
        feature_variance = np.abs(features - np.mean(features))
        
        contributions = []
        for idx, name in enumerate(self.feature_names):
            contributions.append({
                "feature": name,
                "contribution": float(feature_variance[idx])
            })
        
        # Sort by contribution and return top 5
        contributions.sort(key=lambda x: x['contribution'], reverse=True)
        
        # Normalize contributions to sum to 1
        total = sum(c['contribution'] for c in contributions[:5])
        for c in contributions[:5]:
            c['contribution'] = round(c['contribution'] / (total + 1e-8), 3)
        
        return contributions[:5]
    
    def _classify_anomaly_type(self, features: np.ndarray, 
                               score: float) -> Tuple[str, str]:
        """Classify anomaly into categories based on feature patterns"""
        # IPDR-dominated anomalies
        ipdr_features = features[:7]
        cdr_features = features[7:12]
        fused_features = features[12:]
        
        ipdr_magnitude = np.mean(np.abs(ipdr_features))
        cdr_magnitude = np.mean(np.abs(cdr_features))
        fused_magnitude = np.mean(np.abs(fused_features))
        
        if ipdr_magnitude > max(cdr_magnitude, fused_magnitude):
            anomaly_type = "network_behavior"
            # Check specific patterns
            if features[0] > 2.0:  # bytes_sent
                subtype = "data_exfiltration_pattern"
            elif features[2] > 2.0:  # domain_entropy
                subtype = "c2_communication_pattern"
            else:
                subtype = "suspicious_network_activity"
        
        elif cdr_magnitude > max(ipdr_magnitude, fused_magnitude):
            anomaly_type = "call_behavior"
            if features[7] > 2.0:  # call_frequency
                subtype = "spam_calling_pattern"
            elif features[9] > 0.8:  # repeat_call_ratio
                subtype = "automated_dialing_pattern"
            else:
                subtype = "suspicious_call_activity"
        
        else:
            anomaly_type = "cross_domain"
            if features[14] > 2.0:  # geo_velocity_anomaly
                subtype = "impossible_travel"
            elif features[12] < 0.3:  # imei_imsi_uniqueness
                subtype = "device_cloning"
            else:
                subtype = "behavioral_drift"
        
        return anomaly_type, subtype
    
    def stream_process(self, event: Dict) -> Optional[Dict]:
        """Process streaming events with sliding window"""
        # Extract features from event
        features = self._extract_features(event)
        self.window_buffer.append((event['entity_id'], features))
        
        # Check if window is full
        if len(self.window_buffer) == self.window_buffer.maxlen:
            # Aggregate features over window
            entity_ids = [e[0] for e in self.window_buffer]
            features_batch = np.array([e[1] for e in self.window_buffer])
            
            # Detect anomalies
            results = self.detect_anomalies(features_batch, entity_ids)
            
            # Return most recent result
            return results[-1]
        
        return None
    
    def _extract_features(self, event: Dict) -> np.ndarray:
        """Extract feature vector from raw event"""
        features = []
        
        # IPDR features
        ipdr = event.get('ipdr', {})
        features.extend([
            ipdr.get('bytes_sent', 0),
            ipdr.get('bytes_received', 0),
            ipdr.get('domain_entropy', 0),
            ipdr.get('port_variety', 0),
            ipdr.get('protocol_frequency', 0),
            ipdr.get('session_duration', 0),
            ipdr.get('vpn_usage', 0)
        ])
        
        # CDR features
        cdr = event.get('cdr', {})
        features.extend([
            cdr.get('call_frequency', 0),
            cdr.get('distinct_callees', 0),
            cdr.get('repeat_call_ratio', 0),
            cdr.get('tower_movement_speed', 0),
            cdr.get('call_duration_std', 0)
        ])
        
        # Fused features
        fused = event.get('fused', {})
        features.extend([
            fused.get('imei_imsi_uniqueness', 1.0),
            fused.get('device_sharing_score', 0),
            fused.get('geo_velocity_anomaly', 0),
            fused.get('behavioral_drift', 0),
            fused.get('cross_domain_corr', 0)
        ])
        
        return np.array(features)
    
    def explain_with_llm(self, anomaly_result: Dict) -> str:
        """Generate human-readable explanation using Ollama"""
        return self.explainer.explain_anomaly(anomaly_result)
    
    def partial_update(self, new_data: np.ndarray):
        """Incremental model updates for online learning"""
        X_scaled = self.scaler.transform(new_data)
        
        # Update LOF (supports novelty detection)
        self.lof.fit(X_scaled)
        
        # Fine-tune autoencoder
        X_tensor = torch.FloatTensor(X_scaled)
        self.autoencoder.train()
        
        for _ in range(5):  # Quick fine-tuning
            reconstructed, _ = self.autoencoder(X_tensor)
            loss = self.ae_criterion(reconstructed, X_tensor)
            
            self.ae_optimizer.zero_grad()
            loss.backward()
            self.ae_optimizer.step()

# ==================== MANUAL INPUT INTERFACE ====================
def manual_detection_interface():
    """Interactive interface for manual feature input"""
    print("\n=== Telecom Fraud Anomaly Detection - Manual Input ===\n")
    
    agent = UnsupervisedAnomalyAgent()
    
    # Generate synthetic baseline for demo
    print("Generating baseline data for training...")
    baseline = np.random.randn(1000, 17) * 0.5 + 1.0
    baseline = np.clip(baseline, 0, None)
    agent.train_models(baseline, epochs=30)
    
    print("\n✓ Models trained. Ready for manual input.\n")
    
    while True:
        print("\n--- Enter Feature Values (or 'quit' to exit) ---")
        
        entity_id = input("Entity ID (IMSI/IMEI): ").strip()
        if entity_id.lower() == 'quit':
            break
        
        try:
            # IPDR features
            print("\nIPDR Features:")
            bytes_sent = float(input("  bytes_sent (MB): "))
            bytes_received = float(input("  bytes_received (MB): "))
            domain_entropy = float(input("  domain_entropy (0-5): "))
            port_variety = float(input("  port_variety (0-100): "))
            protocol_freq = float(input("  protocol_frequency (0-1): "))
            session_dur = float(input("  session_duration (sec): "))
            vpn_usage = float(input("  vpn_usage (0/1): "))
            
            # CDR features
            print("\nCDR Features:")
            call_freq = float(input("  call_frequency (calls/hour): "))
            distinct_callees = float(input("  distinct_callees: "))
            repeat_ratio = float(input("  repeat_call_ratio (0-1): "))
            tower_speed = float(input("  tower_movement_speed (km/h): "))
            call_dur_std = float(input("  call_duration_std (sec): "))
            
            # Fused features
            print("\nFused Features:")
            imei_imsi_uniq = float(input("  imei_imsi_uniqueness (0-1): "))
            device_share = float(input("  device_sharing_score (0-1): "))
            geo_velocity = float(input("  geo_velocity_anomaly (km/h): "))
            behav_drift = float(input("  behavioral_drift (0-1): "))
            cross_corr = float(input("  cross_domain_corr (-1 to 1): "))
            
            # Create feature vector
            features = np.array([[
                bytes_sent, bytes_received, domain_entropy, port_variety,
                protocol_freq, session_dur, vpn_usage,
                call_freq, distinct_callees, repeat_ratio, tower_speed, call_dur_std,
                imei_imsi_uniq, device_share, geo_velocity, behav_drift, cross_corr
            ]])
            
            # Detect anomalies
            results = agent.detect_anomalies(features, [entity_id])
            result = results[0]
            
            # Display results
            print("\n" + "="*60)
            print("ANOMALY DETECTION RESULT")
            print("="*60)
            print(f"Entity ID: {result['entity_id']}")
            print(f"Anomaly Score: {result['anomaly_score']:.3f}")
            print(f"Anomaly Type: {result['anomaly_type']}")
            print(f"Subtype: {result['anomaly_subtype']}")
            print(f"Confidence: {result['confidence']:.3f}")
            
            print("\nModel Evidence:")
            for model, score in result['model_evidence'].items():
                print(f"  {model}: {score:.3f}")
            
            print("\nTop Contributing Features:")
            for feat in result['top_features']:
                print(f"  {feat['feature']}: {feat['contribution']:.3f}")
            
            # Get LLM explanation
            print("\n--- AI Explanation (via Ollama) ---")
            explanation = agent.explain_with_llm(result)
            print(explanation)
            print("="*60)
            
        except ValueError as e:
            print(f"\n✗ Invalid input: {e}")
        except Exception as e:
            print(f"\n✗ Error: {e}")

# ==================== BATCH PROCESSING ====================
def batch_process_file(file_path: str, agent: UnsupervisedAnomalyAgent):
    """Process CSV file of events"""
    df = pd.read_csv(file_path)
    
    entity_ids = df['entity_id'].tolist()
    features = df[agent.feature_names].values
    
    results = agent.detect_anomalies(features, entity_ids)
    
    # Save results
    output_df = pd.DataFrame(results)
    output_df.to_csv('anomaly_results.csv', index=False)
    print(f"✓ Results saved to anomaly_results.csv")
    
    return results

# ==================== SAMPLE DATA ====================
SAMPLE_SCENARIOS = {
    "normal_user": {
        "entity_id": "IMSI-404201111111111",
        "description": "Normal mobile user - daily commuter",
        "features": [
            50.0,    # bytes_sent (MB)
            45.0,    # bytes_received (MB)
            3.2,     # domain_entropy (diverse domains)
            15.0,    # port_variety
            0.85,    # protocol_frequency (mostly HTTPS)
            120.0,   # session_duration (sec)
            0.0,     # vpn_usage
            12.0,    # call_frequency (calls/hour)
            8.0,     # distinct_callees
            0.6,     # repeat_call_ratio (calls friends/family)
            25.0,    # tower_movement_speed (km/h - commuting)
            45.0,    # call_duration_std (varied conversations)
            1.0,     # imei_imsi_uniqueness
            0.0,     # device_sharing_score
            0.0,     # geo_velocity_anomaly
            0.1,     # behavioral_drift (stable behavior)
            0.7      # cross_domain_corr (normal correlation)
        ]
    },
    
    "data_exfiltration": {
        "entity_id": "IMSI-404202222222222",
        "description": "Malware-infected device exfiltrating data to C2 server",
        "features": [
            5000.0,  # bytes_sent (5GB - HUGE)
            500.0,   # bytes_received (500MB)
            0.1,     # domain_entropy (single suspicious domain)
            87.0,    # port_variety (scanning behavior)
            0.95,    # protocol_frequency (encrypted traffic)
            1800.0,  # session_duration (long sustained connection)
            1.0,     # vpn_usage (hiding traffic)
            3.0,     # call_frequency (very low - bot behavior)
            3.0,     # distinct_callees
            0.0,     # repeat_call_ratio (no repeated calls)
            0.0,     # tower_movement_speed (stationary)
            2.0,     # call_duration_std (minimal variance)
            1.0,     # imei_imsi_uniqueness
            0.0,     # device_sharing_score
            0.0,     # geo_velocity_anomaly
            0.9,     # behavioral_drift (sudden change)
            -0.4     # cross_domain_corr (high data, low calls)
        ]
    },
    
    "sim_farm": {
        "entity_id": "IMSI-404203333333333",
        "description": "SIM farm for spam calling/SMS fraud",
        "features": [
            10.0,    # bytes_sent (minimal data)
            8.0,     # bytes_received
            1.2,     # domain_entropy (limited domains)
            5.0,     # port_variety (only basic ports)
            0.3,     # protocol_frequency (mostly HTTP)
            30.0,    # session_duration (short sessions)
            0.0,     # vpn_usage
            250.0,   # call_frequency (EXTREMELY HIGH - automated)
            220.0,   # distinct_callees (all different numbers)
            0.02,    # repeat_call_ratio (no repeats - spam)
            0.0,     # tower_movement_speed (stationary in data center)
            1.5,     # call_duration_std (identical short calls)
            0.3,     # imei_imsi_uniqueness (multiple SIMs per device)
            0.9,     # device_sharing_score (device shared by many SIMs)
            0.0,     # geo_velocity_anomaly
            0.95,    # behavioral_drift (completely different)
            -0.6     # cross_domain_corr (low data, extremely high calls)
        ]
    },
    
    "device_cloning": {
        "entity_id": "IMSI-404204444444444",
        "description": "Cloned device operating simultaneously in multiple locations",
        "features": [
            80.0,    # bytes_sent
            75.0,    # bytes_received
            2.8,     # domain_entropy
            20.0,    # port_variety
            0.8,     # protocol_frequency
            150.0,   # session_duration
            0.0,     # vpn_usage
            18.0,    # call_frequency
            12.0,    # distinct_callees
            0.5,     # repeat_call_ratio
            450.0,   # tower_movement_speed (IMPOSSIBLE TRAVEL)
            50.0,    # call_duration_std
            0.2,     # imei_imsi_uniqueness (multiple IMEI per IMSI)
            0.0,     # device_sharing_score
            8.5,     # geo_velocity_anomaly (teleportation detected)
            0.7,     # behavioral_drift
            0.6      # cross_domain_corr
        ]
    },
    
    "account_takeover": {
        "entity_id": "IMSI-404205555555555",
        "description": "Legitimate account taken over by fraudster",
        "features": [
            850.0,   # bytes_sent (sudden spike in usage)
            800.0,   # bytes_received
            4.5,     # domain_entropy (accessing many new services)
            45.0,    # port_variety (exploring system)
            0.7,     # protocol_frequency
            600.0,   # session_duration (long reconnaissance)
            1.0,     # vpn_usage (hiding location)
            35.0,    # call_frequency (higher than normal)
            30.0,    # distinct_callees (calling new numbers)
            0.1,     # repeat_call_ratio (no familiar contacts)
            120.0,   # tower_movement_speed (different location)
            80.0,    # call_duration_std
            1.0,     # imei_imsi_uniqueness
            0.0,     # device_sharing_score
            3.2,     # geo_velocity_anomaly (sudden location change)
            0.85,    # behavioral_drift (behavior completely changed)
            0.3      # cross_domain_corr (unusual pattern)
        ]
    },
    
    "international_roaming": {
        "entity_id": "IMSI-404206666666666",
        "description": "Legitimate user traveling internationally (edge case)",
        "features": [
            120.0,   # bytes_sent (moderate usage abroad)
            110.0,   # bytes_received
            3.5,     # domain_entropy
            18.0,    # port_variety
            0.9,     # protocol_frequency
            180.0,   # session_duration
            1.0,     # vpn_usage (using VPN abroad)
            8.0,     # call_frequency (reduced calling)
            6.0,     # distinct_callees
            0.7,     # repeat_call_ratio (still calling family)
            850.0,   # tower_movement_speed (flight)
            60.0,    # call_duration_std
            1.0,     # imei_imsi_uniqueness
            0.0,     # device_sharing_score
            6.0,     # geo_velocity_anomaly (flight speed)
            0.4,     # behavioral_drift (some change expected)
            0.65     # cross_domain_corr
        ]
    },
    
    "iot_botnet": {
        "entity_id": "IMSI-404207777777777",
        "description": "IoT device compromised and part of DDoS botnet",
        "features": [
            2500.0,  # bytes_sent (constant DDoS traffic)
            100.0,   # bytes_received (minimal inbound)
            0.05,    # domain_entropy (attacking single target)
            120.0,   # port_variety (port scanning)
            0.4,     # protocol_frequency (mix of protocols)
            3600.0,  # session_duration (continuous connection)
            0.0,     # vpn_usage
            0.0,     # call_frequency (no calls - IoT device)
            0.0,     # distinct_callees
            0.0,     # repeat_call_ratio
            0.0,     # tower_movement_speed
            0.0,     # call_duration_std
            1.0,     # imei_imsi_uniqueness
            0.0,     # device_sharing_score
            0.0,     # geo_velocity_anomaly
            0.99,    # behavioral_drift (completely different from normal IoT)
            -0.9     # cross_domain_corr (data only, no voice)
        ]
    }
}

def run_sample_scenarios():
    """Run detection on all sample scenarios"""
    print("\n" + "="*70)
    print("RUNNING SAMPLE FRAUD DETECTION SCENARIOS")
    print("="*70)
    
    agent = UnsupervisedAnomalyAgent()
    
    # Generate synthetic baseline for training
    print("\n[1/3] Generating baseline normal data (1000 samples)...")
    np.random.seed(42)
    baseline = np.random.randn(1000, 17) * 0.3 + np.array([
        50, 45, 3.0, 15, 0.8, 120, 0.0,  # IPDR features
        10, 8, 0.6, 20, 45,                # CDR features  
        1.0, 0.0, 0.0, 0.1, 0.7            # Fused features
    ])
    baseline = np.clip(baseline, 0, None)
    
    print("[2/3] Training models (30 epochs)...")
    agent.train_models(baseline, epochs=30)
    
    print("\n[3/3] Running detection on scenarios...\n")
    
    # Prepare batch data
    entity_ids = []
    features_batch = []
    
    for scenario_name, scenario in SAMPLE_SCENARIOS.items():
        entity_ids.append(scenario['entity_id'])
        features_batch.append(scenario['features'])
    
    features_batch = np.array(features_batch)
    
    # Detect anomalies
    results = agent.detect_anomalies(features_batch, entity_ids)
    
    # Display results for each scenario
    for scenario_name, result in zip(SAMPLE_SCENARIOS.keys(), results):
        scenario_info = SAMPLE_SCENARIOS[scenario_name]
        
        print("\n" + "="*70)
        print(f"SCENARIO: {scenario_name.upper().replace('_', ' ')}")
        print("="*70)
        print(f"Description: {scenario_info['description']}")
        print(f"Entity ID: {result['entity_id']}")
        print(f"\n{'ANOMALY SCORE:':<20} {result['anomaly_score']:.3f}")
        
        # Visual score indicator
        score_bar = "█" * int(result['anomaly_score'] * 50)
        print(f"{'Score Visual:':<20} [{score_bar:<50}]")
        
        # Risk level
        if result['anomaly_score'] > 0.8:
            risk = "🔴 CRITICAL - Immediate Investigation Required"
        elif result['anomaly_score'] > 0.6:
            risk = "🟠 HIGH - Review Recommended"
        elif result['anomaly_score'] > 0.4:
            risk = "🟡 MEDIUM - Monitor Closely"
        else:
            risk = "🟢 LOW - Normal Behavior"
        print(f"{'Risk Level:':<20} {risk}")
        
        print(f"\n{'Anomaly Type:':<20} {result['anomaly_type']}")
        print(f"{'Subtype:':<20} {result['anomaly_subtype']}")
        print(f"{'Confidence:':<20} {result['confidence']:.3f}")
        print(f"{'Cluster ID:':<20} {result['cluster_id']}")
        
        print("\nModel Evidence:")
        for model, score in result['model_evidence'].items():
            bar = "█" * int(score * 30)
            print(f"  {model:<25} {score:.3f} [{bar}]")
        
        print("\nTop Contributing Features:")
        for feat in result['top_features']:
            contribution_bar = "▓" * int(feat['contribution'] * 40)
            print(f"  {feat['feature']:<30} {feat['contribution']:.3f} [{contribution_bar}]")
        
        # Get LLM explanation
        print("\n--- AI Explanation (via Ollama LLM) ---")
        try:
            explanation = agent.explain_with_llm(result)
            print(explanation)
        except Exception as e:
            print(f"[LLM unavailable: {e}]")
            # Provide rule-based explanation
            if result['anomaly_score'] > 0.8:
                print("High anomaly detected. Key indicators:")
                for feat in result['top_features'][:3]:
                    print(f"  • {feat['feature']} shows significant deviation")
    
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    scores = [r['anomaly_score'] for r in results]
    print(f"Total Scenarios: {len(results)}")
    print(f"Average Score: {np.mean(scores):.3f}")
    print(f"Highest Score: {max(scores):.3f} ({entity_ids[scores.index(max(scores))]})")
    print(f"Lowest Score: {min(scores):.3f} ({entity_ids[scores.index(min(scores))]})")
    print(f"\nScenarios by Risk Level:")
    print(f"  🔴 Critical (>0.8): {sum(1 for s in scores if s > 0.8)}")
    print(f"  🟠 High (0.6-0.8): {sum(1 for s in scores if 0.6 < s <= 0.8)}")
    print(f"  🟡 Medium (0.4-0.6): {sum(1 for s in scores if 0.4 < s <= 0.6)}")
    print(f"  🟢 Low (<0.4): {sum(1 for s in scores if s <= 0.4)}")
    
    print("\n" + "="*70)

# ==================== MAIN MENU ====================
def main_menu():
    """Main menu for agent interaction"""
    print("\n" + "="*70)
    print("UNSUPERVISED ANOMALY DETECTION AGENT FOR TELECOM FRAUD")
    print("="*70)
    print("\nSelect Mode:")
    print("  1. Run Sample Scenarios (Pre-loaded fraud examples)")
    print("  2. Manual Input (Enter your own feature values)")
    print("  3. Batch Process CSV File")
    print("  4. Exit")
    print("="*70)
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        run_sample_scenarios()
        input("\nPress Enter to return to menu...")
        main_menu()
    
    elif choice == "2":
        manual_detection_interface()
        input("\nPress Enter to return to menu...")
        main_menu()
    
    elif choice == "3":
        file_path = input("Enter CSV file path: ").strip()
        try:
            agent = UnsupervisedAnomalyAgent()
            print("Training models on baseline...")
            baseline = np.random.randn(1000, 17) * 0.3 + 1.0
            baseline = np.clip(baseline, 0, None)
            agent.train_models(baseline, epochs=30)
            
            results = batch_process_file(file_path, agent)
            print(f"✓ Processed {len(results)} records")
        except Exception as e:
            print(f"✗ Error: {e}")
        
        input("\nPress Enter to return to menu...")
        main_menu()
    
    elif choice == "4":
        print("\nExiting... Goodbye!")
        return
    
    else:
        print("Invalid choice. Please try again.")
        main_menu()

if __name__ == "__main__":
    main_menu()