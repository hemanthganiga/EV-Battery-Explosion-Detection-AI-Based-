import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from model import LSTMTemperaturePredictor
from data_loader import BatteryDataset
from anomaly_detection import AnomalyDetector
from cooling_control import CoolingController
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Temperature thresholds
CRITICAL_TEMP = 70.0      # EMERGENCY - Evacuate immediately
DANGER_TEMP = 55.0        # DANGER - Change battery immediately  
HIGH_TEMP = 45.0          # WARNING - Start cooling
MEDIUM_TEMP = 35.0        # CAUTION - Monitor closely

def log_alert(timestamp, temp, level, message):
    """Log temperature alerts with timestamp"""
    print(f"\n{'='*70}")
    print(f"[{timestamp}] {level}")
    print(f"Temperature: {temp:.2f}°C")
    print(f"Message: {message}")
    print(f"{'='*70}\n")

def check_temperature_alert(current_temp, predicted_temp, index):
    """Check temperature and issue appropriate alerts"""
    alerts = []
    
    # Check actual temperature
    if current_temp >= CRITICAL_TEMP:
        alerts.append({
            'level': '🔴🔴 EMERGENCY - IMMEDIATE EVACUATION REQUIRED!!!',
            'temp': current_temp,
            'action': 'EVACUATE',
            'message': f'⚠️  ACTUAL TEMPERATURE CRITICAL: {current_temp:.2f}°C\n'
                      f'    🔴 LEAVE THE VEHICLE IMMEDIATELY!\n'
                      f'    🔴 MOVE AWAY FROM THE BATTERY!\n'
                      f'    Battery temperature has reached EMERGENCY level!\n'
                      f'    EXTREME RISK of thermal runaway, explosion, and fire!\n'
                      f'    EVACUATE NOW - Move to safe distance (>100m away)'
        })
    elif current_temp >= DANGER_TEMP:
        alerts.append({
            'level': '🔴 DANGER - CHANGE BATTERY IMMEDIATELY!!!',
            'temp': current_temp,
            'action': 'CHANGE_BATTERY',
            'message': f'⚠️  ACTUAL TEMPERATURE CRITICAL: {current_temp:.2f}°C\n'
                      f'    🔴 REPLACE BATTERY RIGHT NOW!\n'
                      f'    Battery temperature is dangerously high!\n'
                      f'    DO NOT CONTINUE DRIVING!\n'
                      f'    Action: Pull over safely and replace the battery IMMEDIATELY!'
        })
    elif current_temp >= HIGH_TEMP:
        alerts.append({
            'level': '🟠 WARNING - COOLING SYSTEM ACTIVATED',
            'temp': current_temp,
            'action': 'START_COOLING',
            'message': f'⚠️  ACTUAL TEMPERATURE HIGH: {current_temp:.2f}°C\n'
                      f'    ✓ Cooling system started\n'
                      f'    Battery temperature is elevated!\n'
                      f'    Action: Monitor closely and reduce driving intensity!'
        })
    elif current_temp >= MEDIUM_TEMP:
        alerts.append({
            'level': '🟡 CAUTION - MONITOR TEMPERATURE',
            'temp': current_temp,
            'action': 'MONITOR',
            'message': f'⚠️  ACTUAL TEMPERATURE ELEVATED: {current_temp:.2f}°C\n'
                      f'    Action: Monitor closely and prepare for cooling!'
        })
    
    # Check predicted temperature for early warning
    if predicted_temp >= CRITICAL_TEMP:
        alerts.append({
            'level': '🔴🔴 PREDICTION - EVACUATION RECOMMENDED!',
            'temp': predicted_temp,
            'action': 'EVACUATE',
            'message': f'⚠️  PREDICTED TEMPERATURE CRITICAL: {predicted_temp:.2f}°C\n'
                      f'    Model predicts EMERGENCY temperature!\n'
                      f'    URGENT: Evacuate area NOW!\n'
                      f'    Take immediate preventive action!'
        })
    elif predicted_temp >= DANGER_TEMP:
        alerts.append({
            'level': '🔴 PREDICTION - BATTERY REPLACEMENT URGENT!',
            'temp': predicted_temp,
            'action': 'CHANGE_BATTERY',
            'message': f'⚠️  PREDICTED TEMPERATURE CRITICAL: {predicted_temp:.2f}°C\n'
                      f'    Model predicts dangerous temperature rise!\n'
                      f'    Action: Stop and replace battery immediately!'
        })
    elif predicted_temp >= HIGH_TEMP:
        alerts.append({
            'level': '🟠 PREDICTION WARNING - COOLING NEEDED',
            'temp': predicted_temp,
            'message': f'⚠️  PREDICTED TEMPERATURE HIGH: {predicted_temp:.2f}°C\n'
                      f'    Model predicts temperature will reach cooling threshold!\n'
                      f'    Action: Prepare cooling system!'
        })
    
    return alerts

def main():
    # Paths
    data_dir = 'data'
    metadata_path = 'metadata.csv'

    # Load data
    dataset = BatteryDataset(data_dir, metadata_path)
    df = dataset[0]  # Example: use first file
    print('Available columns:', df.columns)
    temp = df['Temperature_measured'].values.reshape(-1, 1)
    
    # Normalize temperature data for better training
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    temp_scaled = scaler.fit_transform(temp)
    print(f'Temperature range: {temp.min():.2f}°C to {temp.max():.2f}°C')
    print(f'Scaled range: {temp_scaled.min():.4f} to {temp_scaled.max():.4f}')

    # Prepare data for LSTM
    seq_len = 20  # Increased from 10 for better context
    X, y = [], []
    for i in range(len(temp_scaled) - seq_len):
        X.append(temp_scaled[i:i+seq_len])
        y.append(temp_scaled[i+seq_len])
    X = np.array(X)
    y = np.array(y)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    
    # Split into train/validation
    split_idx = int(0.8 * len(X_tensor))
    X_train, X_val = X_tensor[:split_idx], X_tensor[split_idx:]
    y_train, y_val = y_tensor[:split_idx], y_tensor[split_idx:]
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=16, shuffle=False)

    # Model
    model = LSTMTemperaturePredictor(input_size=1, hidden_size=64, num_layers=2, output_size=1)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Train with validation
    print("\n" + "="*70)
    print("TRAINING LSTM MODEL")
    print("="*70)
    best_val_loss = float('inf')
    patience_counter = 0
    
    model.train()
    for epoch in range(100):  # Increased from 2 to 100 epochs
        train_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb.unsqueeze(1) if yb.dim() == 1 else yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb)
                loss = criterion(pred, yb.unsqueeze(1) if yb.dim() == 1 else yb)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1:3d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print(f'Early stopping at epoch {epoch+1}')
                break
        
        model.train()
    
    print("="*70)

    # Predict
    model.eval()
    with torch.no_grad():
        pred_temp_scaled = model(X_tensor).numpy()
    
    # Inverse scaling to get actual temperature
    pred_temp = scaler.inverse_transform(pred_temp_scaled)
    
    # Create full prediction array (pad with original temps for first seq_len points)
    full_pred_temp = np.vstack([temp[:seq_len], pred_temp])

    # Anomaly Detection
    detector = AnomalyDetector()
    detector.fit(temp)
    anomalies = detector.predict(temp)

    # Cooling Control
    controller = CoolingController()
    
    # Temperature monitoring and alerts
    print("\n" + "="*80)
    print("BATTERY TEMPERATURE MONITORING & COOLING CONTROL SYSTEM")
    print("="*80)
    print(f"🟡 CAUTION Threshold:        {MEDIUM_TEMP}°C - Monitor temperature")
    print(f"🟠 WARNING Threshold:        {HIGH_TEMP}°C - START COOLING")
    print(f"🔴 DANGER Threshold:         {DANGER_TEMP}°C - CHANGE BATTERY IMMEDIATELY")
    print(f"🔴🔴 EVACUATION Threshold:   {CRITICAL_TEMP}°C - EVACUATE AREA NOW")
    print("="*80 + "\n")
    
    # Monitor temperatures and detect peaks
    high_temp_indices = []
    danger_temp_indices = []
    critical_temp_indices = []
    all_alerts = []
    
    for i in range(len(temp)):
        current_temp = temp[i][0]
        predicted_temp = full_pred_temp[i][0] if i < len(full_pred_temp) else 0
        
        # Update cooling controller
        cooling_on, cooling_level = controller.update(current_temp)
        
        # Check for alerts
        alerts = check_temperature_alert(current_temp, predicted_temp, i)
        
        if alerts:
            if any(alert['action'] == 'EVACUATE' for alert in alerts if 'action' in alert):
                critical_temp_indices.append(i)
            elif any(alert['action'] == 'CHANGE_BATTERY' for alert in alerts if 'action' in alert):
                danger_temp_indices.append(i)
            else:
                high_temp_indices.append(i)
            
            for alert in alerts:
                log_alert(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                         alert['temp'], alert['level'], alert['message'])
                all_alerts.append(alert)
    
    # Summary report
    print("\n" + "="*80)
    print("TEMPERATURE MONITORING SUMMARY REPORT")
    print("="*80)
    print(f"Total data points: {len(temp)}")
    print(f"Cooling events detected (>45°C): {len(high_temp_indices)}")
    print(f"Battery change events (>55°C): {len(danger_temp_indices)}")
    print(f"Evacuation events (>70°C): {len(critical_temp_indices)}")
    print(f"\nMax actual temperature: {temp.max():.2f}°C")
    print(f"Max predicted temperature: {full_pred_temp.max():.2f}°C")
    print(f"Average actual temperature: {temp.mean():.2f}°C")
    print(f"Average predicted temperature: {full_pred_temp.mean():.2f}°C")
    
    if critical_temp_indices:
        print(f"\n🔴🔴 CRITICAL: {len(critical_temp_indices)} EVACUATION events detected!")
    elif danger_temp_indices:
        print(f"\n🔴 WARNING: {len(danger_temp_indices)} BATTERY CHANGE events detected!")
    elif high_temp_indices:
        print(f"\n🟠 INFO: {len(high_temp_indices)} COOLING events detected!")
    else:
        print("\n✅ SAFE: No dangerous temperature events detected - Battery operating normally!")
    
    print("="*80 + "\n")

    # Plot with temperature zones and alerts
    plt.figure(figsize=(16, 8))
    plt.plot(temp, label='Actual Temp', linewidth=2.5, color='blue')
    plt.plot(full_pred_temp, label='Predicted Temp', linewidth=2.5, color='orange', alpha=0.8)
    
    # Add colored background zones
    plt.axhspan(0, MEDIUM_TEMP, alpha=0.1, color='green', label='Safe Zone')
    plt.axhspan(MEDIUM_TEMP, HIGH_TEMP, alpha=0.1, color='yellow', label='Caution Zone')
    plt.axhspan(HIGH_TEMP, DANGER_TEMP, alpha=0.1, color='orange', label='Cooling Zone')
    plt.axhspan(DANGER_TEMP, CRITICAL_TEMP, alpha=0.1, color='red', label='Danger Zone')
    plt.axhspan(CRITICAL_TEMP, 100, alpha=0.15, color='darkred', label='Evacuation Zone')
    
    # Add temperature threshold lines
    plt.axhline(y=CRITICAL_TEMP, color='darkred', linestyle='--', linewidth=2.5, label=f'EVACUATION ({CRITICAL_TEMP}°C)', alpha=0.9)
    plt.axhline(y=DANGER_TEMP, color='red', linestyle='--', linewidth=2.5, label=f'CHANGE BATTERY ({DANGER_TEMP}°C)', alpha=0.9)
    plt.axhline(y=HIGH_TEMP, color='orange', linestyle='--', linewidth=2.5, label=f'START COOLING ({HIGH_TEMP}°C)', alpha=0.9)
    plt.axhline(y=MEDIUM_TEMP, color='yellow', linestyle='--', linewidth=2, label=f'MONITOR ({MEDIUM_TEMP}°C)', alpha=0.7)
    
    # Mark anomalies
    anomaly_mask = anomalies == -1
    if anomaly_mask.any():
        plt.scatter(np.arange(len(temp))[anomaly_mask], temp[anomaly_mask], 
                   color='red', s=150, marker='X', label='Anomaly', zorder=5, edgecolors='darkred', linewidth=2)
    
    # Mark high temperature peaks (cooling level)
    if high_temp_indices:
        plt.scatter(high_temp_indices, temp[high_temp_indices], 
                   color='orange', s=180, marker='^', label='Cooling Event (45°C+)', zorder=5, edgecolors='darkorange', linewidth=1.5)
    
    # Mark danger temperature peaks (battery change)
    if danger_temp_indices:
        plt.scatter(danger_temp_indices, temp[danger_temp_indices], 
                   color='red', s=200, marker='s', label='Change Battery (55°C+)', zorder=5, edgecolors='darkred', linewidth=2)
    
    # Mark critical temperature peaks (evacuation)
    if critical_temp_indices:
        plt.scatter(critical_temp_indices, temp[critical_temp_indices], 
                   color='darkred', s=250, marker='*', label='EVACUATE (70°C+)', zorder=6, edgecolors='black', linewidth=2)
    
    plt.title('Battery Temperature Monitoring System\nWith Multi-Level Cooling Control & Safety Alerts', 
             fontsize=15, fontweight='bold')
    plt.xlabel('Time Index', fontsize=13)
    plt.ylabel('Temperature (°C)', fontsize=13)
    plt.legend(loc='upper left', fontsize=10, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()  