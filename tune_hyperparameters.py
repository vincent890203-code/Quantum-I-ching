"""Quantum I-Ching 專案超參數優化模組.

使用 Optuna 進行貝葉斯優化，尋找最佳超參數組合。
優化目標：最大化高信心 Precision（預測機率 > 0.65 時的準確率）。
"""

import json
import os
import random
from datetime import datetime
from typing import Dict

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from config import settings
from data_loader import MarketDataLoader
from data_processor import DataProcessor
from market_encoder import MarketEncoder
from model_lstm import QuantumLSTM


def set_random_seed(seed: int = 42) -> None:
    """設置隨機種子，確保實驗可重現."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_high_confidence_precision(
    model: nn.Module,
    val_loader: DataLoader,
    confidence_threshold: float = 0.65
) -> float:
    """計算高信心 Precision.
    
    Args:
        model: 訓練好的模型。
        val_loader: 驗證資料 DataLoader。
        confidence_threshold: 信心閾值（預設 0.65）。
    
    Returns:
        高信心 Precision（如果沒有高信心預測，返回 0.0）。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # 收集所有預測機率和真實標籤
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            
            outputs = model(x)
            y = y.view_as(outputs)
            
            # 收集預測機率和標籤
            all_probs.append(outputs.cpu().numpy().flatten())
            all_labels.append(y.cpu().numpy().flatten())
    
    # 合併所有批次
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    
    # 篩選高信心預測（預測機率 >= 閾值）
    high_confidence_mask = all_probs >= confidence_threshold
    num_high_conf = high_confidence_mask.sum()
    
    if num_high_conf == 0:
        # 如果沒有高信心預測，返回 0
        return 0.0
    
    # 計算 Precision（高信心預測中，實際為高波動的比例）
    high_conf_labels = all_labels[high_confidence_mask]
    precision = high_conf_labels.mean()  # 標籤為 1（高波動）的比例
    
    return float(precision)


def objective(trial: optuna.Trial) -> float:
    """Optuna 目標函數：最大化高信心 Precision.
    
    Args:
        trial: Optuna trial 物件。
    
    Returns:
        高信心 Precision 分數（要最大化）。
    """
    # 設置隨機種子以確保可重現性
    set_random_seed(42)
    
    # 建議超參數
    sequence_length = trial.suggest_categorical("sequence_length", [5, 10, 20, 30])
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256])
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    
    print(f"\n[Trial {trial.number}] 測試超參數組合：")
    print(f"  sequence_length: {sequence_length}")
    print(f"  hidden_dim: {hidden_dim}")
    print(f"  num_layers: {num_layers}")
    print(f"  dropout: {dropout:.4f}")
    print(f"  learning_rate: {learning_rate:.6f}")
    
    # 載入資料
    loader = MarketDataLoader()
    default_symbol = (
        settings.TARGET_TICKERS[0] if settings.TARGET_TICKERS else "NVDA"
    )
    raw_data = loader.fetch_data(tickers=[default_symbol])
    
    if raw_data.empty:
        print(f"[ERROR] 無法獲取 {default_symbol} 的市場資料")
        return 0.0
    
    encoder = MarketEncoder()
    encoded_data = encoder.generate_hexagrams(raw_data)
    
    if encoded_data.empty:
        print("[ERROR] 編碼後的資料為空")
        return 0.0
    
    # 準備資料（使用動態 sequence_length）
    processor = DataProcessor(sequence_length=sequence_length, prediction_window=5)
    try:
        X, y = processor.prepare_data(encoded_data)
    except ValueError as e:
        print(f"[ERROR] 資料準備失敗: {e}")
        return 0.0
    
    # 分割資料（80/20）
    split_idx = int(len(X) * 0.8)
    X_train = X[:split_idx]
    y_train = y[:split_idx]
    X_val = X[split_idx:]
    y_val = y[split_idx:]
    
    # 轉換為張量
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)
    
    # 創建 DataLoader
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 創建模型
    model = QuantumLSTM(
        num_features=9,  # 4 個數值特徵 + 5 個易經特徵
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)
    
    # 訓練設置
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=4, min_lr=1e-6, verbose=False
    )
    
    # 訓練模型（固定 epoch 數，帶 early stopping）
    best_val_loss = float("inf")
    patience_counter = 0
    max_patience = 10
    epochs = 20
    
    for epoch in range(1, epochs + 1):
        # 訓練階段
        model.train()
        train_loss_sum = 0.0
        train_batches = 0
        
        for batch in train_loader:
            x, y_batch = batch
            x = x.to(device)
            y_batch = y_batch.to(device)
            
            outputs = model(x)
            y_batch = y_batch.view_as(outputs)
            loss = criterion(outputs, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss_sum += loss.item()
            train_batches += 1
        
        avg_train_loss = train_loss_sum / max(train_batches, 1)
        
        # 驗證階段
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                x, y_batch = batch
                x = x.to(device)
                y_batch = y_batch.to(device)
                
                outputs = model(x)
                y_batch = y_batch.view_as(outputs)
                loss = criterion(outputs, y_batch)
                
                val_loss_sum += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss_sum / max(val_batches, 1)
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss - 0.0001:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                if epoch % 5 == 0:
                    print(f"  [Trial {trial.number}] Early stopping at epoch {epoch}")
                break
    
    # 計算高信心 Precision（評估指標）
    high_conf_precision = calculate_high_confidence_precision(
        model, val_loader, confidence_threshold=0.65
    )
    
    print(f"  [Trial {trial.number}] 高信心 Precision (>= 0.65): {high_conf_precision:.4f}")
    
    # 報告給 Optuna
    trial.report(high_conf_precision, epoch)
    
    # 如果 trial 應該被剪枝（pruning）
    if trial.should_prune():
        raise optuna.TrialPruned()
    
    return high_conf_precision


def main() -> None:
    """主函數：執行超參數優化."""
    print("=" * 80)
    print("Quantum I-Ching 超參數優化 (Hyperparameter Tuning)")
    print("=" * 80)
    print("\n目標：最大化高信心 Precision（預測機率 >= 0.65 時的準確率）")
    print("方法：Optuna 貝葉斯優化")
    print("Trials: 30-50 次試驗\n")
    
    # 創建 Optuna study
    study = optuna.create_study(
        direction="maximize",
        study_name="quantum_lstm_hyperparameter_optimization",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    
    # 運行優化
    print("開始超參數優化...\n")
    study.optimize(objective, n_trials=30, show_progress_bar=True)
    
    # 打印最佳結果
    print("\n" + "=" * 80)
    print("優化結果 (Optimization Results)")
    print("=" * 80)
    print(f"\n最佳試驗編號: {study.best_trial.number}")
    print(f"最佳高信心 Precision: {study.best_value:.4f}")
    print(f"\n最佳超參數組合:")
    for key, value in study.best_params.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    
    # 保存最佳參數到 JSON 文件
    config_dir = "config"
    os.makedirs(config_dir, exist_ok=True)
    best_params_path = os.path.join(config_dir, "best_params.json")
    
    # 準備保存的參數（轉換 numpy 類型為 Python 原生類型）
    best_params_to_save = {}
    for key, value in study.best_params.items():
        if isinstance(value, (np.integer, np.floating)):
            best_params_to_save[key] = float(value) if isinstance(value, np.floating) else int(value)
        else:
            best_params_to_save[key] = value
    
    # 添加元數據
    best_params_to_save["best_precision"] = float(study.best_value)
    best_params_to_save["best_trial_number"] = int(study.best_trial.number)
    best_params_to_save["optimization_date"] = datetime.now().isoformat()
    
    with open(best_params_path, 'w', encoding='utf-8') as f:
        json.dump(best_params_to_save, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 最佳參數已保存至: {best_params_path}")
    
    # 打印優化歷史摘要
    print("\n" + "=" * 80)
    print("優化歷史摘要 (Optimization History Summary)")
    print("=" * 80)
    print(f"\n總試驗次數: {len(study.trials)}")
    print(f"成功試驗: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    print(f"剪枝試驗: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    
    # 顯示前 5 個最佳結果
    print(f"\n前 5 個最佳結果:")
    sorted_trials = sorted(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE],
        key=lambda t: t.value,
        reverse=True
    )[:5]
    
    for i, trial in enumerate(sorted_trials, 1):
        print(f"\n  {i}. Trial {trial.number} - Precision: {trial.value:.4f}")
        for key, value in trial.params.items():
            if isinstance(value, float):
                print(f"     {key}: {value:.6f}")
            else:
                print(f"     {key}: {value}")


if __name__ == "__main__":
    main()
