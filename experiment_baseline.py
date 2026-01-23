"""Quantum I-Ching 專案基準比較實驗模組.

此模組用於驗證易經特徵（卦象 Embedding）是否真的具有預測能力，
還是模型實作存在問題。

實驗設計：
1. Baseline Model (PureLSTM): 僅使用數值特徵
2. Quantum Model (QuantumLSTM): 使用數值特徵 + 雙流卦象 Embedding
3. 在相同資料、相同超參數下並行訓練，比較驗證損失和準確率
"""

import os
import random
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm.auto import tqdm

from config import settings
from data_loader import MarketDataLoader
from data_processor import DataProcessor
from market_encoder import MarketEncoder
from model_lstm import QuantumLSTM, QuantumTrainer


class PureLSTM(nn.Module):
    """純數值特徵 LSTM 模型（Baseline）.

    僅使用數值技術指標，不使用易經特徵。
    代表傳統技術分析方法。
    """

    def __init__(
        self,
        num_features: int = 4,  # 僅數值特徵：Close, Volume, RVOL, Daily_Return
        hidden_dim: int = 32,  # 降低到 32 以防止過擬合
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        """初始化 PureLSTM 模型.

        Args:
            num_features: 數值特徵數量（預設 4: Close, Volume, RVOL, Daily_Return）。
            hidden_dim: LSTM 隱藏層維度（預設 32）。
            num_layers: LSTM 堆疊層數。
            dropout: dropout 比例。
        """
        super().__init__()

        # 僅使用數值特徵，無卦象 Embedding
        input_dim: int = num_features

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向傳播（僅數值特徵）.

        Args:
            x: 數值特徵，形狀為 (batch_size, seq_len, num_features)。

        Returns:
            預測機率，形狀為 (batch_size, 1)。
        """
        # LSTM 輸出: output 形狀 (batch_size, seq_len, hidden_dim)
        output, _ = self.lstm(x)

        # 取最後一個時間步的輸出
        last_output: torch.Tensor = output[:, -1, :]

        # Dropout + 全連接 + Sigmoid
        out: torch.Tensor = self.dropout(last_output)
        out = self.fc(out)
        out = self.sigmoid(out)

        return out


def set_random_seed(seed: int = 42) -> None:
    """設置隨機種子，確保實驗可重現.

    Args:
        seed: 隨機種子值。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 20,
    learning_rate: float = 0.0005,
    weight_decay: float = 2e-5,
    patience: int = 15,
    min_delta: float = 0.0002,
    model_name: str = "Model",
) -> Dict[str, float]:
    """訓練模型並返回最終驗證指標.

    Args:
        model: 要訓練的模型。
        train_loader: 訓練資料 DataLoader。
        val_loader: 驗證資料 DataLoader。
        epochs: 訓練輪數。
        learning_rate: 學習率。
        weight_decay: L2 正則化係數。
        patience: early stopping 的容忍 epoch 數。
        min_delta: early stopping 判斷改善的最小差值。
        model_name: 模型名稱（用於日誌）。

    Returns:
        包含最終驗證損失和準確率的字典。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=4, min_lr=1e-6, verbose=False
    )

    best_val_loss = float("inf")
    epochs_without_improve = 0

    for epoch in range(1, epochs + 1):
        # 訓練階段
        model.train()
        train_loss_sum = 0.0
        train_batches = 0

        for batch in train_loader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            
            outputs = model(x)
            y = y.view_as(outputs)
            loss = criterion(outputs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_batches += 1

        avg_train_loss = train_loss_sum / max(train_batches, 1)

        # 驗證階段
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        correct = 0
        total = 0
        # 用於計算 Precision 和 Recall
        true_positive = 0
        false_positive = 0
        false_negative = 0

        with torch.no_grad():
            for batch in val_loader:
                x, y = batch
                x = x.to(device)
                y = y.to(device)
                
                outputs = model(x)
                y = y.view_as(outputs)
                loss = criterion(outputs, y)

                val_loss_sum += loss.item()
                val_batches += 1

                predicted = (outputs >= 0.5).float()
                correct += (predicted == y).sum().item()
                total += y.numel()
                
                # 計算 Precision 和 Recall（針對正類：高波動）
                # TP: 預測為高波動且實際為高波動
                # FP: 預測為高波動但實際為低波動
                # FN: 預測為低波動但實際為高波動
                true_positive += ((predicted == 1) & (y == 1)).sum().item()
                false_positive += ((predicted == 1) & (y == 0)).sum().item()
                false_negative += ((predicted == 0) & (y == 1)).sum().item()

        avg_val_loss = val_loss_sum / max(val_batches, 1)
        val_accuracy = correct / max(total, 1)
        
        # 計算 Precision 和 Recall
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1_score = 2 * (precision * recall) / max(precision + recall, 1e-8)

        scheduler.step(avg_val_loss)

        # Early stopping 判斷
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  [{model_name}] Epoch [{epoch}/{epochs}] - "
                f"Train Loss: {avg_train_loss:.4f} - "
                f"Val Loss: {avg_val_loss:.4f} - "
                f"Val Acc: {val_accuracy:.4f} - "
                f"Precision: {precision:.4f} - "
                f"Recall: {recall:.4f}"
            )

        if epochs_without_improve >= patience:
            print(f"  [{model_name}] Early stopping at epoch {epoch}")
            break

    return {
        "val_loss": best_val_loss,
        "val_accuracy": val_accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "final_train_loss": avg_train_loss,
        "final_val_loss": avg_val_loss,
    }


def analyze_confidence_tiers(
    model: nn.Module,
    val_loader: DataLoader,
    model_name: str = "Model",
    thresholds: list = [0.5, 0.55, 0.6, 0.65, 0.7]
) -> Dict[str, Dict[float, float]]:
    """分析不同信心閾值下的模型表現.
    
    Args:
        model: 訓練好的模型。
        val_loader: 驗證資料 DataLoader。
        model_name: 模型名稱（用於日誌）。
        thresholds: 要分析的信心閾值列表。
    
    Returns:
        包含每個閾值下指標的字典：
        {
            "num_trades": {threshold: count},
            "win_rate": {threshold: rate},
            "precision": {threshold: precision}
        }
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
    
    # 計算每個閾值下的指標
    results = {
        "num_trades": {},
        "win_rate": {},
        "precision": {}
    }
    
    for threshold in thresholds:
        # 篩選高信心預測（預測機率 >= 閾值）
        high_confidence_mask = all_probs >= threshold
        num_trades = high_confidence_mask.sum()
        
        if num_trades == 0:
            # 如果沒有高信心預測，設為 NaN
            results["num_trades"][threshold] = 0
            results["win_rate"][threshold] = float('nan')
            results["precision"][threshold] = float('nan')
        else:
            # 獲取高信心預測的標籤
            high_conf_labels = all_labels[high_confidence_mask]
            
            # 計算 Win Rate（這些信號的準確率）
            # 對於高信心預測，我們預測為高波動（label=1）
            # Win Rate = 實際為高波動的比例
            win_rate = high_conf_labels.mean()  # 標籤為 1 的比例
            
            # 計算 Precision（預測為高波動時，實際為高波動的比例）
            # 對於高信心預測（prob >= threshold），我們預測為高波動
            # Precision = 實際為高波動（label == 1）的比例
            precision = high_conf_labels.mean()  # 與 Win Rate 相同，因為我們預測為高波動
            
            results["num_trades"][threshold] = num_trades
            results["win_rate"][threshold] = win_rate
            results["precision"][threshold] = precision
    
    return results


def run_sanity_check() -> bool:
    """執行健全性檢查：驗證模型能否在小資料集上過擬合.

    Returns:
        True 如果通過檢查，False 如果失敗。
    """
    print("=" * 60)
    print("健全性檢查 (Sanity Check)")
    print("=" * 60)
    print("\n目標：驗證模型能否在小資料集（50 筆）上過擬合")
    print("預期：訓練損失應降至接近 0（< 0.01）")
    print(f"設定：200 epochs，預測 T+{settings.PREDICTION_WINDOW} 波動性突破")
    print(f"使用最佳參數：sequence_length={settings.SEQUENCE_LENGTH}\n")

    set_random_seed(42)

    # 載入資料
    loader = MarketDataLoader()
    default_symbol = (
        settings.TARGET_TICKERS[0] if settings.TARGET_TICKERS else "NVDA"
    )
    raw_data = loader.fetch_data(tickers=[default_symbol])

    if raw_data.empty:
        print(f"[ERROR] 無法獲取 {default_symbol} 的市場資料")
        return False

    encoder = MarketEncoder()
    encoded_data = encoder.generate_hexagrams(raw_data)

    if encoded_data.empty:
        print("[ERROR] 編碼後的資料為空")
        return False

    # 準備資料（使用最佳參數：sequence_length=30）
    processor = DataProcessor(sequence_length=settings.SEQUENCE_LENGTH, prediction_window=settings.PREDICTION_WINDOW)
    try:
        X, y = processor.prepare_data(encoded_data)
    except ValueError as e:
        print(f"[ERROR] 資料準備失敗: {e}")
        return False

    # 取前 50 筆作為小資料集
    tiny_size = min(50, len(X))
    X_tiny = X[:tiny_size]
    y_tiny = y[:tiny_size]

    # 轉換為張量
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.tensor(X_tiny, dtype=torch.float32)
    y_tensor = torch.tensor(y_tiny, dtype=torch.float32)

    # 檢查是否有 NaN
    if torch.isnan(X_tensor).any() or torch.isnan(y_tensor).any():
        print("[ERROR] 發現 NaN 值在張量中！")
        return False

    # 創建 DataLoader（使用相同資料作為訓練和驗證）
    dataset = TensorDataset(X_tensor, y_tensor)
    tiny_loader = DataLoader(dataset, batch_size=16, shuffle=False)

    # 檢查標籤分布
    label_dist = np.bincount(y_tiny.flatten().astype(int))
    print(f"[INFO] 標籤分布: 高波動={label_dist[1] if len(label_dist) > 1 else 0}, 低波動={label_dist[0]}")
    if len(label_dist) > 1 and (label_dist[0] == 0 or label_dist[1] == 0):
        print("[WARNING] 標籤完全不平衡！這會導致模型無法學習。")
        return False
    
    # 訓練模型 - 使用最佳超參數
    model = QuantumLSTM(
        num_features=9, 
        hidden_dim=settings.HIDDEN_DIM,  # 256
        num_layers=settings.NUM_LAYERS,  # 1
        dropout=settings.DROPOUT  # 0.35
    ).to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=settings.LEARNING_RATE)  # 0.001
    
    # 添加學習率調度器（可選，但先不用）
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    model.train()
    best_loss = float('inf')
    patience_counter = 0
    
    # 更新：增加到 200 epochs 以確保收斂
    for epoch in range(1, 201):
        epoch_loss = 0.0
        batch_count = 0

        for batch in tiny_loader:
            x, y_batch = batch
            x = x.to(device)
            y_batch = y_batch.to(device)

            # 再次檢查 NaN
            if torch.isnan(x).any() or torch.isnan(y_batch).any():
                print(f"[ERROR] Epoch {epoch}: 發現 NaN 值在 batch 中！")
                return False

            outputs = model(x)
            y_batch = y_batch.view_as(outputs)
            loss = criterion(outputs, y_batch)

            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪以防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

        avg_loss = epoch_loss / max(batch_count, 1)
        
        # 追蹤最佳損失
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch}: Train Loss = {avg_loss:.6f}, Best = {best_loss:.6f}")
        
        # 如果損失已經很低，提前停止
        if avg_loss < 0.001:
            print(f"  [SUCCESS] 損失已降至 {avg_loss:.6f} < 0.001，提前停止")
            break
        
        # 如果50個epoch沒有改善，提高學習率
        if patience_counter >= 50 and epoch < 150:
            current_lr = optimizer.param_groups[0]['lr']
            new_lr = current_lr * 2
            optimizer.param_groups[0]['lr'] = new_lr
            print(f"  [INFO] 提高學習率至 {new_lr:.6f}")
            patience_counter = 0

    final_loss = avg_loss

    print(f"\n最終訓練損失: {final_loss:.6f}")
    if final_loss < 0.01:
        print("[PASS] 健全性檢查通過：模型能夠在小資料集上過擬合")
        print("       這表示模型實作正確，能夠學習資料模式。\n")
        return True
    else:
        print("[FAIL] 健全性檢查失敗：模型無法在小資料集上過擬合")
        print("       可能原因：程式碼錯誤、資料問題、或模型架構問題。\n")
        return False


def run_comparison() -> Dict[str, Dict[str, float]]:
    """執行基準比較實驗.

    Returns:
        包含兩個模型結果的字典。
    """
    print("=" * 60)
    print("基準比較實驗 (Baseline Comparison)")
    print("=" * 60)
    print("\n目標：比較 QuantumLSTM（易經特徵）vs PureLSTM（僅數值特徵）")
    print("方法：特徵工程（手工特徵）替代 Embedding")
    print("預測目標：波動性突破（Volatility Breakout）")
    print("標籤定義：高波動 = |5天報酬率| > 3%, 低波動 = |5天報酬率| <= 3%")
    print("預測時間範圍：T+5 (5天後)")
    print("條件：相同資料、相同超參數、相同隨機種子\n")

    set_random_seed(42)

    # 載入資料
    loader = MarketDataLoader()
    default_symbol = (
        settings.TARGET_TICKERS[0] if settings.TARGET_TICKERS else "NVDA"
    )
    raw_data = loader.fetch_data(tickers=[default_symbol])

    if raw_data.empty:
        raise ValueError(f"無法獲取 {default_symbol} 的市場資料")

    encoder = MarketEncoder()
    encoded_data = encoder.generate_hexagrams(raw_data)

    if encoded_data.empty:
        raise ValueError("編碼後的資料為空")

    # 準備資料（使用最佳參數：sequence_length=30）
    processor = DataProcessor(sequence_length=settings.SEQUENCE_LENGTH, prediction_window=settings.PREDICTION_WINDOW)
    X, y = processor.prepare_data(encoded_data)

    # 分割資料（使用相同的分割邏輯）
    split_idx = int(len(X) * 0.8)

    X_train = X[:split_idx]
    y_train = y[:split_idx]

    X_val = X[split_idx:]
    y_val = y[split_idx:]

    # 轉換為張量
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # QuantumLSTM DataLoader（使用所有 9 個特徵）
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)

    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    quantum_train_dataset = TensorDataset(X_train_t, y_train_t)
    quantum_val_dataset = TensorDataset(X_val_t, y_val_t)

    quantum_train_loader = DataLoader(
        quantum_train_dataset, batch_size=32, shuffle=True
    )
    quantum_val_loader = DataLoader(
        quantum_val_dataset, batch_size=32, shuffle=False
    )

    # PureLSTM DataLoader（僅使用前 4 個數值特徵）
    # 創建只包含數值特徵的資料集
    X_train_numerical = X_train[:, :, :4]  # 只取前 4 個特徵
    X_val_numerical = X_val[:, :, :4]
    
    X_train_numerical_t = torch.tensor(X_train_numerical, dtype=torch.float32)
    X_val_numerical_t = torch.tensor(X_val_numerical, dtype=torch.float32)

    pure_train_dataset = TensorDataset(X_train_numerical_t, y_train_t)
    pure_val_dataset = TensorDataset(X_val_numerical_t, y_val_t)

    pure_train_loader = DataLoader(
        pure_train_dataset, batch_size=32, shuffle=True
    )
    pure_val_loader = DataLoader(
        pure_val_dataset, batch_size=32, shuffle=False
    )

    # 超參數（使用最佳參數）
    hyperparams = {
        "epochs": 20,
        "learning_rate": settings.LEARNING_RATE,  # 0.001
        "weight_decay": 1e-5,
        "patience": 10,
        "min_delta": 0.0001,
    }

    print(f"超參數設定（使用 Optuna 最佳參數）：")
    print(f"  Sequence Length: {settings.SEQUENCE_LENGTH} (月週期)")
    print(f"  Hidden Dim: {settings.HIDDEN_DIM}")
    print(f"  Num Layers: {settings.NUM_LAYERS}")
    print(f"  Dropout: {settings.DROPOUT}")
    print(f"  Epochs: {hyperparams['epochs']}")
    print(f"  Learning Rate: {hyperparams['learning_rate']}")
    print(f"  Weight Decay: {hyperparams['weight_decay']}")
    print(f"  Patience: {hyperparams['patience']}")
    print(f"  Min Delta: {hyperparams['min_delta']}\n")

    # 訓練 QuantumLSTM（使用最佳超參數）
    print("=" * 60)
    print("訓練 QuantumLSTM（特徵工程方法：數值特徵 + 易經手工特徵）")
    print("預測目標：波動性突破（高波動 vs 低波動）")
    print(f"使用最佳超參數：seq_len={settings.SEQUENCE_LENGTH}, hidden_dim={settings.HIDDEN_DIM}, "
          f"layers={settings.NUM_LAYERS}, dropout={settings.DROPOUT}")
    print("=" * 60)
    quantum_model = QuantumLSTM(
        num_features=9, 
        hidden_dim=settings.HIDDEN_DIM,  # 256
        num_layers=settings.NUM_LAYERS,  # 1
        dropout=settings.DROPOUT  # 0.35
    )
    quantum_results = train_model(
        quantum_model,
        quantum_train_loader,
        quantum_val_loader,
        model_name="Quantum",
        **hyperparams,
    )
    
    # 分析 QuantumLSTM 的信心閾值表現
    print("\n" + "=" * 60)
    print("QuantumLSTM 信心閾值分析 (Confidence Tier Analysis)")
    print("=" * 60)
    quantum_confidence = analyze_confidence_tiers(
        quantum_model,
        quantum_val_loader,
        model_name="Quantum",
        thresholds=[0.5, 0.55, 0.6, 0.65, 0.7]
    )
    
    print(f"\n{'閾值':<10} {'# 交易次數':<15} {'Win Rate':<15} {'Precision':<15}")
    print("-" * 60)
    for threshold in [0.5, 0.55, 0.6, 0.65, 0.7]:
        num_trades = quantum_confidence["num_trades"][threshold]
        win_rate = quantum_confidence["win_rate"][threshold]
        precision = quantum_confidence["precision"][threshold]
        
        win_rate_str = f"{win_rate:.4f}" if not np.isnan(win_rate) else "N/A"
        precision_str = f"{precision:.4f}" if not np.isnan(precision) else "N/A"
        
        print(f"{threshold:<10.2f} {num_trades:<15} {win_rate_str:<15} {precision_str:<15}")

    # 重置隨機種子，確保公平比較
    set_random_seed(42)

    # 訓練 PureLSTM（Baseline，使用相同的 sequence_length 以確保公平比較）
    print("\n" + "=" * 60)
    print("訓練 PureLSTM（僅數值特徵：Baseline）")
    print(f"使用相同 sequence_length={settings.SEQUENCE_LENGTH} 以確保公平比較")
    print("=" * 60)
    pure_model = PureLSTM(
        num_features=4, 
        hidden_dim=settings.HIDDEN_DIM,  # 256（與 QuantumLSTM 相同）
        num_layers=settings.NUM_LAYERS,  # 1
        dropout=settings.DROPOUT  # 0.35
    )
    pure_results = train_model(
        pure_model,
        pure_train_loader,
        pure_val_loader,
        model_name="Baseline",
        **hyperparams,
    )
    
    # 分析 PureLSTM 的信心閾值表現
    print("\n" + "=" * 60)
    print("PureLSTM 信心閾值分析 (Confidence Tier Analysis)")
    print("=" * 60)
    baseline_confidence = analyze_confidence_tiers(
        pure_model,
        pure_val_loader,
        model_name="Baseline",
        thresholds=[0.5, 0.55, 0.6, 0.65, 0.7]
    )
    
    print(f"\n{'閾值':<10} {'# 交易次數':<15} {'Win Rate':<15} {'Precision':<15}")
    print("-" * 60)
    for threshold in [0.5, 0.55, 0.6, 0.65, 0.7]:
        num_trades = baseline_confidence["num_trades"][threshold]
        win_rate = baseline_confidence["win_rate"][threshold]
        precision = baseline_confidence["precision"][threshold]
        
        win_rate_str = f"{win_rate:.4f}" if not np.isnan(win_rate) else "N/A"
        precision_str = f"{precision:.4f}" if not np.isnan(precision) else "N/A"
        
        print(f"{threshold:<10.2f} {num_trades:<15} {win_rate_str:<15} {precision_str:<15}")

    return {
        "quantum": quantum_results,
        "baseline": pure_results,
        "quantum_confidence": quantum_confidence,
        "baseline_confidence": baseline_confidence,
    }


def main() -> None:
    """主函數：執行健全性檢查和基準比較."""
    # 步驟 1: 健全性檢查
    sanity_passed = run_sanity_check()

    if not sanity_passed:
        print("[WARNING] 健全性檢查失敗！")
        print("建議：檢查模型實作、資料處理流程或程式碼錯誤。")
        print("實驗將繼續執行，但結果可能不可靠。\n")
    else:
        print()

    # 步驟 2: 基準比較
    try:
        results = run_comparison()

        # 步驟 3: 結果比較
        print("\n" + "=" * 60)
        print("實驗結果比較")
        print("=" * 60)

        quantum_val_loss = results["quantum"]["val_loss"]
        quantum_val_acc = results["quantum"]["val_accuracy"]
        quantum_precision = results["quantum"]["precision"]
        quantum_recall = results["quantum"]["recall"]
        quantum_f1 = results["quantum"]["f1_score"]
        
        baseline_val_loss = results["baseline"]["val_loss"]
        baseline_val_acc = results["baseline"]["val_accuracy"]
        baseline_precision = results["baseline"]["precision"]
        baseline_recall = results["baseline"]["recall"]
        baseline_f1 = results["baseline"]["f1_score"]

        print(f"\n{'指標':<20} {'QuantumLSTM':<20} {'PureLSTM (Baseline)':<20}")
        print("-" * 80)
        print(f"{'驗證損失':<20} {quantum_val_loss:<20.4f} {baseline_val_loss:<20.4f}")
        print(f"{'驗證準確率':<20} {quantum_val_acc:<20.4f} {baseline_val_acc:<20.4f}")
        print(f"{'Precision (高波動)':<20} {quantum_precision:<20.4f} {baseline_precision:<20.4f}")
        print(f"{'Recall (高波動)':<20} {quantum_recall:<20.4f} {baseline_recall:<20.4f}")
        print(f"{'F1-Score':<20} {quantum_f1:<20.4f} {baseline_f1:<20.4f}")

        improvement_loss = baseline_val_loss - quantum_val_loss
        improvement_acc = quantum_val_acc - baseline_val_acc
        improvement_precision = quantum_precision - baseline_precision
        improvement_recall = quantum_recall - baseline_recall

        print(f"\n改善幅度：")
        print(f"  驗證損失改善: {improvement_loss:+.4f} ({improvement_loss/baseline_val_loss*100:+.2f}%)")
        print(f"  驗證準確率改善: {improvement_acc:+.4f} ({improvement_acc/baseline_val_acc*100:+.2f}%)")
        print(f"  Precision 改善: {improvement_precision:+.4f} ({improvement_precision/baseline_precision*100:+.2f}%)")
        print(f"  Recall 改善: {improvement_recall:+.4f} ({improvement_recall/baseline_recall*100:+.2f}%)")

        # 結論
        print("\n" + "=" * 60)
        print("結論 (Conclusion)")
        print("=" * 60)

        if quantum_val_loss < baseline_val_loss:
            print("[SUCCESS] QuantumLSTM 優於 Baseline")
            print(f"   易經特徵（手工特徵工程）具有預測波動性突破的能力")
            print(f"   驗證損失降低了 {abs(improvement_loss):.4f} ({abs(improvement_loss/baseline_val_loss*100):.2f}%)")
            if improvement_recall > 0:
                print(f"   ⭐ Recall 提升 {improvement_recall:.4f} - 能更好地捕捉高波動事件")
            if improvement_precision > 0:
                print(f"   ⭐ Precision 提升 {improvement_precision:.4f} - 預測的高波動更準確")
        elif quantum_val_loss > baseline_val_loss:
            print("❌ Baseline 優於 QuantumLSTM")
            print(f"   易經特徵可能沒有預測波動性突破的能力，或需要進一步優化")
            print(f"   驗證損失增加了 {abs(improvement_loss):.4f} ({abs(improvement_loss/baseline_val_loss*100):.2f}%)")
        else:
            print("➖ 兩個模型表現相當")
            print(f"   易經特徵的預測能力有限，或需要調整模型架構")
        
        print(f"\n💡 解讀：")
        print(f"   - Precision: 預測為高波動時，實際為高波動的比例（越高越好）")
        print(f"   - Recall: 實際高波動事件中，被正確預測的比例（越高越好）")
        print(f"   - 對於波動性策略，Recall 更重要（不能錯過大波動）")

        # 步驟 4: 信心閾值比較分析
        print("\n" + "=" * 80)
        print("信心閾值比較分析 (Confidence Tier Comparison)")
        print("=" * 80)
        print("\n目標：驗證高信心預測是否具有更高的準確率")
        print("假設：如果 QuantumLSTM 的 Win Rate 隨閾值增加而上升，")
        print("      說明易經特徵在高信心區間具有 Alpha\n")
        
        quantum_confidence = results.get("quantum_confidence", {})
        baseline_confidence = results.get("baseline_confidence", {})
        
        if quantum_confidence and baseline_confidence:
            print(f"{'閾值':<10} {'QuantumLSTM':<40} {'PureLSTM (Baseline)':<40}")
            print(f"{'':<10} {'# 交易':<12} {'Win Rate':<12} {'Precision':<12} {'# 交易':<12} {'Win Rate':<12} {'Precision':<12}")
            print("-" * 80)
            
            for threshold in [0.5, 0.55, 0.6, 0.65, 0.7]:
                q_trades = quantum_confidence["num_trades"].get(threshold, 0)
                q_win_rate = quantum_confidence["win_rate"].get(threshold, float('nan'))
                q_precision = quantum_confidence["precision"].get(threshold, float('nan'))
                
                b_trades = baseline_confidence["num_trades"].get(threshold, 0)
                b_win_rate = baseline_confidence["win_rate"].get(threshold, float('nan'))
                b_precision = baseline_confidence["precision"].get(threshold, float('nan'))
                
                q_win_rate_str = f"{q_win_rate:.4f}" if not np.isnan(q_win_rate) else "N/A"
                q_precision_str = f"{q_precision:.4f}" if not np.isnan(q_precision) else "N/A"
                b_win_rate_str = f"{b_win_rate:.4f}" if not np.isnan(b_win_rate) else "N/A"
                b_precision_str = f"{b_precision:.4f}" if not np.isnan(b_precision) else "N/A"
                
                print(f"{threshold:<10.2f} {q_trades:<12} {q_win_rate_str:<12} {q_precision_str:<12} "
                      f"{b_trades:<12} {b_win_rate_str:<12} {b_precision_str:<12}")
            
            # 分析趨勢
            print("\n" + "-" * 80)
            print("趨勢分析：")
            
            # 計算 Win Rate 的斜率（從 0.5 到 0.7）
            quantum_win_rates = []
            baseline_win_rates = []
            thresholds_list = [0.5, 0.55, 0.6, 0.65, 0.7]
            
            for threshold in thresholds_list:
                q_wr = quantum_confidence["win_rate"].get(threshold, float('nan'))
                b_wr = baseline_confidence["win_rate"].get(threshold, float('nan'))
                if not np.isnan(q_wr):
                    quantum_win_rates.append(q_wr)
                if not np.isnan(b_wr):
                    baseline_win_rates.append(b_wr)
            
            if len(quantum_win_rates) >= 2 and len(baseline_win_rates) >= 2:
                # 計算簡單線性趨勢（最後值 - 第一個值）
                quantum_slope = quantum_win_rates[-1] - quantum_win_rates[0] if len(quantum_win_rates) >= 2 else 0
                baseline_slope = baseline_win_rates[-1] - baseline_win_rates[0] if len(baseline_win_rates) >= 2 else 0
                
                print(f"  QuantumLSTM Win Rate 變化: {quantum_win_rates[0]:.4f} → {quantum_win_rates[-1]:.4f} "
                      f"(斜率: {quantum_slope:+.4f})")
                print(f"  PureLSTM Win Rate 變化: {baseline_win_rates[0]:.4f} → {baseline_win_rates[-1]:.4f} "
                      f"(斜率: {baseline_slope:+.4f})")
                
                if quantum_slope > baseline_slope and quantum_slope > 0:
                    print(f"\n  [SUCCESS] 驗證假設：QuantumLSTM 的 Win Rate 隨信心閾值增加而上升")
                    print(f"     這證明易經特徵在高信心區間具有 Alpha（超額收益）")
                    print(f"     建議：使用更高的信心閾值（如 0.65-0.7）進行實際交易")
                elif quantum_slope > 0:
                    print(f"\n  ⚠️  QuantumLSTM 的 Win Rate 有上升趨勢，但改善幅度有限")
                else:
                    print(f"\n  ❌ QuantumLSTM 的 Win Rate 未隨信心閾值增加而上升")
                    print(f"     易經特徵可能不具備高信心 Alpha")
        else:
            print("[WARNING] 無法獲取信心閾值分析結果")

        if abs(improvement_loss) < 0.001:
            print("\n⚠️  注意：改善幅度很小（< 0.001），可能沒有統計顯著性")
            print("   建議：")
            print("   1. 增加訓練資料量")
            print("   2. 使用交叉驗證進行更嚴格的評估")
            print("   3. 檢查特徵工程是否正確")

    except Exception as e:
        print(f"\n[ERROR] 實驗執行失敗: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
