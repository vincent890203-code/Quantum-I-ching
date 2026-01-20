"""Quantum I-Ching 專案 LSTM 模型模組.

此模組定義結合數值特徵與易經卦象嵌入的混合式 LSTM 模型，
以及對應的訓練與評估流程。
"""

from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


class QuantumLSTM(nn.Module):
    """量子易經 LSTM 模型.

    結合卦象嵌入與數值技術指標，進行二分類預測（上漲/下跌）。
    """

    def __init__(
        self,
        num_hexagrams: int = 64,
        embedding_dim: int = 8,
        num_numerical_features: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        """初始化 QuantumLSTM 模型.

        Args:
            num_hexagrams: 卦象總數（預設 64）。
            embedding_dim: 卦象嵌入維度。
            num_numerical_features: 數值特徵數量（預設 4: Close, Volume, RVOL, Daily_Return）。
            hidden_dim: LSTM 隱藏層維度。
            num_layers: LSTM 堆疊層數。
            dropout: dropout 比例。
        """
        super().__init__()

        # 卦象嵌入層
        self.hex_embedding = nn.Embedding(num_hexagrams, embedding_dim)

        # LSTM 輸入維度 = 數值特徵 + 卦象嵌入
        input_dim: int = num_numerical_features + embedding_dim

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

    def forward(
        self,
        x_num: torch.Tensor,
        x_hex: torch.Tensor,
    ) -> torch.Tensor:
        """前向傳播.

        Args:
            x_num: 數值特徵，形狀為 (batch_size, seq_len, num_features)。
            x_hex: 卦象 ID，形狀為 (batch_size, seq_len)。

        Returns:
            預測機率，形狀為 (batch_size, 1)。
        """
        # 卦象嵌入: (batch_size, seq_len, embedding_dim)
        hex_embedded: torch.Tensor = self.hex_embedding(x_hex)

        # 特徵融合: 在最後一個維度拼接數值特徵與卦象嵌入
        x: torch.Tensor = torch.cat([x_num, hex_embedded], dim=2)

        # LSTM 輸出: output 形狀 (batch_size, seq_len, hidden_dim)
        output, _ = self.lstm(x)

        # 取最後一個時間步的輸出
        last_output: torch.Tensor = output[:, -1, :]

        # Dropout + 全連接 + Sigmoid
        out: torch.Tensor = self.dropout(last_output)
        out = self.fc(out)
        out = self.sigmoid(out)

        return out


class QuantumTrainer:
    """量子易經 LSTM 訓練器.

    負責模型的訓練與評估流程。
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.001,
        patience: int = 5,
        min_delta: float = 0.0,
    ) -> None:
        """初始化訓練器.

        Args:
            model: 要訓練的模型。
            learning_rate: 學習率。
            patience: early stopping 的容忍 epoch 數。
            min_delta: early stopping 判斷改善的最小差值。
        """
        self.device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.patience = patience
        self.min_delta = min_delta
        self.learning_rate = learning_rate

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 20,
    ) -> Dict[str, List[float]]:
        """訓練模型.

        Args:
            train_loader: 訓練資料 DataLoader。
            val_loader: 驗證資料 DataLoader。
            epochs: 訓練輪數。

        Returns:
            包含訓練與驗證損失歷史的字典：
            {
                "train_loss": [...],
                "val_loss": [...],
            }
        """
        import os

        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
        best_val_loss: float = float("inf")
        best_model_path: str = os.path.join("data", "best_model.pth")
        os.makedirs("data", exist_ok=True)

        # 印出訓練超參數
        print("[INFO] Training configuration:")
        print(f"  Device: {self.device}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Patience: {self.patience}")
        print(f"  Min delta: {self.min_delta}")
        print(f"  Epochs: {epochs}")

        epochs_without_improve: int = 0

        for epoch in range(1, epochs + 1):
            # 訓練階段
            self.model.train()
            train_loss_sum: float = 0.0
            train_batches: int = 0

            train_iter = tqdm(
                train_loader,
                desc=f"Epoch {epoch}/{epochs} [Train]",
                leave=False,
            )

            for batch in train_iter:
                x_num, x_hex, y = batch
                x_num = x_num.to(self.device)
                x_hex = x_hex.to(self.device)
                y = y.to(self.device)

                # 前向傳播
                outputs: torch.Tensor = self.model(x_num, x_hex)

                # 確保 y 形狀與輸出一致
                y = y.view_as(outputs)

                loss: torch.Tensor = self.criterion(outputs, y)

                # 反向傳播與更新
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss_sum += loss.item()
                train_batches += 1

                train_iter.set_postfix({"loss": loss.item()})

            avg_train_loss: float = train_loss_sum / max(train_batches, 1)

            # 驗證階段
            self.model.eval()
            val_loss_sum: float = 0.0
            val_batches: int = 0

            val_iter = tqdm(
                val_loader,
                desc=f"Epoch {epoch}/{epochs} [Val]",
                leave=False,
            )

            with torch.no_grad():
                for batch in val_iter:
                    x_num, x_hex, y = batch
                    x_num = x_num.to(self.device)
                    x_hex = x_hex.to(self.device)
                    y = y.to(self.device)

                    outputs = self.model(x_num, x_hex)
                    y = y.view_as(outputs)
                    loss = self.criterion(outputs, y)

                    val_loss_sum += loss.item()
                    val_batches += 1

                    val_iter.set_postfix({"loss": loss.item()})

            avg_val_loss: float = val_loss_sum / max(val_batches, 1)

            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(avg_val_loss)

            print(
                f"Epoch [{epoch}/{epochs}] - "
                f"Train Loss: {avg_train_loss:.4f} - "
                f"Val Loss: {avg_val_loss:.4f}"
            )

            # 儲存最佳模型與 early stopping 判斷
            if avg_val_loss < best_val_loss - self.min_delta:
                best_val_loss = avg_val_loss
                epochs_without_improve = 0
                torch.save(self.model.state_dict(), best_model_path)
                print(
                    f"[INFO] 新最佳模型已儲存至 {best_model_path} "
                    f"(Val Loss: {best_val_loss:.4f})"
                )
            else:
                epochs_without_improve += 1
                print(
                    f"[INFO] 驗證損失未改善，連續 {epochs_without_improve} 個 epoch"
                )

                if epochs_without_improve >= self.patience:
                    print(
                        "[EARLY STOPPING] 驗證損失在 "
                        f"{self.patience} 個 epoch 內未顯著改善，提前停止訓練。"
                    )
                    break

        return history

    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """在測試集上評估模型.

        Args:
            test_loader: 測試資料 DataLoader。

        Returns:
            包含平均損失與準確率的字典：
            {
                "loss": float,
                "accuracy": float,
            }
        """
        self.model.eval()
        test_loss_sum: float = 0.0
        test_batches: int = 0
        correct: int = 0
        total: int = 0

        with torch.no_grad():
            for batch in test_loader:
                x_num, x_hex, y = batch
                x_num = x_num.to(self.device)
                x_hex = x_hex.to(self.device)
                y = y.to(self.device)

                outputs: torch.Tensor = self.model(x_num, x_hex)
                y = y.view_as(outputs)
                loss: torch.Tensor = self.criterion(outputs, y)

                test_loss_sum += loss.item()
                test_batches += 1

                # 二分類：0/1
                predicted: torch.Tensor = (outputs >= 0.5).float()
                correct += (predicted == y).sum().item()
                total += y.numel()

        avg_loss: float = test_loss_sum / max(test_batches, 1)
        accuracy: float = correct / max(total, 1)

        print(
            f"[EVAL] Test Loss: {avg_loss:.4f} - "
            f"Accuracy: {accuracy:.4f}"
        )

        return {"loss": avg_loss, "accuracy": accuracy}


if __name__ == "__main__":
    """當前模組作為腳本執行時，進行簡單的訓練流程示範.

    流程:
        1. 使用 MarketDataLoader 載入預設標的資料。
        2. 使用 MarketEncoder 生成卦象與技術指標。
        3. 使用 DataProcessor 產生訓練/驗證 DataLoader。
        4. 使用 QuantumTrainer 進行訓練，並印出 Train / Val Loss。
    """
    from data_loader import MarketDataLoader
    from market_encoder import MarketEncoder
    from data_processor import DataProcessor
    from config import settings

    # 1. 載入與編碼資料
    loader = MarketDataLoader()
    default_symbol: str = (
        settings.TARGET_TICKERS[0] if settings.TARGET_TICKERS else "NVDA"
    )
    raw_data = loader.fetch_data(tickers=[default_symbol])

    if raw_data.empty:
        print(f"[ERROR] 無法獲取 {default_symbol} 的市場資料，訓練中止。")
    else:
        encoder = MarketEncoder()
        encoded_data = encoder.generate_hexagrams(raw_data)

        if encoded_data.empty:
            print(
                "[ERROR] 編碼後的資料為空，可能是資料天數不足 "
                "(至少需要 26 天)。訓練中止。"
            )
        else:
            # 2. 準備訓練資料
            processor = DataProcessor(sequence_length=10, prediction_window=1)
            try:
                X_num, X_hex, y = processor.prepare_data(encoded_data)
            except ValueError as e:
                print(f"[ERROR] 資料準備失敗: {e}")
                raise SystemExit(1)

            train_loader, val_loader = processor.split_data(X_num, X_hex, y)

            # 3. 建立模型與訓練器
            model = QuantumLSTM()
            trainer = QuantumTrainer(model)

            # 4. 執行訓練
            print(
                f"[INFO] 開始訓練 QuantumLSTM 模型，標的: {default_symbol}，"
                "sequence_length=10, prediction_window=1"
            )
            history = trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=20,
            )

            print("[INFO] 訓練完成。最後一個 epoch 的損失：")
            print(
                f"  Train Loss: {history['train_loss'][-1]:.4f}, "
                f"Val Loss: {history['val_loss'][-1]:.4f}"
            )

