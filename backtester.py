"""Quantum I-Ching 專案回測模組.

此模組使用已訓練的 QuantumLSTM 模型，
在未見過的測試資料上進行回測，評估策略績效。
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from config import settings
from data_loader import MarketDataLoader
from data_processor import DataProcessor
from market_encoder import MarketEncoder
from model_lstm import QuantumLSTM


class QuantumBacktester:
    """量子易經回測器.

    使用已訓練的 LSTM 模型對測試資料進行回測，
    評估策略相對於市場的表現。
    """

    def __init__(
        self,
        symbol: str,
        model_path: str = "data/best_model.pth",
    ) -> None:
        """初始化回測器.

        Args:
            symbol: 回測的標的股票代號。
            model_path: 已訓練模型的權重檔路徑。
        """
        self.symbol = symbol
        self.model_path = model_path
        self.device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # 檢查模型檔案是否存在
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"找不到模型檔案: {self.model_path}\n"
                "請先訓練模型並儲存為 data/best_model.pth。"
            )

        # 載入市場資料並生成卦象
        loader = MarketDataLoader()
        encoder = MarketEncoder()

        raw_data = loader.fetch_data(tickers=[symbol])
        if raw_data.empty:
            raise ValueError(f"無法獲取 {symbol} 的市場資料，回測無法進行。")

        encoded_data = encoder.generate_hexagrams(raw_data)
        if encoded_data.empty:
            raise ValueError(
                f"{symbol} 的編碼資料為空，請確認資料長度是否足夠。"
            )

        self.encoded_df = encoded_data

        # 初始化資料處理器
        self.processor = DataProcessor(sequence_length=10, prediction_window=1)

        # 初始化並載入模型
        self.model = QuantumLSTM().to(self.device)
        state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def _prepare_aligned_dataframe(self) -> pd.DataFrame:
        """建立與序列對齊的清洗後 DataFrame.

        這個方法複製 DataProcessor.prepare_data 的前處理邏輯，
        以確保序列與價格/日期對齊。

        Returns:
            清洗後且與序列對齊的 DataFrame。
        """
        df = self.encoded_df.copy()

        required_numerical: List[str] = [
            "Close",
            "Volume",
            "RVOL",
            "Daily_Return",
        ]
        has_hexagram_binary: bool = "Hexagram_Binary" in df.columns
        has_ritual_sequence: bool = "Ritual_Sequence" in df.columns

        # 刪除 NaN
        df = df.dropna(subset=required_numerical)
        if has_hexagram_binary:
            df = df.dropna(subset=["Hexagram_Binary"])
        if has_ritual_sequence:
            df = df.dropna(subset=["Ritual_Sequence"])

        return df

    def run_backtest(self) -> Dict[str, float]:
        """執行回測流程.

        Returns:
            包含策略績效指標的字典。
        """
        # ===== 步驟 1: 準備資料 =====
        df_clean = self._prepare_aligned_dataframe()

        # 由 DataProcessor 產生序列資料
        X_num, X_hex, y = self.processor.prepare_data(self.encoded_df)
        total_sequences: int = X_num.shape[0]

        # 使用與 DataProcessor.split_data 相同的拆分比例
        train_split: float = 0.8
        split_idx: int = int(total_sequences * train_split)

        # 測試集序列索引（全域）
        test_start: int = split_idx
        test_end: int = total_sequences
        test_indices: np.ndarray = np.arange(test_start, test_end)
        num_test_sequences: int = len(test_indices)

        if num_test_sequences == 0:
            raise ValueError("測試集序列數量為 0，無法進行回測。")

        # 建立測試集資料張量
        X_num_test = torch.tensor(
            X_num[test_start:test_end], dtype=torch.float32
        )
        X_hex_test = torch.tensor(
            X_hex[test_start:test_end], dtype=torch.long
        )
        y_test = torch.tensor(y[test_start:test_end], dtype=torch.float32)

        test_dataset = list(zip(X_num_test, X_hex_test, y_test))

        # ===== 步驟 2: 模型推論 =====
        self.model.eval()
        all_probs: List[float] = []

        with torch.no_grad():
            for x_num, x_hex, _ in test_dataset:
                x_num = x_num.unsqueeze(0).to(self.device)
                x_hex = x_hex.unsqueeze(0).to(self.device)

                outputs: torch.Tensor = self.model(x_num, x_hex)
                prob: float = float(outputs.squeeze().cpu().item())
                all_probs.append(prob)

        probs = np.array(all_probs, dtype=np.float32)
        signals = (probs > 0.5).astype(int)  # 1=持有, 0=空手

        # ===== 步驟 3: 計算報酬 =====
        closes = df_clean["Close"].to_numpy()
        index_array = np.arange(len(df_clean))

        # 每個序列對應的時間點索引（與 DataProcessor 的 label_idx 一致）
        seq_len: int = self.processor.sequence_length
        pred_window: int = self.processor.prediction_window

        # 驗證序列數量與 df_clean 長度關係（可選）
        expected_sequences = len(df_clean) - seq_len - pred_window + 1
        if expected_sequences != total_sequences:
            print(
                "[WARN] 序列數量與清洗後資料長度不一致，"
                f"expected={expected_sequences}, actual={total_sequences}"
            )

        # 測試集時間索引（在 df_clean 中的列位置）
        time_indices = test_indices + seq_len - 1  # label_idx 對應位置

        # 市場報酬：使用相鄰日報酬
        market_returns: List[float] = []
        for idx in time_indices:
            if idx <= 0:
                market_returns.append(0.0)
            else:
                prev_price = closes[idx - 1]
                curr_price = closes[idx]
                ret = (
                    (curr_price - prev_price) / prev_price
                    if prev_price != 0
                    else 0.0
                )
                market_returns.append(ret)

        market_returns_arr = np.array(market_returns, dtype=np.float32)

        # 策略報酬：t 的報酬由 t-1 的訊號決定（避免未來資訊洩漏）
        shifted_signals = np.concatenate(
            ([0], signals[:-1])
        )  # 第一天無部位
        strategy_returns_arr = market_returns_arr * shifted_signals

        # ===== 步驟 4: 累積績效 =====
        cumulative_market = np.cumprod(1.0 + market_returns_arr)
        cumulative_strategy = np.cumprod(1.0 + strategy_returns_arr)

        total_market_return = cumulative_market[-1] - 1.0
        total_strategy_return = cumulative_strategy[-1] - 1.0

        # 勝率：僅在有部位的日子上計算
        active_mask = shifted_signals == 1
        if active_mask.any():
            wins = (strategy_returns_arr[active_mask] > 0).sum()
            total_trades = active_mask.sum()
            win_rate = wins / total_trades
        else:
            win_rate = 0.0

        # ===== 步驟 5: 視覺化 =====
        dates = df_clean.index[time_indices]

        plt.figure(figsize=(10, 6))
        plt.plot(dates, cumulative_market, label="Market", alpha=0.7)
        plt.plot(dates, cumulative_strategy, label="Strategy", alpha=0.9)
        plt.title(f"Quantum I-Ching Backtest - {self.symbol}")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        os.makedirs("data", exist_ok=True)
        fig_path = os.path.join("data", "backtest_result.png")
        plt.savefig(fig_path)
        plt.close()

        print("\n=== Backtest Summary ===")
        print(f"Symbol: {self.symbol}")
        print(f"Total Market Return: {total_market_return * 100:.2f}%")
        print(f"Total Strategy Return: {total_strategy_return * 100:.2f}%")
        print(f"Win Rate: {win_rate * 100:.2f}%")
        print(f"Backtest figure saved to: {fig_path}\n")

        return {
            "total_market_return": float(total_market_return),
            "total_strategy_return": float(total_strategy_return),
            "win_rate": float(win_rate),
        }


if __name__ == "__main__":
    # 使用 settings 中的預設標的進行回測
    default_symbol: str = (
        settings.TARGET_TICKERS[0] if settings.TARGET_TICKERS else "NVDA"
    )

    try:
        backtester = QuantumBacktester(symbol=default_symbol)
        backtester.run_backtest()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
    except ValueError as e:
        print(f"[ERROR] {e}")
    except Exception as e:
        print(f"[ERROR] 發生未預期的錯誤: {e}")

