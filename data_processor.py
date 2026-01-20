"""Quantum I-Ching 專案資料處理模組.

此模組負責將市場資料和易經卦象轉換為適合 LSTM 訓練的格式。
提供時間序列資料的標準化和序列生成功能。
"""

from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from config import HEXAGRAM_MAP
from iching_core import IChingCore


class QuantumDataset(Dataset):
    """量子易經資料集類別.

    PyTorch Dataset，用於提供 (features, hexagram_id, label) 三元組。
    """

    def __init__(
        self,
        X_num: torch.Tensor,
        X_hex: torch.Tensor,
        y: torch.Tensor
    ) -> None:
        """初始化資料集.

        Args:
            X_num: 數值特徵張量，形狀為 (N, sequence_length, num_features)
            X_hex: 卦象 ID 張量，形狀為 (N, sequence_length)
            y: 標籤張量，形狀為 (N, 1)
        """
        self.X_num = X_num
        self.X_hex = X_hex
        self.y = y

    def __len__(self) -> int:
        """返回資料集大小."""
        return len(self.X_num)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """獲取單一資料樣本.

        Args:
            idx: 樣本索引

        Returns:
            (數值特徵, 卦象 ID, 標籤) 三元組
        """
        return self.X_num[idx], self.X_hex[idx], self.y[idx]


class DataProcessor:
    """資料處理器類別.

    負責將原始市場資料和易經卦象轉換為適合 LSTM 訓練的格式。
    """

    def __init__(
        self,
        sequence_length: int = 10,
        prediction_window: int = 1
    ) -> None:
        """初始化資料處理器.

        Args:
            sequence_length: 回看期間長度（例如 10 天）
            prediction_window: 預測窗口長度（例如 1 天，表示預測明天）
        """
        self.sequence_length = sequence_length
        self.prediction_window = prediction_window
        self.scaler = StandardScaler()
        self.core = IChingCore()

    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """準備訓練資料.

        從 DataFrame 中提取特徵、卦象 ID 和標籤，並進行標準化和序列生成。

        Args:
            df: 包含市場資料和易經卦象的 DataFrame
                - 必須包含欄位：'Close', 'Volume', 'RVOL', 'Daily_Return'
                - 必須包含欄位：'Hexagram_Binary' 或 'Ritual_Sequence'

        Returns:
            (X_num, X_hex, y) 三元組：
            - X_num: 數值特徵陣列，形狀為 (N, sequence_length, num_features)
            - X_hex: 卦象 ID 陣列，形狀為 (N, sequence_length)
            - y: 標籤陣列，形狀為 (N, 1)

        Raises:
            ValueError: 如果必要欄位不存在或資料不足
        """
        # 複製資料以避免修改原始 DataFrame
        df = df.copy()

        # 檢查必要欄位
        required_numerical = ['Close', 'Volume', 'RVOL', 'Daily_Return']
        missing_numerical = [col for col in required_numerical if col not in df.columns]
        if missing_numerical:
            raise ValueError(
                f"缺少必要數值欄位: {missing_numerical}\n"
                f"可用欄位: {list(df.columns)}"
            )

        # 檢查卦象欄位
        has_hexagram_binary = 'Hexagram_Binary' in df.columns
        has_ritual_sequence = 'Ritual_Sequence' in df.columns

        if not (has_hexagram_binary or has_ritual_sequence):
            raise ValueError(
                "缺少卦象欄位。需要 'Hexagram_Binary' 或 'Ritual_Sequence'"
            )

        # 處理 NaN 值：刪除包含 NaN 的行
        initial_len = len(df)
        df = df.dropna(subset=required_numerical)
        if has_hexagram_binary:
            df = df.dropna(subset=['Hexagram_Binary'])
        if has_ritual_sequence:
            df = df.dropna(subset=['Ritual_Sequence'])

        dropped_count = initial_len - len(df)
        if dropped_count > 0:
            print(f"[INFO] 刪除了 {dropped_count} 筆包含 NaN 的資料")

        if len(df) < self.sequence_length + self.prediction_window:
            raise ValueError(
                f"資料不足。需要至少 {self.sequence_length + self.prediction_window} 筆資料，"
                f"實際只有 {len(df)} 筆"
            )

        # 提取數值特徵
        numerical_features = df[required_numerical].values

        # 標準化數值特徵（只對數值特徵進行標準化，不包括卦象 ID）
        numerical_features_scaled = self.scaler.fit_transform(numerical_features)

        # 提取卦象 ID
        hexagram_ids = []
        if has_hexagram_binary:
            # 從 Hexagram_Binary 獲取卦象 ID
            for binary_str in df['Hexagram_Binary']:
                hex_info = HEXAGRAM_MAP.get(str(binary_str), {"id": 0})
                hex_id = hex_info.get("id", 0)
                # 轉換為 0-indexed（1-64 -> 0-63）
                hexagram_ids.append(hex_id - 1 if hex_id > 0 else 0)
        else:
            # 從 Ritual_Sequence 計算卦象 ID
            for ritual_seq in df['Ritual_Sequence']:
                try:
                    ritual_list = [int(char) for char in str(ritual_seq)]
                    interpretation = self.core.interpret_sequence(ritual_list)
                    hex_id = interpretation['current_hex'].get("id", 0)
                    # 轉換為 0-indexed（1-64 -> 0-63）
                    hexagram_ids.append(hex_id - 1 if hex_id > 0 else 0)
                except (ValueError, KeyError):
                    hexagram_ids.append(0)  # 預設值

        hexagram_ids = np.array(hexagram_ids, dtype=np.int64)

        # 創建標籤（二分類：上漲=1，下跌=0）
        close_prices = df['Close'].values
        targets = []
        for i in range(len(df) - self.prediction_window):
            current_price = close_prices[i]
            future_price = close_prices[i + self.prediction_window]
            # 如果未來價格 > 當前價格，標籤為 1（上漲），否則為 0（下跌）
            target = 1 if future_price > current_price else 0
            targets.append(target)

        targets = np.array(targets, dtype=np.float32).reshape(-1, 1)

        # 生成序列（滑動窗口）
        sequences_num = []
        sequences_hex = []
        labels = []

        for i in range(len(df) - self.sequence_length - self.prediction_window + 1):
            # 數值特徵序列
            seq_num = numerical_features_scaled[i:i + self.sequence_length]
            sequences_num.append(seq_num)

            # 卦象 ID 序列
            seq_hex = hexagram_ids[i:i + self.sequence_length]
            sequences_hex.append(seq_hex)

            # 標籤（窗口結束後的下一個時間點）
            label_idx = i + self.sequence_length - 1
            if label_idx < len(targets):
                labels.append(targets[label_idx])
            else:
                # 如果超出範圍，使用最後一個標籤
                labels.append(targets[-1])

        # 轉換為 numpy 陣列
        X_num = np.array(sequences_num, dtype=np.float32)
        X_hex = np.array(sequences_hex, dtype=np.int64)
        y = np.array(labels, dtype=np.float32)

        print(
            f"[INFO] 資料準備完成：\n"
            f"  - 序列數量: {len(X_num)}\n"
            f"  - 序列長度: {self.sequence_length}\n"
            f"  - 數值特徵維度: {X_num.shape[2]}\n"
            f"  - 標籤分布: 上漲={np.sum(y == 1)}, 下跌={np.sum(y == 0)}"
        )

        return X_num, X_hex, y

    def split_data(
        self,
        X_num: np.ndarray,
        X_hex: np.ndarray,
        y: np.ndarray,
        train_split: float = 0.8
    ) -> Tuple[DataLoader, DataLoader]:
        """分割資料為訓練集和測試集.

        使用時間序列分割（不進行隨機打亂），前 80% 用於訓練，後 20% 用於測試。

        Args:
            X_num: 數值特徵陣列，形狀為 (N, sequence_length, num_features)
            X_hex: 卦象 ID 陣列，形狀為 (N, sequence_length)
            y: 標籤陣列，形狀為 (N, 1)
            train_split: 訓練集比例，預設為 0.8

        Returns:
            (train_loader, test_loader) 二元組
        """
        # 時間序列分割（不進行隨機打亂）
        split_idx = int(len(X_num) * train_split)

        X_num_train = X_num[:split_idx]
        X_hex_train = X_hex[:split_idx]
        y_train = y[:split_idx]

        X_num_test = X_num[split_idx:]
        X_hex_test = X_hex[split_idx:]
        y_test = y[split_idx:]

        # 轉換為 PyTorch 張量
        X_num_train_tensor = torch.tensor(X_num_train, dtype=torch.float32)
        X_hex_train_tensor = torch.tensor(X_hex_train, dtype=torch.long)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

        X_num_test_tensor = torch.tensor(X_num_test, dtype=torch.float32)
        X_hex_test_tensor = torch.tensor(X_hex_test, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

        # 創建 Dataset
        train_dataset = QuantumDataset(
            X_num_train_tensor,
            X_hex_train_tensor,
            y_train_tensor
        )
        test_dataset = QuantumDataset(
            X_num_test_tensor,
            X_hex_test_tensor,
            y_test_tensor
        )

        # 創建 DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True,  # 訓練時可以打亂
            num_workers=0  # Windows 上建議設為 0
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=False,  # 測試時不打亂
            num_workers=0
        )

        print(
            f"[INFO] 資料分割完成：\n"
            f"  - 訓練集: {len(train_dataset)} 筆\n"
            f"  - 測試集: {len(test_dataset)} 筆"
        )

        return train_loader, test_loader
