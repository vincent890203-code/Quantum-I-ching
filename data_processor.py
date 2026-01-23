"""Quantum I-Ching 專案資料處理模組.

此模組負責將市場資料和易經卦象轉換為適合 LSTM 訓練的格式。
提供時間序列資料的標準化和序列生成功能。
"""

import sys
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

    PyTorch Dataset，用於提供 (X, y) 二元組。
    使用特徵工程方法（數值特徵 + 易經手工特徵）。
    """

    def __init__(
        self,
        X: torch.Tensor,
        y: torch.Tensor
    ) -> None:
        """初始化資料集.

        Args:
            X: 合併特徵張量，形狀為 (N, sequence_length, num_features)
            y: 標籤張量，形狀為 (N, 1)
        """
        self.X = X
        self.y = y

    def __len__(self) -> int:
        """返回資料集大小."""
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """獲取單一資料樣本.

        Args:
            idx: 樣本索引

        Returns:
            (特徵, 標籤) 二元組
        """
        return self.X[idx], self.y[idx]


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

    def extract_iching_features(self, ritual_sequence: str) -> np.ndarray:
        """提取易經數值特徵.

        從儀式數字序列中提取手工特徵，用於替代 Embedding 方法。

        Args:
            ritual_sequence: 儀式數字序列字串（例如 "987896"），長度為 6。

        Returns:
            包含 5 個特徵的 numpy 陣列：
            - Yang_Count_Main: 主卦中陽線數量（7, 9 的數量）
            - Yang_Count_Future: 未來卦中陽線數量（轉換後的陽線數量）
            - Moving_Lines_Count: 動爻數量（6, 9 的數量）
            - Energy_Delta: 能量變化（Yang_Count_Future - Yang_Count_Main）
            - Conflict_Score: 衝突分數（上卦和下卦和的絕對差值）

        Raises:
            ValueError: 如果 ritual_sequence 長度不等於 6
        """
        if len(ritual_sequence) != 6:
            raise ValueError(
                f"儀式數字序列必須包含恰好 6 個元素，實際得到 {len(ritual_sequence)} 個"
            )

        # 轉換為整數列表
        ritual_nums = [int(char) for char in ritual_sequence]

        # 1. Yang_Count_Main: 主卦中陽線數量（7, 9 的數量）
        yang_count_main = sum(1 for num in ritual_nums if num in [7, 9])

        # 2. 計算未來卦的二進制字串
        future_binary = self.core.calculate_future_hexagram(ritual_nums)
        
        # Yang_Count_Future: 未來卦中陽線數量（"1" 的數量）
        yang_count_future = sum(1 for char in future_binary if char == "1")

        # 3. Moving_Lines_Count: 動爻數量（6, 9 的數量）
        moving_lines_count = sum(1 for num in ritual_nums if num in [6, 9])

        # 4. Energy_Delta: 能量變化
        energy_delta = yang_count_future - yang_count_main

        # 5. Conflict_Score: 上卦和下卦和的絕對差值
        # 上卦：前3爻（索引 0-2），下卦：後3爻（索引 3-5）
        lower_trigram_sum = sum(ritual_nums[:3])  # 下卦（底部3爻）
        upper_trigram_sum = sum(ritual_nums[3:])  # 上卦（頂部3爻）
        conflict_score = abs(upper_trigram_sum - lower_trigram_sum)

        return np.array([
            float(yang_count_main),
            float(yang_count_future),
            float(moving_lines_count),
            float(energy_delta),
            float(conflict_score)
        ], dtype=np.float32)

    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """準備訓練資料（特徵工程方法）.

        從 DataFrame 中提取數值特徵和易經手工特徵，並進行標準化和序列生成。
        使用特徵工程方法替代 Embedding 方法。
        
        重要：所有特徵（數值 + 易經）都會被標準化，確保尺度一致。

        Args:
            df: 包含市場資料和易經卦象的 DataFrame
                - 必須包含欄位：'Close', 'Volume', 'RVOL', 'Daily_Return'
                - 必須包含欄位：'Ritual_Sequence'

        Returns:
            (X, y) 二元組：
            - X: 合併特徵陣列，形狀為 (N, sequence_length, num_features)
                - 數值特徵：Close, Volume, RVOL, Daily_Return（4 個特徵）
                - 易經特徵：Yang_Count_Main, Yang_Count_Future, Moving_Lines_Count, 
                            Energy_Delta, Conflict_Score（5 個特徵）
                - 總共 9 個特徵，全部標準化
            - y: 標籤陣列，形狀為 (N, 1)，預測 T+5 趨勢

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
        if 'Ritual_Sequence' not in df.columns:
            raise ValueError("缺少卦象欄位。需要 'Ritual_Sequence'")

        # 處理 NaN 值：刪除包含 NaN 的行
        initial_len = len(df)
        df = df.dropna(subset=required_numerical + ['Ritual_Sequence'])

        dropped_count = initial_len - len(df)
        if dropped_count > 0:
            print(f"[INFO] 刪除了 {dropped_count} 筆包含 NaN 的資料")

        # 檢查資料是否足夠（需要 sequence_length + prediction_window + 額外的5天用於標籤）
        min_required = self.sequence_length + self.prediction_window + 4  # +4 因為標籤需要 T+5
        if len(df) < min_required:
            raise ValueError(
                f"資料不足。需要至少 {min_required} 筆資料（sequence_length={self.sequence_length}, "
                f"prediction_window={self.prediction_window}, 標籤需要 T+5），"
                f"實際只有 {len(df)} 筆"
            )

        # 提取數值特徵
        numerical_features = df[required_numerical].values
        print(f"[INFO] 提取數值特徵完成，形狀: {numerical_features.shape}")
        sys.stdout.flush()

        # 提取易經特徵
        print("[INFO] 開始提取易經特徵...")
        sys.stdout.flush()
        iching_features_list = []
        for idx, ritual_seq in enumerate(df['Ritual_Sequence']):
            if pd.isna(ritual_seq) or len(str(ritual_seq)) != 6:
                # 如果無效，使用零特徵
                iching_features_list.append(np.zeros(5, dtype=np.float32))
            else:
                try:
                    iching_features = self.extract_iching_features(str(ritual_seq))
                    iching_features_list.append(iching_features)
                except (ValueError, TypeError) as e:
                    print(f"[WARNING] 無法提取易經特徵: {ritual_seq}, 錯誤: {e}")
                    sys.stdout.flush()
                    iching_features_list.append(np.zeros(5, dtype=np.float32))
            
            # 每處理1000筆顯示進度
            if (idx + 1) % 1000 == 0:
                print(f"[INFO] 已處理 {idx + 1}/{len(df)} 筆易經特徵")
                sys.stdout.flush()

        iching_features = np.array(iching_features_list, dtype=np.float32)
        print(f"[INFO] 提取易經特徵完成，形狀: {iching_features.shape}")
        sys.stdout.flush()

        # 合併所有特徵（數值特徵 + 易經特徵）
        all_features = np.hstack([numerical_features, iching_features])
        print(f"[INFO] 合併特徵完成，總形狀: {all_features.shape}")
        sys.stdout.flush()

        # 檢查合併後是否有 NaN
        if np.isnan(all_features).any():
            nan_rows = np.isnan(all_features).any(axis=1)
            nan_count = nan_rows.sum()
            print(f"[WARNING] 發現 {nan_count} 筆包含 NaN 的特徵，將被刪除")
            all_features = all_features[~nan_rows]
            df = df[~nan_rows].reset_index(drop=True)
            if len(df) < min_required:
                raise ValueError(
                    f"刪除 NaN 後資料不足。需要至少 {min_required} 筆資料，"
                    f"實際只有 {len(df)} 筆"
                )

        # 定義特徵名稱（用於報告）
        feature_names = required_numerical + [
            'Yang_Count_Main', 'Yang_Count_Future', 'Moving_Lines_Count', 
            'Energy_Delta', 'Conflict_Score'
        ]
        
        print("[INFO] 開始檢查常數特徵...")
        sys.stdout.flush()
        # 檢查並處理常數特徵（方差為0）
        feature_vars = np.var(all_features, axis=0)
        constant_features = []
        for i, var in enumerate(feature_vars):
            if var < 1e-8:  # 方差接近0，視為常數特徵
                constant_features.append(i)
                print(f"[WARNING] 特徵 {feature_names[i]} 是常數特徵（方差={var:.8f}），將跳過標準化")
                sys.stdout.flush()
        
        print(f"[INFO] 常數特徵檢查完成，發現 {len(constant_features)} 個常數特徵")
        sys.stdout.flush()
        
        # 對於常數特徵，直接設為0（標準化後應該為0）
        all_features_scaled = np.zeros_like(all_features)
        
        # 只標準化非常數特徵
        print("[INFO] 開始標準化特徵...")
        sys.stdout.flush()
        if len(constant_features) < len(feature_vars):
            non_constant_mask = ~np.isin(np.arange(len(feature_vars)), constant_features)
            non_constant_features = all_features[:, non_constant_mask]
            
            # 標準化非常數特徵
            scaled_non_constant = self.scaler.fit_transform(non_constant_features)
            all_features_scaled[:, non_constant_mask] = scaled_non_constant
            
            # 常數特徵設為0（標準化後mean=0）
            for idx in constant_features:
                all_features_scaled[:, idx] = 0.0
        else:
            # 所有特徵都是常數，全部設為0
            print("[WARNING] 所有特徵都是常數！")
            sys.stdout.flush()
            all_features_scaled = np.zeros_like(all_features)
        
        print("[INFO] 標準化完成")
        sys.stdout.flush()

        # 檢查標準化後是否有 NaN 或 inf
        if np.isnan(all_features_scaled).any() or np.isinf(all_features_scaled).any():
            nan_inf_mask = np.isnan(all_features_scaled).any(axis=1) | np.isinf(all_features_scaled).any(axis=1)
            nan_inf_count = nan_inf_mask.sum()
            print(f"[ERROR] 標準化後發現 {nan_inf_count} 筆包含 NaN/Inf 的特徵！")
            sys.stdout.flush()
            # 將 NaN/Inf 替換為0
            all_features_scaled = np.nan_to_num(all_features_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            print("[INFO] 已將 NaN/Inf 替換為 0")
            sys.stdout.flush()

        # 驗證標準化結果
        feature_means = np.mean(all_features_scaled, axis=0)
        feature_stds = np.std(all_features_scaled, axis=0)
        
        print("\n" + "=" * 60)
        print("特徵標準化統計 (Feature Normalization Statistics)")
        print("=" * 60)
        print(f"{'特徵名稱':<25} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
        print("-" * 60)
        sys.stdout.flush()
        for i, name in enumerate(feature_names):
            mean_val = feature_means[i]
            std_val = feature_stds[i]
            min_val = np.min(all_features_scaled[:, i])
            max_val = np.max(all_features_scaled[:, i])
            print(f"{name:<25} {mean_val:>11.6f} {std_val:>11.6f} {min_val:>11.6f} {max_val:>11.6f}")
        
        sys.stdout.flush()
        
        # 驗證標準化是否正確（mean 應該接近 0，std 應該接近 1）
        # 只計算非常數特徵的誤差
        non_constant_mask = ~np.isin(np.arange(len(feature_vars)), constant_features) if constant_features else np.ones(len(feature_vars), dtype=bool)
        if non_constant_mask.any():
            mean_abs_error = np.abs(feature_means[non_constant_mask]).mean()
            std_error = np.abs(feature_stds[non_constant_mask] - 1.0).mean()
        else:
            mean_abs_error = 0.0
            std_error = 0.0
        
        print("-" * 60)
        # 使用科學記數法顯示小數值，確保能看到非零但很小的值
        if mean_abs_error < 1e-6:
            print(f"標準化驗證（非常數特徵）：Mean 絕對誤差 = {mean_abs_error:.2e} (應接近 0) [OK]")
        else:
            print(f"標準化驗證（非常數特徵）：Mean 絕對誤差 = {mean_abs_error:.8f} (應接近 0)")
        
        if std_error < 1e-6:
            print(f"標準化驗證（非常數特徵）：Std 絕對誤差 = {std_error:.2e} (應接近 0) [OK]")
        else:
            print(f"標準化驗證（非常數特徵）：Std 絕對誤差 = {std_error:.8f} (應接近 0)")
        
        if constant_features:
            print(f"常數特徵數量: {len(constant_features)}")
            print(f"常數特徵索引: {constant_features}")
        print("=" * 60 + "\n")
        sys.stdout.flush()

        # 創建標籤（二分類：預測 T+5 波動性突破，高波動=1，低波動=0）
        print("[INFO] 開始創建標籤（T+5 波動性突破）...")
        sys.stdout.flush()
        close_prices = df['Close'].values
        targets = []
        
        # 波動性突破邏輯：
        # 1. 計算未來 5 天報酬率
        # 2. 如果絕對值 > 閾值（3%），標籤為 1（高波動/突破）
        # 3. 否則標籤為 0（低波動/停滯）
        volatility_threshold = 0.03  # 3% 的 5 天移動，可調整
        
        for i in range(len(df) - 5):  # 最後5行無法生成標籤
            current_price = close_prices[i]
            future_price = close_prices[i + 5]  # T+5
            
            # 計算未來 5 天報酬率
            future_return_5d = (future_price - current_price) / current_price
            
            # 波動性突破：如果絕對報酬率 > 閾值，標籤為 1
            target = 1 if abs(future_return_5d) > volatility_threshold else 0
            targets.append(target)
        
        targets = np.array(targets, dtype=np.float32).reshape(-1, 1)
        print(f"[INFO] 標籤創建完成，共 {len(targets)} 個標籤")
        print(f"[INFO] 波動性閾值: {volatility_threshold*100:.1f}% (5天)")
        print(f"[INFO] 標籤分布: 高波動={np.sum(targets == 1)}, 低波動={np.sum(targets == 0)}")
        sys.stdout.flush()
        
        # 注意：targets 的長度是 len(df) - 5
        # targets[i] 對應於時間點 i 的標籤（比較 Close[i] 和 Close[i+5]）

        # 生成序列（滑動窗口）
        print("[INFO] 開始生成序列...")
        sys.stdout.flush()
        sequences = []
        labels = []

        # 序列生成邏輯：
        # - 對於時間點 i，特徵是 all_features_scaled[i:i+sequence_length]
        # - 標籤應該是對應於時間點 i+sequence_length-1 的標籤
        # - 但標籤需要 T+5，所以我們需要確保 i+sequence_length-1+5 < len(df)
        # - 即 i < len(df) - sequence_length - 4
        
        max_seq_idx = len(df) - self.sequence_length - 4  # -4 因為標籤需要 T+5
        print(f"[INFO] 最大序列索引: {max_seq_idx}, 目標數量: {len(targets)}")
        sys.stdout.flush()
        
        for i in range(max_seq_idx + 1):
            # 確保不會超出範圍
            if i + self.sequence_length > len(all_features_scaled):
                break
                
            # 合併特徵序列
            seq = all_features_scaled[i:i + self.sequence_length]
            
            # 檢查序列中是否有 NaN
            if np.isnan(seq).any():
                print(f"[WARNING] 序列 {i} 包含 NaN，跳過")
                sys.stdout.flush()
                continue
                
            sequences.append(seq)

            # 標籤對應於時間點 i+sequence_length-1
            label_idx = i + self.sequence_length - 1
            if label_idx < len(targets):
                labels.append(targets[label_idx])
            else:
                # 如果超出範圍，跳過這個序列
                sequences.pop()
                continue
            
            # 每生成1000個序列顯示進度
            if (i + 1) % 1000 == 0:
                print(f"[INFO] 已生成 {len(sequences)} 個序列...")
                sys.stdout.flush()
        
        print(f"[INFO] 序列生成完成，共 {len(sequences)} 個序列")
        sys.stdout.flush()

        # 轉換為 numpy 陣列
        X = np.array(sequences, dtype=np.float32)
        y = np.array(labels, dtype=np.float32)

        # 最終檢查：確保沒有 NaN
        if np.isnan(X).any() or np.isnan(y).any():
            raise ValueError("錯誤：特徵或標籤中包含 NaN 值！")

        print(
            f"[INFO] 資料準備完成（特徵工程方法，預測 T+5 波動性突破）：\n"
            f"  - 序列數量: {len(X)}\n"
            f"  - 序列長度: {self.sequence_length}\n"
            f"  - 總特徵維度: {X.shape[2]} (數值特徵: 4, 易經特徵: 5)\n"
            f"  - 標籤分布: 高波動={np.sum(y == 1)}, 低波動={np.sum(y == 0)}\n"
            f"  - 預測目標: 波動性突破（5天內絕對報酬率 > 3%）"
        )
        sys.stdout.flush()

        return X, y

    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_split: float = 0.8
    ) -> Tuple[DataLoader, DataLoader]:
        """分割資料為訓練集和測試集.

        使用時間序列分割（不進行隨機打亂），前 80% 用於訓練，後 20% 用於測試。

        Args:
            X: 合併特徵陣列，形狀為 (N, sequence_length, num_features)
            y: 標籤陣列，形狀為 (N, 1)
            train_split: 訓練集比例，預設為 0.8

        Returns:
            (train_loader, test_loader) 二元組
        """
        # 時間序列分割（不進行隨機打亂）
        split_idx = int(len(X) * train_split)

        X_train = X[:split_idx]
        y_train = y[:split_idx]

        X_test = X[split_idx:]
        y_test = y[split_idx:]

        # 轉換為 PyTorch 張量
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

        # 創建 Dataset
        train_dataset = QuantumDataset(X_train_tensor, y_train_tensor)
        test_dataset = QuantumDataset(X_test_tensor, y_test_tensor)

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
