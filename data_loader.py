"""Quantum I-Ching 專案資料載入模組.

此模組負責從 Yahoo Finance 獲取歷史股票價格資料。
"""

import logging
from typing import Optional

import pandas as pd
import yfinance as yf

from config import settings


class MarketDataLoader:
    """市場資料載入器類別.

    使用 yfinance 從 Yahoo Finance 獲取歷史股票價格資料。
    支援批量下載多檔股票，並具備錯誤處理機制。
    
    Attributes:
        logger: logging.Logger 實例，用於記錄操作日誌
    """

    def __init__(self) -> None:
        """初始化 MarketDataLoader 實例."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        # 如果 logger 還沒有 handler，添加一個控制台 handler
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def _format_ticker(self, ticker: str, market_type: Optional[str] = None) -> str:
        """根據市場類型格式化 ticker.

        Args:
            ticker: 原始股票代號字串。
            market_type: 市場類型（'US', 'TW', 'CRYPTO'），若為 None 則使用 settings.MARKET_TYPE。

        Returns:
            已根據市場類型調整後的 ticker 字串。
        """
        if market_type is None:
            market_type = settings.MARKET_TYPE

        if market_type == "TW":
            # 台股：確保有 .TW 後綴
            if not ticker.endswith(".TW"):
                return f"{ticker}.TW"
        elif market_type == "CRYPTO":
            # 加密貨幣：確保有 -USD 後綴
            if not ticker.endswith("-USD"):
                return f"{ticker}-USD"

        # US 或其他情況：直接回傳原始 ticker
        return ticker

    def fetch_data(
        self,
        tickers: Optional[list[str]] = None,
        market_type: Optional[str] = None
    ) -> pd.DataFrame:
        """獲取歷史股票價格資料.

        從 Yahoo Finance 下載指定股票代號的歷史價格資料。
        資料包含 Open, High, Low, Close, Volume 等欄位。

        Args:
            tickers: 股票代號清單。如果為 None，則使用 config.settings.TARGET_TICKERS。
            market_type: 市場類型（'US', 'TW', 'CRYPTO'），若為 None 則使用 settings.MARKET_TYPE。

        Returns:
            包含歷史價格資料的 pandas DataFrame。
            - 當下載多檔股票時，返回 MultiIndex DataFrame（columns: [Ticker, OHLCV]）
            - 當下載單檔股票時，返回標準 DataFrame（columns: OHLCV）
            - 如果所有股票下載失敗，返回空的 DataFrame。

        Raises:
            不會拋出異常，所有錯誤都會被記錄到日誌中。
        """
        if tickers is None:
            tickers = list(settings.TARGET_TICKERS)

        if not tickers:
            self.logger.warning("股票代號清單為空，返回空 DataFrame")
            return pd.DataFrame()

        # 根據市場類型調整 ticker 格式
        if market_type is None:
            market_type = settings.MARKET_TYPE

        processed_tickers = []
        for ticker in tickers:
            processed_ticker = self._format_ticker(ticker, market_type=market_type)
            processed_tickers.append(processed_ticker)
            if processed_ticker != ticker:
                self.logger.debug(
                    f"Ticker 格式調整: {ticker} -> {processed_ticker}"
                )

        self.logger.info(
            f"開始下載股票資料: {processed_tickers}, "
            f"日期範圍: {settings.START_DATE} 至 {settings.END_DATE}, "
            f"市場類型: {market_type}"
        )

        try:
            # 如果只有一個 ticker，嘗試直接使用 Ticker 物件（更可靠）
            if len(processed_tickers) == 1:
                ticker = processed_tickers[0]
                try:
                    ticker_obj = yf.Ticker(ticker)
                    data = ticker_obj.history(
                        start=settings.START_DATE,
                        end=settings.END_DATE
                    )
                    
                    if not data.empty:
                        self.logger.info(
                            f"成功下載 {ticker} 的資料（使用 Ticker 物件），共 {len(data)} 筆記錄"
                        )
                        return data
                    else:
                        self.logger.warning(
                            f"使用 Ticker 物件下載 {ticker} 的資料為空，嘗試使用 download 方法"
                        )
                except Exception as e:
                    self.logger.debug(
                        f"使用 Ticker 物件下載失敗: {e}，嘗試使用 download 方法"
                    )
            # 使用 yf.download 批量下載資料
            # 預設行為會返回 MultiIndex DataFrame（多檔股票時）
            # columns 結構：[Ticker, OHLCV]，方便存取 data['AAPL']['Close']
            data = yf.download(
                tickers=processed_tickers,
                start=settings.START_DATE,
                end=settings.END_DATE,
                progress=False,  # 關閉進度條輸出，保持日誌整潔
                # 注意：不同版本的 yfinance 可能參數不同
                # show_errors 參數在某些版本中不存在，已移除
            )

            # 檢查資料是否為空
            if data.empty:
                self.logger.warning(
                    f"下載的資料為空。可能原因：日期範圍內無資料、股票代號無效等。"
                )
                return pd.DataFrame()

            # 處理單一股票的情況
            # 注意：即使只有一個 ticker，yfinance 也可能返回 MultiIndex DataFrame
            if len(processed_tickers) == 1:
                ticker = processed_tickers[0]
                # 如果是 MultiIndex，提取該 ticker 的資料
                if isinstance(data.columns, pd.MultiIndex):
                    # 取得第一層的所有 ticker
                    level0_values = data.columns.get_level_values(0).unique()
                    self.logger.debug(
                        f"MultiIndex 結構，第一層 ticker: {list(level0_values)}"
                    )
                    
                    # 檢查 ticker 是否存在於第一層
                    if ticker in level0_values:
                        ticker_data = data[ticker].copy()
                        self.logger.info(
                            f"成功下載 {ticker} 的資料，共 {len(ticker_data)} 筆記錄（MultiIndex 結構）"
                        )
                        return ticker_data
                    else:
                        # 如果 ticker 不匹配，但只有一個 ticker，使用它
                        if len(level0_values) == 1:
                            actual_ticker = level0_values[0]
                            ticker_data = data[actual_ticker].copy()
                            self.logger.info(
                                f"成功下載 {actual_ticker} 的資料（請求 {ticker}），共 {len(ticker_data)} 筆記錄"
                            )
                            return ticker_data
                        else:
                            # 多個 ticker 但都不匹配，記錄警告並使用第一個
                            self.logger.warning(
                                f"Ticker {ticker} 不存在於下載的資料中。"
                                f"實際 ticker: {list(level0_values)}。"
                                f"使用第一個 ticker: {level0_values[0]}"
                            )
                            actual_ticker = level0_values[0]
                            ticker_data = data[actual_ticker].copy()
                            return ticker_data
                else:
                    # 標準 DataFrame，直接返回
                    self.logger.info(f"成功下載 {ticker} 的資料，共 {len(data)} 筆記錄")
                    return data
            else:
                # 多檔股票時，確保資料結構正確（MultiIndex columns）
                if isinstance(data.columns, pd.MultiIndex):
                    # 標準格式：[Ticker, OHLCV]
                    self.logger.info(
                        f"成功下載 {len(processed_tickers)} 檔股票的資料，共 {len(data)} 筆記錄"
                    )
                    return data
                else:
                    # 這種情況很少見，但需要處理
                    self.logger.warning(
                        "多檔股票下載但資料結構異常，返回原始資料"
                    )
                    return data

        except Exception as e:
            self.logger.error(
                f"下載股票資料時發生錯誤: {str(e)}", exc_info=True
            )
            return pd.DataFrame()