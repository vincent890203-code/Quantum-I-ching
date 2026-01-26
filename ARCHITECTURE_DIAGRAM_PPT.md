# Quantum I-Ching 系統架構圖（PowerPoint 版本 - 16:9 優化）

## 完整系統架構（簡報優化版 - 人性化描述）

```mermaid
flowchart LR
    subgraph DataLayer["📊 資料獲取層"]
        Yahoo[Yahoo Finance<br/>金融資料來源]
        Loader[市場資料載入器<br/>(MarketDataLoader)]
        Config[系統設定與<br/>六十四卦對照表<br/>(Settings & HEXAGRAM_MAP)]
        
        Yahoo --> Loader
        Config --> Loader
    end

    subgraph EncodingLayer["🔢 卦象編碼層"]
        Encoder[市場編碼器<br/>(MarketEncoder)]
        Core[易經核心運算<br/>(IChingCore)]
        HexMap[六十四卦對照表<br/>(HEXAGRAM_MAP)]
        
        Encoder --> Core
        Core --> HexMap
    end

    subgraph KnowledgeLayer["📚 知識檢索層"]
        Setup[易經資料庫設置<br/>(setup_iching_db.py)]
        JSON[易經完整資料庫<br/>(iching_complete.json)]
        KLoader[知識載入器<br/>(IChingKnowledgeLoader)]
        VectorDB[向量資料庫<br/>(IChingVectorStore)]
        
        Setup --> JSON
        JSON --> KLoader
        KLoader --> VectorDB
    end

    subgraph AppLayer["📱 應用層"]
        Dashboard[網頁儀表板<br/>(Streamlit Dashboard)]
        Oracle[AI 解讀模組<br/>(Oracle 類別)]
        CLI[命令列工具<br/>(CLI 工具)]
        Gemini[Google Gemini<br/>大型語言模型]
        
        Dashboard --> Oracle
        CLI --> Encoder
        Oracle --> Gemini
    end

    subgraph MLLayer["🤖 機器學習層"]
        Processor[資料處理器<br/>(DataProcessor)]
        LSTM[量子長短期記憶模型<br/>(QuantumLSTM)]
        Backtest[策略回測器<br/>(QuantumBacktester)]
        
        Processor --> LSTM
        LSTM --> Backtest
    end

    Loader --> Encoder
    Encoder --> Core
    Core --> Dashboard
    Core --> Oracle
    Core --> Processor
    Oracle --> VectorDB
    Processor --> LSTM

    classDef dataLayer fill:#e8f5e9,stroke:#2e7d32,stroke-width:5px,color:#000
    classDef encodingLayer fill:#fff3e0,stroke:#f57c00,stroke-width:5px,color:#000
    classDef knowledgeLayer fill:#f3e5f5,stroke:#7b1fa2,stroke-width:5px,color:#000
    classDef appLayer fill:#e1f5fe,stroke:#0277bd,stroke-width:5px,color:#000
    classDef mlLayer fill:#fce4ec,stroke:#c2185b,stroke-width:5px,color:#000
    classDef external fill:#fff9c4,stroke:#f9a825,stroke-width:5px,color:#000

    class Yahoo,Loader,Config dataLayer
    class Encoder,Core,HexMap encodingLayer
    class Setup,JSON,KLoader,VectorDB knowledgeLayer
    class Dashboard,Oracle,CLI appLayer
    class Processor,LSTM,Backtest mlLayer
    class Gemini external
```

---

## 詳細資料流程圖（簡報優化版 - 人性化描述）

```mermaid
flowchart LR
    Start([👤 使用者輸入<br/>股票代號或問題]) --> P1
    
    subgraph P1["📊 資料獲取"]
        DL[市場資料載入器<br/>(MarketDataLoader)]
        Format[格式化股票代號<br/>(格式化 ticker)]
        Raw[原始市場資料<br/>(raw_df OHLCV)]
        
        DL --> Format
        Format --> Raw
    end

    subgraph P2["🔢 卦象編碼"]
        ME[市場編碼器<br/>(MarketEncoder)]
        Tech[計算技術指標<br/>(RVOL 相對成交量)]
        Map[大衍之數映射<br/>(6/7/8/9 四象)]
        Roll[滾動窗口處理<br/>(6天)]
        IC1[易經核心運算<br/>(IChingCore)<br/>計算本卦與之卦]
        Encoded[編碼後的卦象資料<br/>(encoded_df)]
        
        ME --> Tech
        Tech --> Map
        Map --> Roll
        Roll --> IC1
        IC1 --> Encoded
    end

    subgraph P3["🔮 卦象解讀"]
        IC2[易經核心處理<br/>(IChingCore)]
        Hex[取得卦象資訊<br/>(interpret_sequence)]
        State[組成市場狀態<br/>(market_state)]
        Viz[視覺化處理<br/>(視覺化)]
        
        IC2 --> Hex
        Hex --> State
        State --> Viz
    end

    subgraph P4["🔮 AI 解讀"]
        Oracle[Oracle 解讀模組<br/>(Oracle 類別)]
        Strategy[解析之卦策略<br/>(解析策略)]
        JSON[易經知識庫<br/>(iching_complete.json)]
        Extract[抽取易經經文<br/>(抽取經文)]
        Prompt[建立系統提示<br/>(建立提示)]
        Gemini[Google Gemini API<br/>(Gemini API)]
        Response[結構化回答<br/>(結構化回答)]
        
        Oracle --> Strategy
        Strategy --> Extract
        Extract --> JSON
        JSON --> Prompt
        Prompt --> Gemini
        Gemini --> Response
    end

    subgraph P5["📱 結果顯示"]
        Result[顯示最終結果<br/>K線圖 + 卦象 + AI解讀]
    end

    P1 --> P2
    P2 --> P3
    P3 --> P4
    P4 --> P5

    classDef phase1 fill:#e8f5e9,stroke:#2e7d32,stroke-width:5px,color:#000
    classDef phase2 fill:#fff3e0,stroke:#f57c00,stroke-width:5px,color:#000
    classDef phase3 fill:#f3e5f5,stroke:#7b1fa2,stroke-width:5px,color:#000
    classDef phase4 fill:#fce4ec,stroke:#c2185b,stroke-width:5px,color:#000
    classDef phase5 fill:#e1f5fe,stroke:#0277bd,stroke-width:5px,color:#000
    classDef external fill:#fff9c4,stroke:#f9a825,stroke-width:5px,color:#000
    classDef start fill:#e3f2fd,stroke:#1976d2,stroke-width:5px,color:#000

    class DL,Format,Raw phase1
    class ME,Tech,Map,Roll,IC1,Encoded phase2
    class IC2,Hex,State,Viz phase3
    class Oracle,Strategy,Extract,Prompt,Response phase4
    class Result phase5
    class JSON,Gemini external
    class Start start
```

---

## 大衍之數映射流程（簡報優化版 - 人性化描述）

```mermaid
flowchart LR
    Start([市場歷史資料<br/>開高低收成交量<br/>(OHLCV)]) --> Calc[計算技術指標<br/>(技術指標計算)]
    Calc --> RVOL[相對成交量<br/>(RVOL)<br/>成交量 / 20日均量]
    RVOL --> Percentile[百分位數計算<br/>(RVOL_Percentile)<br/>滾動窗口百分位數]
    
    Percentile --> Map{大衍之數映射<br/>根據百分位數<br/>轉換為四象}
    
    Map -->|0-6.25%| Y6[6 老陰<br/>極低能量狀態]
    Map -->|6.25-50%| Y8[8 少陰<br/>低至中能量狀態]
    Map -->|50-81.25%| Y7[7 少陽<br/>中至高能量狀態]
    Map -->|81.25-100%| Y9[9 老陽<br/>極高能量狀態]
    
    Y6 --> Window[滾動窗口整合<br/>(滾動窗口 6天)<br/>整合過去6天的四象]
    Y8 --> Window
    Y7 --> Window
    Y9 --> Window
    
    Window --> Sequence[儀式序列<br/>(Ritual_Sequence)<br/>例如: 987896]
    Sequence --> Binary[二進制卦象<br/>(Hexagram_Binary)<br/>例如: 101010]
    Binary --> Hexagram[查詢卦象對照表<br/>(HEXAGRAM_MAP)<br/>取得對應的卦象]
    
    style Start fill:#e3f2fd,stroke:#1976d2,stroke-width:5px
    style Calc fill:#e8f5e9,stroke:#2e7d32,stroke-width:5px
    style RVOL fill:#e8f5e9,stroke:#2e7d32,stroke-width:5px
    style Percentile fill:#e8f5e9,stroke:#2e7d32,stroke-width:5px
    style Map fill:#fff3e0,stroke:#f57c00,stroke-width:5px
    style Y6 fill:#ffebee,stroke:#c62828,stroke-width:5px
    style Y8 fill:#fff3e0,stroke:#ef6c00,stroke-width:5px
    style Y7 fill:#e8f5e9,stroke:#2e7d32,stroke-width:5px
    style Y9 fill:#e1f5fe,stroke:#0277bd,stroke-width:5px
    style Window fill:#f3e5f5,stroke:#7b1fa2,stroke-width:5px
    style Sequence fill:#f3e5f5,stroke:#7b1fa2,stroke-width:5px
    style Binary fill:#f3e5f5,stroke:#7b1fa2,stroke-width:5px
    style Hexagram fill:#e1f5fe,stroke:#0277bd,stroke-width:5px
```
