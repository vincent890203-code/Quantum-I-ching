# 本地環境安裝腳本（PowerShell）

Write-Host "正在檢查 Python 環境..." -ForegroundColor Cyan
python --version

Write-Host "`n正在升級 pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip setuptools wheel

Write-Host "`n正在安裝專案依賴套件..." -ForegroundColor Green
Write-Host "這可能需要幾分鐘時間，請耐心等待..." -ForegroundColor Yellow

# 先安裝 PyTorch（如果還沒安裝）
Write-Host "`n[步驟 1/3] 安裝 PyTorch..." -ForegroundColor Cyan
pip install torch>=2.0.0

# 安裝 transformers（修復 LRScheduler 問題）
Write-Host "`n[步驟 2/3] 安裝 transformers（修復 LRScheduler 兼容性問題）..." -ForegroundColor Cyan
pip install transformers>=4.30.0

# 安裝其他依賴
Write-Host "`n[步驟 3/3] 安裝其他依賴..." -ForegroundColor Cyan
pip install -r requirements.txt

Write-Host "`n安裝完成！正在驗證關鍵套件..." -ForegroundColor Green

$errors = @()

python -c "import sentence_transformers; print('✓ sentence-transformers 已安裝')" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ sentence-transformers 安裝失敗" -ForegroundColor Red
    $errors += "sentence-transformers"
} else {
    Write-Host "✓ sentence-transformers 安裝成功" -ForegroundColor Green
}

python -c "import torch; print(f'✓ torch {torch.__version__} 已安裝')" 2>$null
python -c "import transformers; print(f'✓ transformers {transformers.__version__} 已安裝')" 2>$null
python -c "import chromadb; print('✓ chromadb 已安裝')" 2>$null
python -c "import streamlit; print('✓ streamlit 已安裝')" 2>$null
python -c "import google.generativeai; print('✓ google-generativeai 已安裝')" 2>$null

if ($errors.Count -eq 0) {
    Write-Host "`n所有依賴安裝完成！" -ForegroundColor Green
    Write-Host "現在可以執行: streamlit run dashboard.py" -ForegroundColor Cyan
} else {
    Write-Host "`n部分套件安裝失敗，請手動安裝：" -ForegroundColor Yellow
    foreach ($err in $errors) {
        Write-Host "  pip install $err" -ForegroundColor Yellow
    }
}
