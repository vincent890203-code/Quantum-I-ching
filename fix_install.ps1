# 快速修復腳本（PowerShell）

Write-Host "正在清除 pip 快取..." -ForegroundColor Yellow
pip cache purge

Write-Host "`n正在安裝 PyTorch（使用最新可用版本）..." -ForegroundColor Cyan
pip install torch --upgrade

Write-Host "`n正在安裝 transformers..." -ForegroundColor Cyan
pip install transformers>=4.30.0 --upgrade

Write-Host "`n正在安裝 sentence-transformers..." -ForegroundColor Cyan
pip install sentence-transformers>=2.3.0 --upgrade

Write-Host "`n正在安裝其他依賴..." -ForegroundColor Cyan
pip install -r requirements.txt

Write-Host "`n驗證安裝..." -ForegroundColor Green
python -c "import torch; print(f'✓ torch {torch.__version__}')"
python -c "import transformers; print(f'✓ transformers {transformers.__version__}')"
python -c "import sentence_transformers; print('✓ sentence-transformers')"

Write-Host "`n完成！" -ForegroundColor Green
