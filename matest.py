# 您可以執行一個小腳本來檢查
import timm

# 搜尋 Mamba 相關模型
print("--- Searching for 'mamba' ---")
mamba_models = timm.list_models('*mamba*')
print(mamba_models)

# 搜尋 Vim 相關模型 (Vision Mamba)
print("\n--- Searching for 'vim' ---")
vim_models = timm.list_models('*vim*')
print(vim_models)
