# save_load.py
import json
import os
from parameters import FurnaceParameters

def save_parameters(params, filename=None):
    """保存参数到文件"""
    if filename is None:
        filename = params.case_name
    
    # 确保cases目录存在
    os.makedirs('cases', exist_ok=True)
    
    filepath = f"cases/{filename}.json"
    
    # 转换为可序列化的字典
    data = {}
    for key, value in params.__dict__.items():
        data[key] = value
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"参数已保存: {filepath}")
    return filepath

def load_parameters(filename):
    """从文件加载参数"""
    filepath = f"cases/{filename}.json"
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"参数文件不存在: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    params = FurnaceParameters(data['case_name'])
    for key, value in data.items():
        if hasattr(params, key):
            setattr(params, key, value)
    
    print(f"参数已加载: {filepath}")
    return params

def list_saved_cases():
    """列出所有保存的算例"""
    if not os.path.exists('cases'):
        return []
    
    cases = []
    for file in os.listdir('cases'):
        if file.endswith('.json'):
            cases.append(file[:-5])  # 去掉.json后缀
    
    return sorted(cases)