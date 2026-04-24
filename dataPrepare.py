import pandas as pd
import os
import shutil
import json
from config import (dataset_config, item_file, random_file, train_file, test_file,
                    CURRENT_DATASET, DATASET_DIR, descriptions_file)  # ✅ 导入 DATASET_DIR

def load_image_descriptions():
    """加载图片描述JSON文件，返回 {item_id: [description1, description2, ...]} 的字典"""
    if descriptions_file is None or not os.path.exists(descriptions_file):
        print("⚠️ 未找到图片描述文件，将跳过多模态初始化")
        return {}
    
    print(f"加载图片描述: {descriptions_file}")
    
    with open(descriptions_file, 'r', encoding='utf-8') as f:
        descriptions_data = json.load(f)
    
    # 构建 item_id -> [descriptions] 的映射
    item_descriptions = {}
    for entry in descriptions_data:
        image_path = entry['image_path']
        description = entry['description']
        
        # 从路径中提取 item_id
        filename = os.path.basename(image_path)
        item_id = filename.split('_')[0]
        
        if item_id not in item_descriptions:
            item_descriptions[item_id] = []
        item_descriptions[item_id].append(description)
    
    print(f"✅ 成功加载 {len(item_descriptions)} 个物品的图片描述")
    return item_descriptions

def get_required_item_ids():
    """从random文件中获取物品ID（去重）"""
    print(f"从{CURRENT_DATASET}.random中提取物品ID...")
    
    random_df = pd.read_csv(random_file, sep='\t', names=['user_id', 'candidates'], header=None)
    required_items = set()
    
    for _, row in random_df.iterrows():
        candidates = str(row['candidates']).split()
        required_items.update(candidates)
    
    print(f"{CURRENT_DATASET}.random中共有 {len(required_items)} 个唯一物品")
    return required_items

def load_item_titles(required_items):
    """只加载需要的物品的ID和标题"""
    print("加载物品标题...")
    
    item_titles = {}
    
    with open(item_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line_num % 50000 == 0:
                print(f"  已处理 {line_num} 行...")
            
            fields = line.strip().split('\t')
            if len(fields) >= 2:
                item_id = fields[0]
                if item_id in required_items:
                    title = fields[1]
                    item_titles[item_id] = title
    
    print(f"✅ 成功加载 {len(item_titles)} 个物品的标题")
    
    # 检查缺失
    missing_items = required_items - set(item_titles.keys())
    if missing_items:
        print(f"❌ 缺失 {len(missing_items)} 个物品，请检查数据")
        print(f"缺失物品示例: {list(missing_items)[:5]}")
        return None
    
    return item_titles

def createRandomDF(file_path):
    """读取随机负样本文件"""
    df = pd.read_csv(file_path, sep='\t', names=['user_id', 'candidates'], header=None)
    df['candidates'] = df['candidates'].apply(lambda x: str(x).split())
    return df

def createInterDF(file_path):
    """读取交互数据文件"""
    return pd.read_csv(file_path, sep='\t')

def prepare_initial_memory():
    """生成两种模式的初始记忆文件（basic 和 description）"""
    print("=" * 60)
    print(f"开始为 {CURRENT_DATASET} 数据集生成初始记忆...")
    print("=" * 60)
    
    # 1. 加载图片描述（用于 description 模式）
    item_descriptions = load_image_descriptions()
    
    # 2. 获取物品ID和标题
    required_items = get_required_item_ids()
    item_titles = load_item_titles(required_items)
    if item_titles is None:
        return
    
    # 3. 读取交互数据（用于获取用户ID）
    train_df = createInterDF(train_file)
    test_df = createInterDF(test_file)
    all_user_ids = train_df['user_id:token'].unique()
    
    # ✅ 4. 生成两种模式的初始记忆
    modes = {
        "basic": {
            "dir": f"dataset/initial_basic/{DATASET_DIR}",
            "use_descriptions": False,
            "name": "基础模式（仅标题）"
        },
        "description": {
            "dir": f"dataset/initial_description/{DATASET_DIR}",
            "use_descriptions": True,
            "name": "多模态模式（标题+图片描述）"
        }
    }
    
    for mode_key, mode_config in modes.items():
        print("\n" + "-" * 60)
        print(f"生成 {mode_config['name']}")
        print("-" * 60)
        
        initial_dir = mode_config['dir']
        use_descriptions = mode_config['use_descriptions']
        
        # 创建目录
        item_dir = f"{initial_dir}/item"
        user_dir = f"{initial_dir}/user"
        user_long_dir = f"{initial_dir}/user-long"
        
        os.makedirs(item_dir, exist_ok=True)
        os.makedirs(user_dir, exist_ok=True)
        os.makedirs(user_long_dir, exist_ok=True)
        
        # 生成物品初始记忆
        item_template = dataset_config["item_init_template"]
        items_with_image = 0
        items_without_image = 0
        
        for item_id in required_items:
            item_id = str(item_id).strip()
            title = item_titles[item_id]
            
            # 基础描述
            init_item_memory = item_template.format(title=title)
            
            # 添加图片描述（仅在 description 模式）
            if use_descriptions and item_id in item_descriptions:
                image_descs = item_descriptions[item_id]
                combined_image_desc = " ".join(image_descs)
                init_item_memory += f" Image description: {combined_image_desc}"
                items_with_image += 1
            else:
                items_without_image += 1
            
            with open(f"{item_dir}/item.{item_id}", "w", encoding="utf-8") as f:
                f.write(init_item_memory)
        
        # 生成用户初始记忆
        user_template = dataset_config["user_init_template"]
        
        for user_id in all_user_ids:
            user_id = str(user_id).strip()
            init_user_memory = user_template
            
            with open(f"{user_dir}/user.{user_id}", "w", encoding="utf-8") as f:
                f.write(init_user_memory)
            
            with open(f"{user_long_dir}/user.{user_id}", "w", encoding="utf-8") as f:
                f.write("")
        
        # 输出统计
        print(f"✅ {mode_config['name']} 初始记忆生成完成：")
        print(f"  - 物品数: {len(required_items)}")
        if use_descriptions:
            print(f"    · 包含图片描述: {items_with_image}")
            print(f"    · 仅标题: {items_without_image}")
        print(f"  - 用户数: {len(all_user_ids)}")
        print(f"  - 保存路径: {initial_dir}")
    
    print("\n" + "=" * 60)
    print(f"✅ {CURRENT_DATASET} 数据集的两种模式初始记忆已全部生成！")
    print("=" * 60)

if __name__ == "__main__":
    prepare_initial_memory()

def createItemDF(file_path):
    """创建物品DataFrame（兼容性函数）"""
    required_items = get_required_item_ids()
    item_titles = load_item_titles(required_items)
    
    if item_titles is None:
        return None
    
    # 转换为DataFrame格式
    data = []
    for item_id, title in item_titles.items():
        data.append({
            'item_id:token': item_id,
            'title:token_seq': title,
            'category:token_seq': '',
            'image_urls:token_seq': '',
            'description:token_seq': ''
        })
    
    return pd.DataFrame(data)