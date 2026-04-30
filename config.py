"""
统一配置文件 - Multi-modal AgentCF
✅ 所有路径在此集中配置，其他文件零硬编码
✅ 支持多数据集：CDs, All_Beauty, Cell_Phones_and_Accessories, Fashion
"""

# ============= 数据集选择（主要修改这里）=============
CURRENT_DATASET = "CDs"
# 可选: "CDs" | "All_Beauty" | "Cell_Phones" | "Fashion"

# ============= 模式选择 =============
TRAIN_MODE = "description"   # 训练模式: "basic" | "description"
EVAL_MODE = "rrf"          # 评估模式: "basic" | "description" | "embedding" | "rrf"

# ============= 负样本配置 =============
USE_FIXED_NEGATIVES = True   # 是否使用预生成的固定负样本
NEGATIVE_SAMPLE_SEED = 42    # 负样本seed

# ============= 数据集元信息配置 =============
DATASETS_META = {
    "CDs": {
        "name": "CDs_and_Vinyl",
        "prefix": "CDs",
        "data_dir": "dataset/CDs/",
        "dataset_dir": "CDs",
        "user_init_template": "I enjoy listening to CDs and vinyl records very much.",
        "item_init_template": "This is a CD or vinyl record titled '{title}'.",
        "image_dir": "/root/cds_images_CDs",
    },
    "All_Beauty": {
        "name": "All_Beauty",
        "prefix": "All_Beauty",
        "data_dir": "dataset/All_Beauty/",
        "dataset_dir": "All_Beauty",
        "user_init_template": "I am interested in beauty and personal care products.",
        "item_init_template": "This is a beauty product titled '{title}'.",
        "image_dir": "/root/cds_images_beauty",
    },
    "Cell_Phones": {
        "name": "Cell_Phones_and_Accessories",
        "prefix": "Cell_Phones_and_Accessories",
        "data_dir": "dataset/Cell_Phones_and_Accessories/",
        "dataset_dir": "Cell_Phones_and_Accessories",
        "user_init_template": "I am interested in cell phones and related accessories.",
        "item_init_template": "This is a cell phone or accessory titled '{title}'.",
        "image_dir": "/root/cds_images_cell",
    },
    # ✅ 新增 Fashion 数据集配置
    "Fashion": {
        "name": "Fashion",
        "prefix": "Fashion",
        "data_dir": "dataset/Fashion/",
        "dataset_dir": "Fashion",
        "user_init_template": "I am interested in fashion and clothing products.",
        "item_init_template": "This is a fashion item titled '{title}'.",
        "image_dir": "/root/cds_images_fashion",  # 如果有图片，请修改为实际路径
    }
}

# ============= 获取当前数据集配置 =============
current_dataset_meta = DATASETS_META[CURRENT_DATASET]
DATASET_PREFIX = current_dataset_meta["prefix"]
DATA_BASE_DIR = current_dataset_meta["data_dir"]
DATASET_DIR = current_dataset_meta["dataset_dir"]  # ✅ 新增：获取输出目录名

# 具体数据文件（输入文件，使用 DATA_BASE_DIR + DATASET_PREFIX）
item_file = f"{DATA_BASE_DIR}{DATASET_PREFIX}.item"
random_file = f"{DATA_BASE_DIR}{DATASET_PREFIX}.random"
train_file = f"{DATA_BASE_DIR}{DATASET_PREFIX}.train.inter"
test_file = f"{DATA_BASE_DIR}{DATASET_PREFIX}.test.inter"
descriptions_file = f"{DATA_BASE_DIR}descriptions.json" # dataset/Fashion/descriptions.json

# ✅ 固定负样本文件路径（输出文件，使用 DATASET_DIR）
TRAIN_NEGATIVES_FILE = f"dataset/{DATASET_DIR}/train_negatives_seed{NEGATIVE_SAMPLE_SEED}.json"
EVAL_CANDIDATES_FILE = f"dataset/{DATASET_DIR}/eval_candidates_seed{NEGATIVE_SAMPLE_SEED}.json"

# ============= 训练模式配置 =============
TRAIN_CONFIGS = {
    "basic": {
        "initial_dir": f"dataset/initial_basic/{DATASET_DIR}",  # ✅ 修改：使用 DATASET_DIR
        "memory_name": f"AgentCF_{DATASET_DIR}_basic",  # ✅ 修改：使用 DATASET_DIR
        "use_descriptions": False,
        "descriptions_file": None
    },
    "description": {
        "initial_dir": f"dataset/initial_description/{DATASET_DIR}",  # ✅ 修改：使用 DATASET_DIR
        "memory_name": f"AgentCF_{DATASET_DIR}_description",  # ✅ 修改：使用 DATASET_DIR
        "use_descriptions": True,
        "descriptions_file": descriptions_file
    }
}

# ============= 评估模式配置 =============
EVAL_CONFIGS = {
    "basic": {
        "method_name": f"AgentCF_{DATASET_DIR}_basic",  # ✅ 修改：使用 DATASET_DIR
        "memory_dir": f"memory/AgentCF_{DATASET_DIR}_basic",  # ✅ 修改：使用 DATASET_DIR
        "use_embedding": False,
        "use_llm": True
    },
    "description": {
        "method_name": f"AgentCF_{DATASET_DIR}_description",  # ✅ 修改：使用 DATASET_DIR
        "memory_dir": f"memory/AgentCF_{DATASET_DIR}_description",  # ✅ 修改：使用 DATASET_DIR
        "use_embedding": False,
        "use_llm": True
    },
    "embedding": {
        "method_name": f"AgentCF_{DATASET_DIR}_embedding",  # ✅ 修改：使用 DATASET_DIR
        "embedding_dir": f"dataset/embeddings/{DATASET_DIR}",  # ✅ 修改：使用 DATASET_DIR
        "use_embedding": True,
        "use_llm": False
    },
    "rrf": {
        "method_name": f"AgentCF_{DATASET_DIR}_rrf",  # ✅ 修改：使用 DATASET_DIR
        "memory_dir": f"memory/AgentCF_{DATASET_DIR}_description",  # ✅ 修改：使用 DATASET_DIR
        "embedding_dir": f"dataset/embeddings/{DATASET_DIR}",  # ✅ 修改：使用 DATASET_DIR
        "use_embedding": True,
        "use_llm": True,
        "fusion_method": "RRF",
        "rrf_k": 60
    }
}

# ============= 获取当前配置 =============
train_config = TRAIN_CONFIGS[TRAIN_MODE]
eval_config = EVAL_CONFIGS[EVAL_MODE]

# 评估方法标识（用于显示和日志命名）
eval_method_name = eval_config['method_name']

# 训练相关配置（仅用于训练脚本）
exp_name = train_config["memory_name"]
initial_memory_dir = train_config["initial_dir"]
MEMORY_BASE_DIR = f"memory/{exp_name}"

# ✅ 日志输出目录（使用 DATASET_DIR）
LOG_DIR = f"log/{DATASET_DIR}"

# ============= 模型配置 =============
candidate_num = 10
model = "glm-4.5"
evaluation_times = 1

# ============= 训练配置 =============
update_negative_samples = True
random_seed = NEGATIVE_SAMPLE_SEED
save_negative_samples = True
save_ranking_results = True
# 断点续训配置

# ============= 断点续训配置 =============
CHECKPOINT_FILE = f"log/{DATASET_DIR}/checkpoint.json"

# ============= 异步配置 =============
async_training_batch_size = 4
async_training_max_concurrent = 4
async_evaluation_batch_size = 5
async_evaluation_max_concurrent =10

# ============= 数据集元信息（向后兼容）=============
dataset_config = current_dataset_meta

# ============= Embedding生成配置 =============
IMAGE_DIR = current_dataset_meta["image_dir"]
EMBEDDING_MODEL = "Alibaba-NLP/gme-Qwen2-VL-7B-Instruct"
EMBEDDING_USER_DIR = f"memory/AgentCF_{DATASET_DIR}_description/user"  # ✅ 修改：使用 DATASET_DIR
EMBEDDING_OUTPUT_DIR = f"dataset/embeddings/{DATASET_DIR}"  # ✅ 修改：使用 DATASET_DIR

# ============= 消融实验配置 =============
ABLATION_TYPE = "no_user"          # 可选: "auto" | "no_user" | "no_item"
ABLATION_BASE_MODE = "description"  # 可选: "basic" | "description"

# 消融实验配置字典
ABLATION_TRAIN_CONFIGS = {
    "auto_basic": {
        "initial_dir": f"dataset/initial_basic/{DATASET_DIR}",
        "memory_name": f"AgentCF_{DATASET_DIR}_ablation_auto_basic",
        "ablation_type": "auto"
    },
    "auto_description": {
        "initial_dir": f"dataset/initial_description/{DATASET_DIR}",
        "memory_name": f"AgentCF_{DATASET_DIR}_ablation_auto_description",
        "ablation_type": "auto"
    },
    "no_user_basic": {
        "initial_dir": f"dataset/initial_basic/{DATASET_DIR}",
        "memory_name": f"AgentCF_{DATASET_DIR}_ablation_no_user_basic",
        "ablation_type": "no_user"
    },
    "no_user_description": {
        "initial_dir": f"dataset/initial_description/{DATASET_DIR}",
        "memory_name": f"AgentCF_{DATASET_DIR}_ablation_no_user_description",
        "ablation_type": "no_user"
    },
    "no_item_basic": {
        "initial_dir": f"dataset/initial_basic/{DATASET_DIR}",
        "memory_name": f"AgentCF_{DATASET_DIR}_ablation_no_item_basic",
        "ablation_type": "no_item"
    },
    "no_item_description": {
        "initial_dir": f"dataset/initial_description/{DATASET_DIR}",
        "memory_name": f"AgentCF_{DATASET_DIR}_ablation_no_item_description",
        "ablation_type": "no_item"
    }
}

# 获取当前消融实验配置
ablation_config_key = f"{ABLATION_TYPE}_{ABLATION_BASE_MODE}"
ablation_train_config = ABLATION_TRAIN_CONFIGS[ablation_config_key]
ablation_exp_name = ablation_train_config["memory_name"]
ablation_initial_memory_dir = ablation_train_config["initial_dir"]
ABLATION_MEMORY_DIR = f"memory/{ablation_exp_name}"
ABLATION_LOG_DIR = f"log_ablation/{DATASET_DIR}"


ABLATION_LOG_DIR = f"log_ablation/{DATASET_DIR}"

# ✅ 新增：消融实验评估配置
ABLATION_EVAL_CONFIGS = {
    "auto_basic": {
        "method_name": f"AgentCF_{DATASET_DIR}_ablation_auto_basic",
        "memory_dir": f"memory/AgentCF_{DATASET_DIR}_ablation_auto_basic",
        "ablation_type": "auto"
    },
    "auto_description": {
        "method_name": f"AgentCF_{DATASET_DIR}_ablation_auto_description",
        "memory_dir": f"memory/AgentCF_{DATASET_DIR}_ablation_auto_description",
        "ablation_type": "auto"
    },
    "no_user_basic": {
        "method_name": f"AgentCF_{DATASET_DIR}_ablation_no_user_basic",
        "memory_dir": f"memory/AgentCF_{DATASET_DIR}_ablation_no_user_basic",
        "ablation_type": "no_user"
    },
    "no_user_description": {
        "method_name": f"AgentCF_{DATASET_DIR}_ablation_no_user_description",
        "memory_dir": f"memory/AgentCF_{DATASET_DIR}_ablation_no_user_description",
        "ablation_type": "no_user"
    },
    "no_item_basic": {
        "method_name": f"AgentCF_{DATASET_DIR}_ablation_no_item_basic",
        "memory_dir": f"memory/AgentCF_{DATASET_DIR}_ablation_no_item_basic",
        "ablation_type": "no_item"
    },
    "no_item_description": {
        "method_name": f"AgentCF_{DATASET_DIR}_ablation_no_item_description",
        "memory_dir": f"memory/AgentCF_{DATASET_DIR}_ablation_no_item_description",
        "ablation_type": "no_item"
    }
}

# 获取消融实验配置key的辅助函数
def get_ablation_config_key(ablation_type, base_mode):
    """生成消融实验配置的key"""
    return f"{ablation_type}_{base_mode}"

# ============= 长记忆评估配置 =============
ENABLE_LONG_MEMORY_EVAL = True  # True 时启用 AgentCF_long_memory_eval.py
LONG_MEMORY_DATASETS = ["CDs"]  # 可按需调整
LONG_MEMORY_ROUNDS = [1,2, 3, 4,5]  # user-long 中要使用的轮次
LONG_MEMORY_PROMPT_TYPE = "long_basic"  # prompt.py 中的模板 key
LONG_MEMORY_LOG_ROOT = "log_long_eval"
# "Cell_Phones", "Fashion"


# ============= 属性级别监督配置 =============创新点1
ENABLE_ATTRIBUTE_GUIDANCE = True  # 是否启用属性级别引导
ENABLE_SEPARATE_LTM = True
# ATTRIBUTE_DIMENSIONS = [
#     "style", "material", "price", "genre",
#     "functionality", "brand", "color", "quality"
# ]  # 属性维度，可根据数据集调整
# ATTRIBUTE_POLARITY = ["positive", "negative"]  # 属性极性


# ========== UAMG 门控配置（创新点2）==========
ENABLE_MEMORY_GATING = True  # 是否启用记忆门控
GATING_BASE_THRESHOLD = 0.5  # 基础阈值（会自适应调整）
GATING_EARLY_THRESHOLD = 0.7  # 早期阈值（前20次交互）
GATING_LATE_THRESHOLD = 0.3   # 后期阈值（50次交互后）
GATING_TRANSITION_START = 20  # 开始收紧的交互次数
GATING_TRANSITION_END = 50    # 完全收紧的交互次数
