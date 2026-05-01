import re
import json
import os

from config import MEMORY_BASE_DIR
import time as time_module

# STM 存储路径
STM_DIR = f"{MEMORY_BASE_DIR}/stm"
os.makedirs(STM_DIR, exist_ok=True)


def parse_attribute_rationale(response_text):
    """
    使用正则从 LLM 输出中提取属性字典
    匹配格式: - [attribute]: [item_name] | [positive/negative] | [score]
    返回: {dimension: {item_name, polarity, score}}
    """
    attributes = {}
    # 匹配 - [genre]: Item Name | positive | 5 这种格式
    pattern = r"-\s*\[(.*?)\]:\s*(.*?)\s*\|\s*(positive|negative)\s*\|\s*(\d+)"
    matches = re.findall(pattern, response_text)

    for match in matches:
        attr_dim = match[0].strip()
        item_name = match[1].strip()
        polarity = match[2].strip()
        score = int(match[3].strip())

        # 使用 dimension 作为 key
        attributes[attr_dim] = {
            "item_name": item_name,
            "polarity": polarity,
            "score": score
        }
    return attributes


# def process_stm_and_check_ltm(userId, extracted_attrs, new_self_intro):
#     """
#     记录属性历史并判断是否固化到LTM
#     """
#     from config import MEMORY_BASE_DIR
#     import time as time_module
#     import json
#     import os
#
#     # 1. 加载STM
#     stm_file = f"{MEMORY_BASE_DIR}/stm/user_{userId}.json"
#     os.makedirs(os.path.dirname(stm_file), exist_ok=True)
#
#     if os.path.exists(stm_file):
#         with open(stm_file, 'r', encoding='utf-8') as f:
#             stm = json.load(f)
#     else:
#         stm = {"attributes": {}, "history": []}
#
#     # 2. 记录本次交互到history（关键：保留每轮的属性快照）
#     stm["history"].append({
#         "timestamp": time_module.time(),
#         "round": len(stm["history"]),
#         "extracted_attrs": extracted_attrs  # 完整的属性字典
#     })
#
#     # 3. 更新STM维度积分
#     for dim, detail in extracted_attrs.items():
#         if detail.get("polarity") == "positive":
#             if dim not in stm["attributes"]:
#                 stm["attributes"][dim] = {
#                     "count": 0,
#                     "total_score": 0,
#                     "evidence_items": []
#                 }
#
#             stm["attributes"][dim]["count"] += 1
#             stm["attributes"][dim]["total_score"] += detail.get("score", 0)
#             stm["attributes"][dim]["evidence_items"].append(detail.get("item_name"))
#
#     # 4. 固化判断（属性出现3次以上固化到LTM）
#     LTM_THRESHOLD = 2
#     verified_dims = [dim for dim, data in stm["attributes"].items()
#                      if data["count"] >= LTM_THRESHOLD]
#
#     if verified_dims:
#         # 固化到LTM
#         update_user_memory_from_ltm(userId, new_self_intro)
#
#         # 重置计数
#         for dim in verified_dims:
#             stm["attributes"][dim]["count"] = 0
#             stm["attributes"][dim]["evidence_items"] = []
#
#         decision_tag = "UPDATED_TO_LTM"
#     else:
#         decision_tag = "KEEP_IN_STM"
#
#     # 5. 持久化STM
#     with open(stm_file, 'w', encoding='utf-8') as f:
#         json.dump(stm, f, ensure_ascii=False, indent=2)
#
#     return decision_tag


def load_stm_attributes(userId, rounds=[2, 3]):
    """
    加载指定轮次的属性提取结果（用于prompt提示）

    参数:
    - userId: 用户ID
    - rounds: 要加载的轮次列表，默认[2, 3]表示Round 2和Round 3

    返回: 列表，包含这些轮次的属性
    """
    from config import MEMORY_BASE_DIR
    import os
    import json

    history_file = f"{MEMORY_BASE_DIR}/stm_history/user_{userId}.json"

    if not os.path.exists(history_file):
        return None

    # 加载History
    with open(history_file, 'r', encoding='utf-8') as f:
        history_data = json.load(f)

    stm_attributes = []

    for round_num in rounds:
        # 从history中找到对应轮次的属性
        for entry in history_data["history"]:
            if entry["round"] == round_num:
                stm_attributes.append({
                    "round": round_num + 1,  # 转换为1-based显示
                    "attributes": entry["extracted_attrs"]
                })
                break

    return stm_attributes if stm_attributes else None

def compute_stm_score_two_rounds(current_attrs, round_2_attrs, round_3_attrs):
    """
    计算短期记忆分数：当前轮与倒数第二轮和倒数第三轮的平均相似度

    参数:
    - current_attrs: 当前轮（Round 4）的属性
    - round_2_attrs: Round 2的属性
    - round_3_attrs: Round 3的属性

    返回: 0-1之间的分数
    """
    if not current_attrs:
        return 0.0

    # 计算与Round 3的相似度
    score_round_3 = compute_stm_score(current_attrs, round_3_attrs)

    # 计算与Round 2的相似度
    score_round_2 = compute_stm_score(current_attrs, round_2_attrs)

    # 加权平均（Round 3权重更高，因为更近）
    stm_score = 0.6 * score_round_3 + 0.4 * score_round_2

    return stm_score

def compute_stm_score(current_attrs, previous_attrs):
    """
    计算短期记忆分数：当前轮与前一轮的属性相关性

    参数:
    - current_attrs: 当前轮属性 {dim: {"polarity": ..., "score": ...}}
    - previous_attrs: 前一轮属性 {dim: {"polarity": ..., "score": ...}}

    返回: 0-1之间的分数
    """
    if not previous_attrs:
        # 第0轮，没有前一轮，返回中性分数
        return 0.5

    if not current_attrs:
        return 0.0

    current_dims = set(current_attrs.keys())
    prev_dims = set(previous_attrs.keys())
    overlap_dims = current_dims & prev_dims

    if len(current_dims) == 0:
        return 0.0

    # 1. 维度重叠率（权重0.5）
    overlap_ratio = len(overlap_dims) / len(current_dims)

    # 2. 极性一致性（权重0.5）
    if len(overlap_dims) > 0:
        polarity_match = sum(
            1 for dim in overlap_dims
            if current_attrs[dim]["polarity"] == previous_attrs[dim]["polarity"]
        ) # 所有属性中有多少个属性的正负性是一致的
        polarity_consistency = polarity_match / len(overlap_dims)
    else:
        polarity_consistency = 0.0

    # STM分数
    stm_score = 0.5 * overlap_ratio + 0.5 * polarity_consistency

    return stm_score

### 2.2 LTM分数计算（当前轮 vs 所有历史）
def compute_ltm_score(current_attrs, history_attrs_list):
    """
    计算长期记忆分数：当前轮与所有历史轮次的属性一致性

    参数:
    - current_attrs: 当前轮属性
    - history_attrs_list: 历史所有轮次的属性列表 [round0_attrs, round1_attrs, ...]

    返回: 0-1之间的分数
    """
    if not history_attrs_list:
        # 第0轮，没有历史，返回中性分数
        return 0.5

    if not current_attrs:
        return 0.0

    # 统计每个维度在历史中的出现情况
    dim_history = {}

    for hist_attrs in history_attrs_list:
        for dim, detail in hist_attrs.items():
            if dim not in dim_history:
                dim_history[dim] = {
                    "count": 0,
                    "polarity_list": []
                }
            dim_history[dim]["count"] += 1
            dim_history[dim]["polarity_list"].append(detail["polarity"])

    # 计算当前轮与历史的一致性
    current_dims = set(current_attrs.keys())
    hist_dims = set(dim_history.keys())
    overlap_dims = current_dims & hist_dims

    if len(current_dims) == 0:
        return 0.0

    # 1. 维度重叠率（权重0.4）
    overlap_ratio = len(overlap_dims) / len(current_dims)

    # 2. 极性一致性（权重0.6）
    # 对于重叠的维度，检查当前极性是否与历史主导极性一致
    if len(overlap_dims) > 0:
        polarity_match = 0
        for dim in overlap_dims:
            # 历史主导极性（出现最多的极性）
            polarity_counts = {}
            for p in dim_history[dim]["polarity_list"]:
                polarity_counts[p] = polarity_counts.get(p, 0) + 1
            dominant_polarity = max(polarity_counts, key=polarity_counts.get) # 主要是看每个属性的极性在历史轮数里面是正向多还是负向多

            # 当前极性是否匹配
            if current_attrs[dim]["polarity"] == dominant_polarity:
                polarity_match += 1

        polarity_consistency = polarity_match / len(overlap_dims)
    else:
        polarity_consistency = 0.0

    # LTM分数
    ltm_score = 0.4 * overlap_ratio + 0.6 * polarity_consistency

    return ltm_score

### 2.3 综合门控函数
def evaluate_memory_gate(userId, round_num, current_attrs, is_choice_right):
    """
    动态记忆门控评估

    参数:
    - userId: 用户ID
    - round_num: 当前轮次（0-based，0-4）
    - current_attrs: 当前轮提取的属性
    - is_choice_right: 是否选择正确

    返回: {
        "gate_score": float,
        "should_update": bool,
        "stm_score": float,
        "ltm_score": float,
        "threshold": float
    }
    """
    from config import MEMORY_BASE_DIR
    import json
    import os

    # 1. Round 0-3: 强制通过
    if round_num < 4:
        return {
            "gate_score": 1.0,
            "should_update": True,
            "stm_score": 0.0,
            "ltm_score": 0.0,
            "threshold": 0.0,
            "weights": {"alpha": 0.0, "beta": 0.0},
            "round_num": round_num,
            "history_count": 0
        }

    # 2. Round 4: 启用长短记忆门控
    elif round_num == 4:
        # 加载History（从stm_history而不是stm）
        history_file = f"{MEMORY_BASE_DIR}/stm_history/user_{userId}.json"

        if os.path.exists(history_file):
            with open(history_file, 'r', encoding='utf-8') as f:
                history_data = json.load(f)
            history = history_data.get("history", [])
        else:
            history = []

        # 提取Round 2和Round 3的属性（短记忆）
        round_2_attrs = {}
        round_3_attrs = {}

        for entry in history:
            if entry["round"] == 2:
                round_2_attrs = entry.get("extracted_attrs", {})
            elif entry["round"] == 3:
                round_3_attrs = entry.get("extracted_attrs", {})

        # 提取Round 0-3的所有属性（长记忆）
        history_attrs_list = []
        for entry in history:
            if entry["round"] < 4:  # 只取Round 0-3
                history_attrs_list.append(entry.get("extracted_attrs", {}))

        # 计算STM分数（与Round 2和Round 3比较）
        stm_score = compute_stm_score_two_rounds(current_attrs, round_2_attrs, round_3_attrs)

        # 计算LTM分数（与Round 0-3比较）
        ltm_score = compute_ltm_score(current_attrs, history_attrs_list)

        # 综合门控分数
        alpha, beta = 0.6, 0.4  # LTM权重更高
        gate_score = alpha * ltm_score + beta * stm_score

        # 阈值
        threshold = 0.5 if is_choice_right else 0.6

        should_update = gate_score >= threshold

        return {
            "gate_score": gate_score,
            "should_update": should_update,
            "stm_score": stm_score,
            "ltm_score": ltm_score,
            "threshold": threshold,
            "weights": {"alpha": alpha, "beta": beta},
            "round_num": round_num,
            "history_count": len(history)
        }

def save_stm_and_history(userId, extracted_attrs, round_num):
    """
    保存当前轮次的STM和累积的History

    参数:
    - userId: 用户ID
    - extracted_attrs: 当前轮次提取的属性
    - round_num: 当前轮次（0-based）
    """
    from config import MEMORY_BASE_DIR
    import json
    import os
    import time as time_module

    # ========== 1. 保存STM（当前轮次，覆盖式）==========
    stm_dir = f"{MEMORY_BASE_DIR}/stm"
    os.makedirs(stm_dir, exist_ok=True)

    stm_file = f"{stm_dir}/user_{userId}.json"
    stm_data = {
        "user_id": userId,
        "current_round": round_num,
        "timestamp": time_module.time(),
        "extracted_attrs": extracted_attrs
    }

    with open(stm_file, 'w', encoding='utf-8') as f:
        json.dump(stm_data, f, ensure_ascii=False, indent=2)

    print(f"✅ [STM] User {userId} Round {round_num} saved")

    # ========== 2. 追加到History（累积式）==========
    history_dir = f"{MEMORY_BASE_DIR}/stm_history"
    os.makedirs(history_dir, exist_ok=True)

    history_file = f"{history_dir}/user_{userId}.json"

    # 加载或初始化History
    if os.path.exists(history_file):
        with open(history_file, 'r', encoding='utf-8') as f:
            history_data = json.load(f)
    else:
        history_data = {
            "user_id": userId,
            "history": []
        }

    # 追加当前轮次
    history_data["history"].append({
        "round": round_num,
        "timestamp": time_module.time(),
        "extracted_attrs": extracted_attrs
    })

    # 保存History
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(history_data, f, ensure_ascii=False, indent=2)

    print(f"✅ [History] User {userId} Round {round_num} appended to history")

def generate_ltm_from_history(userId, min_occurrences=2):
    """
    从History文件中提取稳定属性，生成LTM prompt

    参数:
    - userId: 用户ID
    - min_occurrences: 最小出现次数（默认2，即至少出现3次）

    返回: 格式化的属性字典，用于prompt构建
    """
    from config import MEMORY_BASE_DIR
    import json
    import os

    history_file = f"{MEMORY_BASE_DIR}/stm_history/user_{userId}.json"

    if not os.path.exists(history_file):
        return None

    # 加载History
    with open(history_file, 'r', encoding='utf-8') as f:
        history_data = json.load(f)

    # 统计每个维度的正向属性出现次数
    dimension_stats = {}

    for entry in history_data["history"]:
        for dim, detail in entry["extracted_attrs"].items():
            if detail.get("polarity") == "positive":
                if dim not in dimension_stats:
                    dimension_stats[dim] = {
                        "count": 0,
                        "total_score": 0,
                        "items": []
                    }

                dimension_stats[dim]["count"] += 1
                dimension_stats[dim]["total_score"] += detail.get("score", 0)
                dimension_stats[dim]["items"].append(detail.get("item_name"))

    # 筛选出现次数 > min_occurrences 的维度
    verified_dims = {
        dim: stats for dim, stats in dimension_stats.items()
        if stats["count"] >= min_occurrences  # > 2 表示至少3次
    }

    if not verified_dims:
        return None

    # 返回格式化的属性字典（用于prompt构建）
    ltm_attributes = {}
    for dim, stats in verified_dims.items():
        avg_score = stats["total_score"] / stats["count"]
        ltm_attributes[dim] = {
            "count": stats["count"],
            "avg_score": avg_score,
            "items": stats["items"]
        }

    return ltm_attributes

