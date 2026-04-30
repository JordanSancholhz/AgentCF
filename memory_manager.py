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


def update_user_memory_from_ltm(userId, new_self_intro):
    """
    固化到长期记忆文件
    """
    from config import MEMORY_BASE_DIR

    user_file = f"{MEMORY_BASE_DIR}/user/user.{userId}"

    # 直接写入新的自我介绍
    with open(user_file, 'w', encoding='utf-8') as f:
        f.write(new_self_intro)

    print(f"✅ [LTM] User {userId} memory updated")


def load_stm_memory(userId, rounds=[2, 3]):
    """
    加载指定轮次的记忆摘要（用于prompt提示）

    参数:
    - userId: 用户ID
    - rounds: 要加载的轮次列表，默认[2, 3]表示Round 2和Round 3

    返回: 字符串，包含这些轮次的偏好摘要
    """
    from config import MEMORY_BASE_DIR
    import os

    stm_summaries = []

    for round_num in rounds:
        # 尝试加载该轮次的记忆快照
        round_path = f"{MEMORY_BASE_DIR}/round_{round_num + 1}/user/user.{userId}"

        if os.path.exists(round_path):
            with open(round_path, 'r', encoding='utf-8') as f:
                memory = f.read().strip()
                # 提取关键偏好（可以简化或截取前100字）
                summary = memory[:100] + "..." if len(memory) > 100 else memory
                stm_summaries.append(f"Round {round_num + 1}: {summary}")

    if stm_summaries:
        return " | ".join(stm_summaries)
    else:
        return None


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
        # 加载STM历史
        stm_file = f"{MEMORY_BASE_DIR}/stm/user_{userId}.json"

        if os.path.exists(stm_file):
            with open(stm_file, 'r', encoding='utf-8') as f:
                stm = json.load(f)
            history = stm.get("history", [])
        else:
            history = []

        # 提取Round 2和Round 3的属性（短记忆）
        round_2_attrs = history[2].get("extracted_attrs", {}) if len(history) > 2 else {}
        round_3_attrs = history[3].get("extracted_attrs", {}) if len(history) > 3 else {}

        # 提取Round 0-3的所有属性（长记忆）
        history_attrs_list = [h.get("extracted_attrs", {}) for h in history[:4]]

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

def process_stm_and_update_memory(userId, extracted_attrs, new_self_intro):
    """
    改进版：分离当前记忆和LTM记忆的更新逻辑

    返回: {
        "current_updated": bool,  # 是否更新了当前记忆
        "ltm_updated": bool,      # 是否更新了LTM
        "decision_tag": str
    }
    """
    from config import MEMORY_BASE_DIR
    import time as time_module
    import json
    import os

    # 1. 加载STM
    stm_file = f"{MEMORY_BASE_DIR}/stm/user_{userId}.json"
    os.makedirs(os.path.dirname(stm_file), exist_ok=True)

    if os.path.exists(stm_file):
        with open(stm_file, 'r', encoding='utf-8') as f:
            stm = json.load(f)
    else:
        stm = {"attributes": {}, "history": []}

    # 2. 记录本次交互到history
    stm["history"].append({
        "timestamp": time_module.time(),
        "round": len(stm["history"]),
        "extracted_attrs": extracted_attrs
    })

    # 3. 更新STM维度积分
    for dim, detail in extracted_attrs.items():
        if detail.get("polarity") == "positive":
            if dim not in stm["attributes"]:
                stm["attributes"][dim] = {
                    "count": 0,
                    "total_score": 0,
                    "evidence_items": []
                }

            stm["attributes"][dim]["count"] += 1
            stm["attributes"][dim]["total_score"] += detail.get("score", 0)
            stm["attributes"][dim]["evidence_items"].append(detail.get("item_name"))

    # 4. 立即更新当前记忆（user/user.{userId}）
    update_current_memory(userId, new_self_intro)
    current_updated = True

    # 5. 判断是否达到LTM固化阈值
    LTM_THRESHOLD = 2  # 出现3次才固化到LTM
    verified_dims = [dim for dim, data in stm["attributes"].items()
                     if data["count"] >= LTM_THRESHOLD]

    ltm_updated = False
    if verified_dims:
        # 固化到LTM（user-ltm/user.{userId}）
        update_ltm_memory(userId, new_self_intro)
        ltm_updated = True

        # 重置计数
        for dim in verified_dims:
            stm["attributes"][dim]["count"] = 0
            stm["attributes"][dim]["evidence_items"] = []

        decision_tag = "UPDATED_TO_LTM"
        print(f"✅ [LTM] User {userId} LTM updated (verified dims: {verified_dims})")
    else:
        decision_tag = "UPDATED_CURRENT_ONLY"

    # 6. 持久化STM
    with open(stm_file, 'w', encoding='utf-8') as f:
        json.dump(stm, f, ensure_ascii=False, indent=2)

    return {
        "current_updated": current_updated,
        "ltm_updated": ltm_updated,
        "decision_tag": decision_tag
    }

### 2.2 辅助函数

def update_current_memory(userId, new_self_intro):
    """
    更新当前记忆（快速响应）
    """
    from config import MEMORY_BASE_DIR
    import os

    user_file = f"{MEMORY_BASE_DIR}/user/user.{userId}"
    os.makedirs(os.path.dirname(user_file), exist_ok=True)

    with open(user_file, 'w', encoding='utf-8') as f:
        f.write(new_self_intro)

    print(f"✅ [Current] User {userId} current memory updated")


def update_ltm_memory(userId, new_self_intro):
    """
    更新长期记忆（高质量基线）
    """
    from config import MEMORY_BASE_DIR
    import os

    ltm_file = f"{MEMORY_BASE_DIR}/user-ltm/user.{userId}"
    os.makedirs(os.path.dirname(ltm_file), exist_ok=True)

    with open(ltm_file, 'w', encoding='utf-8') as f:
        f.write(new_self_intro)

    print(f"✅ [LTM] User {userId} LTM memory updated")


def load_user_memory(userId):
    """
    加载用户当前记忆
    """
    from config import MEMORY_BASE_DIR
    import os

    current_file = f"{MEMORY_BASE_DIR}/user/user.{userId}"

    if os.path.exists(current_file):
        with open(current_file, 'r', encoding='utf-8') as f:
            return f.read().strip()
    else:
        return "I am a new user."


def load_ltm_memory(userId):
    """
    加载LTM记忆（如果存在）
    """
    from config import MEMORY_BASE_DIR
    import os

    ltm_file = f"{MEMORY_BASE_DIR}/user-ltm/user.{userId}"

    if os.path.exists(ltm_file):
        with open(ltm_file, 'r', encoding='utf-8') as f:
            return f.read().strip()
    else:
        return None