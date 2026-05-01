import math

from memory_manager import parse_attribute_rationale, evaluate_memory_gate, save_stm_and_history, \
    generate_ltm_from_history, load_stm_attributes
from prompt import *
import random
import re
from fuzzywuzzy import fuzz
import shutil
import os
from dataPrepare import createInterDF, createItemDF, createRandomDF
from request1 import async_client
import json
import asyncio
import threading
from tqdm import tqdm
import time
from copy import deepcopy

# ✅ 从config导入所有配置，零硬编码
from config import (
    model, train_file, random_file, item_file,
    exp_name, initial_memory_dir, update_negative_samples,
    random_seed, save_negative_samples,
    async_training_batch_size, async_training_max_concurrent,
    USE_FIXED_NEGATIVES, TRAIN_NEGATIVES_FILE,
    MEMORY_BASE_DIR, LOG_DIR,
    ENABLE_ATTRIBUTE_GUIDANCE, ENABLE_MEMORY_GATING, ENABLE_SEPARATE_LTM  # 新增
)

mode = "train"


# ========== UAMG 门控函数（针对 5 轮交互与 Hard Negative 优化，创新点2）==========
def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


async def evaluate_update_uncertainty(
        user_memory: str,
        new_user_update: str,
        pos_item_title: str,
        neg_item_title: str,
        pos_item_memory: str,
        neg_item_memory: str,
        decision_reasoning: str,
        is_choice_right: bool
) -> dict:
    """
    让 LLM 评估 UAMG 的四个核心指标，并计算门控值 gt
    """
    # 错判代表高价值纠偏，直接放行 (Hard Negative)
    if not is_choice_right:
        return {
            'gate_score': 1.0,  # 强制最高门控得分，必然放行
            'confidence': 10.0,
            'metrics': {'Ct': 1.0, 'Dt': 1.0, 'St': 1.0, 'Gt': 1.0},
            'raw_response': "LLM选择错误，触发Hard Negative强制纠偏，直接放行。"
        }

    reflection_prompt = f"""Evaluate update quality (0.0-1.0):

    Old profile: {user_memory[:200]}
    New update: {new_user_update[:200]}
    Chose: {pos_item_title}
    Rejected: {neg_item_title}

    Ct (clarity): reasoning clear?
    Dt (difference): items obviously different?
    St (consistency): update matches old profile?
    Gt (gain): will improve ranking?

    JSON: {{"Ct": 0.85, "Dt": 0.70, "St": 0.90, "Gt": 0.60}}
    """

    #     reflection_prompt = f"""You are an evaluator deciding whether a user's memory profile should be updated based on a recent interaction.
    #
    # **Previous user profile:**
    # {user_memory}
    #
    # **Your proposed update:**
    # {new_user_update}
    #
    # **Interaction context:**
    # - User chose: {pos_item_title} - {pos_item_memory[:200]}...
    # - User rejected: {neg_item_title} - {neg_item_memory[:200]}...
    # - Reason for choice: {decision_reasoning}
    #
    # Evaluate the following 4 gating indicators on a scale of 0.0 to 1.0:
    # 1. "Ct" (Decision Confidence): Is the reasoning clear about user preferences? Does it show a clear positive/negative contrast? (1.0 = extremely clear)
    # 2. "Dt" (Attribute Difference Intensity): Are the positive and negative samples obviously different on key attributes? (1.0 = huge difference, 0.0 = almost identical)
    # 3. "St" (Memory Consistency): Is the new update consistent with the previous user profile? (1.0 = fully consistent, 0.0 = severe conflict)
    # 4. "Gt" (Ranking Gain Potential): Will this update significantly improve the matching score of the chosen item or fix an obvious profile deviation? (1.0 = high gain)
    #
    # Output ONLY a JSON object with the 4 scores. Example format:
    # {{
    #     "Ct": 0.85,
    #     "Dt": 0.70,
    #     "St": 0.90,
    #     "Gt": 0.60
    # }}
    # """

    try:
        response = await async_client.call_api_async(reflection_prompt, model)

        # 解析 JSON 提取四个指标
        match = re.search(r'\{.*?\}', response, re.DOTALL)
        if match:
            scores = json.loads(match.group(0))
        else:
            # 如果没按 JSON 输出，尝试正则粗略提取
            scores = {
                "Ct": float(re.search(r'"?Ct"?\s*:\s*([\d.]+)', response).group(1) or 0.5),
                "Dt": float(re.search(r'"?Dt"?\s*:\s*([\d.]+)', response).group(1) or 0.5),
                "St": float(re.search(r'"?St"?\s*:\s*([\d.]+)', response).group(1) or 0.5),
                "Gt": float(re.search(r'"?Gt"?\s*:\s*([\d.]+)', response).group(1) or 0.5)
            }

        Ct = float(scores.get("Ct", 0.5))
        Dt = float(scores.get("Dt", 0.5))
        St = float(scores.get("St", 0.5))
        Gt = float(scores.get("Gt", 0.5))

        # === 公式计算 ===
        # 超参数：可根据实验效果调节 α, β, γ, δ
        alpha, beta, gamma, delta = 1.0, 1.0, 1.0, 1.0

        # gt = σ(αCt + βDt + γGt - δ(1 - St))
        z = alpha * Ct + beta * Dt + gamma * Gt - delta * (1.0 - St)
        gt = sigmoid(z)

        return {
            'gate_score': gt,
            'confidence': Ct * 10,  # 为了兼容你后面的日志代码
            'metrics': {'Ct': Ct, 'Dt': Dt, 'St': St, 'Gt': Gt},
            'raw_response': response
        }

    except Exception as e:
        print(f"⚠️ UAMG 门控评估失败: {e}，回退到默认放行")
        return {
            'gate_score': 0.8,
            'confidence': 8.0,
            'metrics': {'Ct': 0.5, 'Dt': 0.5, 'St': 0.5, 'Gt': 0.5},
            'raw_response': "Error fallback"
        }


def compute_adaptive_threshold(interaction_count: int) -> float:
    """
    计算自适应门控阈值 τ (Tau)
    注意：现在逻辑反转了，门控得分 gt > τ 时才更新。
    """
    if interaction_count <= 1:
        # 第 1-2 次交互：早期宽松，较低阈值即可更新
        return 0.65
    elif interaction_count <= 3:
        # 第 3-4 次交互：中期收紧
        return 0.75
    else:
        # 第 5 次及以后：后期极其严格，必须各维度分数都很高才能更新
        return 0.85


def should_update_memory(gate_score: float, interaction_count: int) -> bool:
    """
    门控决策：若 gt > τ，执行更新
    """
    threshold = compute_adaptive_threshold(interaction_count)
    return gate_score > threshold


# ========== UAMG 门控函数结束 ==========


# ============= 断点续训辅助函数 =============
def save_checkpoint(batch_idx, total_batches):
    """保存检查点"""
    from config import CHECKPOINT_FILE
    checkpoint = {"batch": batch_idx, "total": total_batches}
    os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
    with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
        json.dump(checkpoint, f)
    print(f"💾 检查点已保存: {batch_idx + 1}/{total_batches} ({(batch_idx + 1) / total_batches * 100:.1f}%)")


def load_checkpoint():
    """加载检查点"""
    from config import CHECKPOINT_FILE
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data
    return None


def clear_checkpoint():
    """清除检查点"""
    from config import CHECKPOINT_FILE
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print("🗑️  检查点已清除")


# ============= 新增：加载固定负样本 =============
def load_fixed_train_negatives():
    """加载预生成的固定训练负样本"""
    if not USE_FIXED_NEGATIVES:
        return None

    if not os.path.exists(TRAIN_NEGATIVES_FILE):
        print(f"❌ 固定负样本文件不存在: {TRAIN_NEGATIVES_FILE}")
        print(f"请先运行: python negative_sampler.py --seed {random_seed} --verify")
        exit(1)

    with open(TRAIN_NEGATIVES_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"✅ 已加载固定训练负样本: {TRAIN_NEGATIVES_FILE}")
    print(f"   正负样本对数: {data['metadata']['total_pairs']}")

    return data['negatives']


def initialize_memory():
    """初始化记忆（支持断点续训）"""
    from config import CHECKPOINT_FILE

    # ✅ 如果记忆已存在
    if os.path.exists(os.path.join(MEMORY_BASE_DIR, "item")) or os.path.exists(os.path.join(MEMORY_BASE_DIR, "user")):
        # ✅ 检查是否有断点文件
        if os.path.exists(CHECKPOINT_FILE):
            # 断点续训：保留现有memory
            print(f"✅ 发现断点，使用现有记忆: {MEMORY_BASE_DIR}")
            return
        else:
            # 没有断点但memory存在：用户需要选择
            print(f"⚠️  记忆已存在: {MEMORY_BASE_DIR}")
            print("检测到已有记忆但没有断点文件")
            choice = input("1-继续使用现有记忆  2-删除重新开始 (1/2): ")
            if choice == "2":
                shutil.rmtree(MEMORY_BASE_DIR)
                print(f"🗑️  已删除: {MEMORY_BASE_DIR}")
            else:
                print(f"✅ 使用现有记忆继续训练")
                return

    # ✅ 从头开始：复制初始记忆
    print(f"🆕 正在创建新记忆目录: {MEMORY_BASE_DIR}")
    os.makedirs(MEMORY_BASE_DIR, exist_ok=True)
    shutil.copytree(f"{initial_memory_dir}/item", os.path.join(MEMORY_BASE_DIR, "item"))
    shutil.copytree(f"{initial_memory_dir}/user", os.path.join(MEMORY_BASE_DIR, "user"))
    shutil.copytree(f"{initial_memory_dir}/user-long", os.path.join(MEMORY_BASE_DIR, "user-long"))
    print(f"✅ 记忆已初始化: {MEMORY_BASE_DIR}")


def save_memory(ratio):
    """保存当前记忆状态"""
    src_folder = MEMORY_BASE_DIR
    dst_folder = f"{MEMORY_BASE_DIR}_{ratio}"
    try:
        shutil.copytree(src_folder, dst_folder)
        print(f"记忆已保存到: {dst_folder}")
    except Exception as e:
        print(f"保存记忆时出错: {e}")


# ✅ 修改：支持固定负样本
def get_neg_item_id(userId, pos_itemId, random_df, used_negatives=None, round_num=None, fixed_negatives=None):
    """获取负样本ID"""
    # 优先使用固定负样本
    if fixed_negatives is not None and round_num is not None:
        key = f"user_{userId}_pos_{pos_itemId}_round_{round_num}"
        if key in fixed_negatives:
            return fixed_negatives[key]
        print(f"⚠️ 固定负样本中未找到 {key}，回退到随机选择")

    # 回退到随机选择
    user_id = int(userId)
    pos_item = str(pos_itemId).strip()

    user_row = random_df[random_df['user_id'] == user_id]
    if len(user_row) == 0:
        return None if used_negatives is not None else None

    candidates = user_row['candidates'].values[0]
    valid_candidates = [c for c in candidates if c != pos_item]

    if used_negatives is not None:
        valid_candidates = [c for c in valid_candidates if c not in used_negatives]

    if len(valid_candidates) == 0:
        valid_candidates = [c for c in candidates if c != pos_item]
        if len(valid_candidates) == 0:
            return None

    return random.choice(valid_candidates)


def create_round_based_batches(interDF):
    """按轮次创建训练批次"""
    batches = []
    user_groups = interDF.groupby('user_id:token')
    all_users = list(user_groups.groups.keys())

    max_rounds = 5

    print(f"📊 批次配置: 用户数={len(all_users)}, 轮次={max_rounds}, 批次大小={async_training_batch_size}")

    for round_num in range(max_rounds):
        user_batches = [all_users[i:i + async_training_batch_size] for i in
                        range(0, len(all_users), async_training_batch_size)]

        for batch_idx, user_batch in enumerate(user_batches):
            batch = []
            for user_id in user_batch:
                if user_id in user_groups.groups:
                    user_interactions = user_groups.get_group(user_id)
                    if round_num < len(user_interactions):
                        interaction = user_interactions.iloc[round_num]
                    else:
                        interaction_index = round_num % len(user_interactions)
                        interaction = user_interactions.iloc[interaction_index]
                    batch.append(interaction)

            if batch:
                batches.append(batch)

    print(f"📦 总共创建 {len(batches)} 个批次")
    return batches


async def process_single_interaction_async(interaction, batch_idx, round_num, itemDF, random_df,
                                           memory_lock, negative_samples_log, used_negatives, fixed_negatives):
    """异步处理单个交互"""
    try:
        pos_itemId = str(interaction["item_id:token"]).strip()
        userId = str(interaction["user_id:token"]).strip()

        # ✅ 使用固定负样本
        neg_itemId = get_neg_item_id(userId, pos_itemId, random_df, used_negatives, round_num, fixed_negatives)

        if neg_itemId:
            used_negatives.add(neg_itemId)

        # ✅ 使用config中的路径
        with memory_lock:
            with open(f"{MEMORY_BASE_DIR}/user/user.{userId}", "r", encoding="utf-8") as file:
                user_memory = file.read()
            with open(f"{MEMORY_BASE_DIR}/item/item.{pos_itemId}", "r", encoding="utf-8") as file:
                pos_item_memory = file.read()
            with open(f"{MEMORY_BASE_DIR}/item/item.{neg_itemId}", "r", encoding="utf-8") as file:
                neg_item_memory = file.read()

        pos_item_row = itemDF[itemDF["item_id:token"] == pos_itemId]
        pos_item_title = str(pos_item_row["title:token_seq"].values[0]) if len(
            pos_item_row) > 0 else f"Item {pos_itemId}"

        neg_item_row = itemDF[itemDF["item_id:token"] == neg_itemId]
        neg_item_title = str(neg_item_row["title:token_seq"].values[0]) if len(
            neg_item_row) > 0 else f"Item {neg_itemId}"

        if save_negative_samples:
            interaction_key = f"user_{userId}_pos_{pos_itemId}_round_{round_num}_batch_{batch_idx}"
            negative_samples_log[interaction_key] = {
                "user_id": userId,
                "pos_item_id": pos_itemId,
                "pos_item_title": pos_item_title,
                "neg_item_id": neg_itemId,
                "round_number": round_num,
                "batch_index": batch_idx
            }

        user_description = user_memory
        list_of_item_description = f"title:{neg_item_title.strip()}. description:{neg_item_memory.strip()}\ntitle:{pos_item_title}. description:{pos_item_memory.strip()}"
        system_prompt = system_prompt_template(user_description, list_of_item_description)

        responseText = await async_client.call_api_async(system_prompt, model)
        if not responseText:
            return

        selected_item_title, system_reason = parse_response(responseText)

        pos_similarity = fuzz.ratio(selected_item_title.lower(), pos_item_title.lower())
        neg_similarity = fuzz.ratio(selected_item_title.lower(), neg_item_title.lower())
        is_choice_right = pos_similarity > neg_similarity

        # 新增------------------------------------------------------------------------------
        attribute_analysis = None
        if ENABLE_ATTRIBUTE_GUIDANCE:
            # --- 关键修改点 ---
            if is_choice_right:
                attr_prompt = attribute_analysis_prompt_correct(
                    user_description, pos_item_title, neg_item_title,
                    pos_item_memory, neg_item_memory, system_reason
                )
            else:
                attr_prompt = attribute_analysis_prompt_incorrect(
                    user_description, pos_item_title, neg_item_title,
                    pos_item_memory, neg_item_memory, system_reason
                )

            # 这里直接调用 API，不再通过 get_attribute_analysis 封装以减少改动
            attr_res = await async_client.call_api_with_metrics(attr_prompt, model)
            attribute_analysis = attr_res["content"]
            # if not attribute_analysis or not attribute_analysis.strip():
            #     print("Warning: LLM returned empty attribute analysis.")
            # print("Attribute analysis: True")

        # 原来的
        # user_prompt, item_prompt = create_prompts(user_description, list_of_item_description,
        #                                          pos_item_title, neg_item_title,
        #                                          system_reason, is_choice_right, attribute_analysis)
        # 新增的
        user_prompt, item_prompt = create_prompts(user_description, list_of_item_description,
                                                  pos_item_title, neg_item_title,
                                                  system_reason, is_choice_right,
                                                  attribute_analysis, userId=userId, round_num=round_num)
        # 新增------------------------------------------------------------------------------创新点1

        user_response = await async_client.call_api_async(user_prompt, model)

        # ========== 初始化门控标志 ==========
        should_update = True  # 默认更新（Round 0-3始终更新）

        if user_response:
            # ========== 新增：属性提取和STM保存 ==========
            extracted_attrs = {}
            if ENABLE_ATTRIBUTE_GUIDANCE:
                # 1. 解析属性
                extracted_attrs = parse_attribute_rationale(user_response)

                # 2. 保存STM和History（始终保存，用于历史记录）
                save_stm_and_history(userId, extracted_attrs, round_num)

            # ========== 新增：UAMG 门控（Round 4启用）==========
            if ENABLE_MEMORY_GATING and ENABLE_ATTRIBUTE_GUIDANCE and round_num == 4:
                # 调用门控评估
                gate_result = evaluate_memory_gate(
                    userId, round_num, extracted_attrs, is_choice_right
                )

                print(f"[Gate] User {userId} Round {round_num}: "
                      f"Score={gate_result['gate_score']:.3f} "
                      f"(LTM={gate_result['ltm_score']:.2f}, "
                      f"STM={gate_result['stm_score']:.2f}), "
                      f"Threshold={gate_result['threshold']:.2f}")

                should_update = gate_result['should_update']

            # ========== 根据门控结果决定是否更新 ==========
            if should_update:
                # 更新用户记忆
                update_user_memory(userId, user_response)
                print(f"✅ [Update] User {userId} memory updated")
            else:
                print(f"⛔ [Reject] User {userId} update blocked by gate (low consistency)")
            # ========== 门控结束 ==========

        item_response = await async_client.call_api_async(item_prompt, model)
        if item_response:
            # ========== 物品更新也受门控控制 ==========
            if should_update:
                update_item_memory(pos_itemId, neg_itemId, item_response, update_neg=update_negative_samples)
                print(f"✅ [Update] Items {pos_itemId}/{neg_itemId} updated")
            else:
                print(f"⛔ [Reject] Items {pos_itemId}/{neg_itemId} update blocked by gate")

        print(f"✅ 用户 {userId} 第{round_num + 1}轮完成")

    except Exception as e:
        print(f"❌ 处理交互时出错: {e}")


async def process_batch_async(batch, batch_idx, round_num, itemDF, random_df, memory_lock, negative_samples_log,
                              fixed_negatives):
    """异步处理单个训练批次"""
    used_negatives = set()
    tasks = []

    for interaction in batch:
        task = asyncio.create_task(
            process_single_interaction_async(interaction, batch_idx, round_num, itemDF, random_df,
                                             memory_lock, negative_samples_log, used_negatives, fixed_negatives)
        )
        tasks.append(task)

    await asyncio.gather(*tasks, return_exceptions=True)


async def process_interaction(interDF, itemDF, random_df):
    """处理训练交互（支持断点续训）"""
    fixed_negatives = load_fixed_train_negatives()
    negative_samples_log = {}
    memory_lock = threading.Lock()
    batches = create_round_based_batches(interDF)

    # ✅ 智能断点检测
    start_idx = 0
    checkpoint = load_checkpoint()

    if checkpoint:
        progress = (checkpoint['batch'] + 1) / checkpoint['total'] * 100
        print(f"\n{'=' * 60}")
        print(f"📊 发现断点: 已完成 {checkpoint['batch'] + 1}/{checkpoint['total']} 批次 ({progress:.1f}%)")
        print(f"{'=' * 60}")
        choice = input("\n选择: 1-继续训练  2-从头开始 (1/2): ")

        if choice == "1":
            start_idx = checkpoint['batch'] + 1
            print(f"🔄 从批次 {start_idx + 1} 继续训练...\n")
        else:
            clear_checkpoint()
            print(f"🆕 从头开始训练...\n")

    # ✅ 主训练循环（从start_idx开始）
    try:
        for i in range(start_idx, len(batches)):
            batch = batches[i]
            users_per_round = len(interDF.groupby('user_id:token'))
            batches_per_round = (users_per_round + async_training_batch_size - 1) // async_training_batch_size
            current_round = i // batches_per_round

            # 处理批次
            # asyncio.run(process_batch_async(batch, i % batches_per_round, current_round,
            #                                itemDF, random_df, memory_lock, negative_samples_log, fixed_negatives))
            await process_batch_async(batch, i % batches_per_round, current_round,
                                      itemDF, random_df, memory_lock, negative_samples_log, fixed_negatives)

            # ✅ 保存检查点（每个批次）
            save_checkpoint(i, len(batches))

            # 保存轮次记忆
            if (i + 1) % batches_per_round == 0:
                save_memory(f"round_{current_round + 1}")
                print(f"✅ 第 {current_round + 1} 轮完成")

    except KeyboardInterrupt:
        print("\n⚠️  训练被用户中断")
        print(f"💾 检查点已保存，下次可继续训练")
        raise
    except Exception as e:
        print(f"\n❌ 训练出错: {e}")
        print(f"💾 检查点已保存，可稍后继续")
        raise

    # ✅ 训练完成，清除检查点
    clear_checkpoint()

    # ========== 新增：门控统计分析 ==========创新点2
    from config import ENABLE_MEMORY_GATING

    if ENABLE_MEMORY_GATING and save_negative_samples:
        # 收集门控统计
        gate_stats = {
            'total_updates': 0,
            'accepted': 0,
            'rejected': 0,
            'avg_confidence_accepted': [],
            'avg_confidence_rejected': [],
            'rejection_by_stage': {'early': 0, 'mid': 0, 'late': 0}
        }

        for key, log in negative_samples_log.items():
            if 'gate_log' in log:
                gate = log['gate_log']
                gate_stats['total_updates'] += 1

                if gate['decision'] == 'ACCEPT':
                    gate_stats['accepted'] += 1
                    gate_stats['avg_confidence_accepted'].append(gate['confidence'])
                else:
                    gate_stats['rejected'] += 1
                    gate_stats['avg_confidence_rejected'].append(gate['confidence'])

                    # 统计拒绝发生在哪个阶段
                    if gate['round'] < 20:
                        gate_stats['rejection_by_stage']['early'] += 1
                    elif gate['round'] < 50:
                        gate_stats['rejection_by_stage']['mid'] += 1
                    else:
                        gate_stats['rejection_by_stage']['late'] += 1

        # 计算接受率
        if gate_stats['total_updates'] > 0:
            acceptance_rate = gate_stats['accepted'] / gate_stats['total_updates']

            print("\n" + "=" * 60)
            print("门控统计 (UAMG - 创新点3)")
            print("=" * 60)
            print(f"总更新尝试: {gate_stats['total_updates']}")
            print(f"接受: {gate_stats['accepted']} ({acceptance_rate * 100:.1f}%)")
            print(f"拒绝: {gate_stats['rejected']} ({(1 - acceptance_rate) * 100:.1f}%)")

            if gate_stats['avg_confidence_accepted']:
                print(
                    f"接受更新的平均置信度: {sum(gate_stats['avg_confidence_accepted']) / len(gate_stats['avg_confidence_accepted']):.2f}/10")
            if gate_stats['avg_confidence_rejected']:
                print(
                    f"拒绝更新的平均置信度: {sum(gate_stats['avg_confidence_rejected']) / len(gate_stats['avg_confidence_rejected']):.2f}/10")

            print(f"拒绝分布: 早期={gate_stats['rejection_by_stage']['early']}, "
                  f"中期={gate_stats['rejection_by_stage']['mid']}, "
                  f"后期={gate_stats['rejection_by_stage']['late']}")
            print("=" * 60)

            # 保存详细门控日志
            gate_log_file = f"{LOG_DIR}/gate_logs_{exp_name}.json"
            gate_logs = [log['gate_log'] for log in negative_samples_log.values() if 'gate_log' in log]
            with open(gate_log_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'statistics': gate_stats,
                    'detailed_logs': gate_logs
                }, f, indent=2, ensure_ascii=False)
            print(f"门控日志已保存: {gate_log_file}")
    # ========== 门控统计结束 ==========

    print("\n🎉 训练完成！")

    # 保存负样本日志
    if save_negative_samples:
        log_file = f"{LOG_DIR}/train_negative_samples_{exp_name}.json"
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(negative_samples_log, f, ensure_ascii=False, indent=2)
        print(f"训练负样本日志已保存: {log_file}")


def parse_response(responseText):
    """解析LLM响应"""
    selected_item_title = re.split(r"Choice:|\n", responseText)[1]
    system_reason = re.split(r"Explanation:", responseText)[-1].strip()
    return selected_item_title, system_reason


# 新增------------------------------------------------------------------------------创新点1
# def create_prompts(user_description, list_of_item_description, pos_item_title,
#                    neg_item_title, system_reason, is_choice_right,
#                    attribute_analysis=None):
#     """
#     创建更新提示
#     attribute_analysis: 如果启用属性监督，传入属性分析结果
#     """
#     if ENABLE_ATTRIBUTE_GUIDANCE and attribute_analysis:
#         # 使用属性增强版 prompt
#         if not is_choice_right:
#             user_prompt = user_prompt_system_role(user_description) + '\n' + \
#                           user_prompt_template_with_attr(list_of_item_description, pos_item_title,
#                                                    neg_item_title, system_reason, attribute_analysis)
#             item_prompt = item_prompt_template_with_attr(user_description, list_of_item_description,
#                                                    pos_item_title, neg_item_title,
#                                                    system_reason, attribute_analysis)
#         else:
#             user_prompt = user_prompt_system_role(user_description) + '\n' + \
#                           user_prompt_template_true_with_attr(list_of_item_description, pos_item_title,
#                                                         neg_item_title, system_reason, attribute_analysis)
#             item_prompt = item_prompt_template_true_with_attr(user_description, list_of_item_description,
#                                                         pos_item_title, neg_item_title, attribute_analysis)
#     else:
#         # 使用原版 prompt
#         if not is_choice_right:
#             user_prompt = user_prompt_system_role(user_description) + '\n' + \
#                           user_prompt_template(list_of_item_description, pos_item_title,
#                                                neg_item_title, system_reason)
#             item_prompt = item_prompt_template(user_description, list_of_item_description,
#                                                pos_item_title, neg_item_title, system_reason)
#         else:
#             user_prompt = user_prompt_system_role(user_description) + '\n' + \
#                           user_prompt_template_true(list_of_item_description, pos_item_title,
#                                                     neg_item_title, system_reason)
#             item_prompt = item_prompt_template_true(user_description, list_of_item_description,
#                                                     pos_item_title, neg_item_title)
#     return user_prompt, item_prompt
# 新增------------------------------------------------------------------------------


# 新增------------------------------------------------------------------------------创新点2
def create_prompts(user_description, list_of_item_description, pos_item_title,
                   neg_item_title, system_reason, is_choice_right,
                   attribute_analysis=None, userId=None, round_num=0):
    """
    创建更新提示（分阶段策略）

    参数:
    - user_description: 用户当前记忆
    - list_of_item_description: 物品描述列表
    - pos_item_title: 正样本标题
    - neg_item_title: 负样本标题
    - system_reason: 系统推理
    - is_choice_right: 是否选择正确
    - attribute_analysis: 属性分析结果
    - userId: 用户ID
    - round_num: 当前轮次（0-based，0-4）

    返回: (user_prompt, item_prompt)
    """

    current_memory = user_description
    ltm_prompt = None
    stm_summaries = None

    # ========== Round 0-3: 使用基础prompt（创新点二上面的4个） ==========
    if round_num < 4:
        if ENABLE_ATTRIBUTE_GUIDANCE and attribute_analysis:
            # 使用带属性分析的prompt（不带LTM/STM）
            if not is_choice_right:
                user_prompt = user_prompt_system_role(current_memory) + '\n' + \
                              user_prompt_template_with_attr(
                                  list_of_item_description, pos_item_title,
                                  neg_item_title, system_reason, attribute_analysis)
                item_prompt = item_prompt_template_with_attr(
                    current_memory, list_of_item_description,
                    pos_item_title, neg_item_title, system_reason, attribute_analysis)
            else:
                user_prompt = user_prompt_system_role(current_memory) + '\n' + \
                              user_prompt_template_true_with_attr(
                                  list_of_item_description, pos_item_title,
                                  neg_item_title, system_reason, attribute_analysis)
                item_prompt = item_prompt_template_true_with_attr(
                    current_memory, list_of_item_description,
                    pos_item_title, neg_item_title, attribute_analysis)
        else:
            # 回退到基础版本（不带属性分析）
            if not is_choice_right:
                user_prompt = user_prompt_system_role(current_memory) + '\n' + \
                              user_prompt_template(list_of_item_description, pos_item_title,
                                                   neg_item_title, system_reason)
                item_prompt = item_prompt_template(current_memory, list_of_item_description,
                                                   pos_item_title, neg_item_title, system_reason)
            else:
                user_prompt = user_prompt_system_role(current_memory) + '\n' + \
                              user_prompt_template_true(list_of_item_description, pos_item_title,
                                                        neg_item_title, system_reason)
                item_prompt = item_prompt_template_true(current_memory, list_of_item_description,
                                                        pos_item_title, neg_item_title)

    # ========== Round 4: 使用带LTM+STM的prompt（创新点二下面的4个）==========
    else:  # round_num == 4
        # 加载LTM（从History动态生成，返回属性字典）
        ltm_attributes = None
        if ENABLE_SEPARATE_LTM and userId:
            ltm_attributes = generate_ltm_from_history(userId, min_occurrences=2)

        # 加载STM（Round 2和Round 3的属性）
        stm_attributes = None
        if userId:
            stm_attributes = load_stm_attributes(userId, [2, 3])

        if ENABLE_ATTRIBUTE_GUIDANCE and attribute_analysis:
            # 使用带属性分析+LTM+STM的prompt
            if not is_choice_right:
                user_prompt = user_prompt_system_role(current_memory) + '\n' + \
                              user_prompt_template_with_attr_ltm(
                                  list_of_item_description, pos_item_title,
                                  neg_item_title, system_reason, attribute_analysis,
                                  ltm_attributes, stm_attributes)
                item_prompt = item_prompt_template_with_attr_ltm(
                    current_memory, list_of_item_description,
                    pos_item_title, neg_item_title, system_reason,
                    attribute_analysis, ltm_attributes, stm_attributes)
            else:
                user_prompt = user_prompt_system_role(current_memory) + '\n' + \
                              user_prompt_template_true_with_attr_ltm(
                                  list_of_item_description, pos_item_title,
                                  neg_item_title, system_reason, attribute_analysis,
                                  ltm_attributes, stm_attributes)
                item_prompt = item_prompt_template_true_with_attr_ltm(
                    current_memory, list_of_item_description,
                    pos_item_title, neg_item_title,
                    attribute_analysis, ltm_attributes, stm_attributes)
        else:
            # 回退到基础版本（不带属性分析）
            if not is_choice_right:
                user_prompt = user_prompt_system_role(current_memory) + '\n' + \
                              user_prompt_template(list_of_item_description, pos_item_title,
                                                   neg_item_title, system_reason)
                item_prompt = item_prompt_template(current_memory, list_of_item_description,
                                                   pos_item_title, neg_item_title, system_reason)
            else:
                user_prompt = user_prompt_system_role(current_memory) + '\n' + \
                              user_prompt_template_true(list_of_item_description, pos_item_title,
                                                        neg_item_title, system_reason)
                item_prompt = item_prompt_template_true(current_memory, list_of_item_description,
                                                        pos_item_title, neg_item_title)

    return user_prompt, item_prompt


# 新增------------------------------------------------------------------------------


def update_user_memory(userId, responseText):
    """更新用户记忆"""
    responseText = responseText.split("My updated self-introduction:")[-1].strip()

    # ✅ 使用config中的路径
    with open(f"{MEMORY_BASE_DIR}/user/user.{userId}", "w", encoding="utf-8") as file:
        file.write(responseText)

    with open(f"{MEMORY_BASE_DIR}/user-long/user.{userId}", "a", encoding="utf-8") as file:
        file.write("\n=====\n")
        file.write(responseText)


def update_item_memory(pos_itemId, neg_itemId, responseText, update_neg=True):
    """更新物品记忆"""
    updated_pos_item_intro = responseText.split("The updated description of the second item is: ")[-1]

    # ✅ 使用config中的路径
    with open(f"{MEMORY_BASE_DIR}/item/item.{pos_itemId}", "w", encoding="utf-8") as file:
        file.write(updated_pos_item_intro)

    if update_neg:
        updated_neg_item_intro = \
        re.split(r"The updated description of the first item is: |The updated description of the second item is: ",
                 responseText)[1]
        with open(f"{MEMORY_BASE_DIR}/item/item.{neg_itemId}", "w", encoding="utf-8") as file:
            file.write(updated_neg_item_intro)


if __name__ == "__main__":
    print(f"开始训练 - {exp_name}")
    print(f"使用固定负样本: {'是' if USE_FIXED_NEGATIVES else '否'}")
    print(f"随机种子: {random_seed}")

    random.seed(random_seed)

    interDF = createInterDF(train_file)
    itemDF = createItemDF(item_file)
    random_df = createRandomDF(random_file)

    print(f"训练数据: {len(interDF)} 条交互")

    initialize_memory()
    # process_interaction(interDF, itemDF, random_df)
    asyncio.run(process_interaction(interDF, itemDF, random_df))

    print("\n训练完成！")

# def create_prompts(user_description, list_of_item_description, pos_item_title,
#                    neg_item_title, system_reason, is_choice_right):
#     """创建更新提示"""
#     if not is_choice_right:
#         user_prompt = user_prompt_system_role(user_description) + '\n' + \
#                      user_prompt_template(list_of_item_description, pos_item_title,
#                                         neg_item_title, system_reason)
#         item_prompt = item_prompt_template(user_description, list_of_item_description,
#                                           pos_item_title, neg_item_title, system_reason)
#     else:
#         user_prompt = user_prompt_system_role(user_description) + '\n' + \
#                      user_prompt_template_true(list_of_item_description, pos_item_title,
#                                               neg_item_title, system_reason)
#         item_prompt = item_prompt_template_true(user_description, list_of_item_description,
#                                                pos_item_title, neg_item_title)
#     return user_prompt, item_prompt


# async def process_single_interaction_async(interaction, batch_idx, round_num, itemDF, random_df,
#                                          memory_lock, negative_samples_log, used_negatives, fixed_negatives):
#     """异步处理单个交互"""
#     try:
#         pos_itemId = str(interaction["item_id:token"]).strip()
#         userId = str(interaction["user_id:token"]).strip()
#
#         # ✅ 使用固定负样本
#         neg_itemId = get_neg_item_id(userId, pos_itemId, random_df, used_negatives, round_num, fixed_negatives)
#
#         if neg_itemId:
#             used_negatives.add(neg_itemId)
#
#         # ✅ 使用config中的路径
#         with memory_lock:
#             with open(f"{MEMORY_BASE_DIR}/user/user.{userId}", "r", encoding="utf-8") as file:
#                 user_memory = file.read()
#             with open(f"{MEMORY_BASE_DIR}/item/item.{pos_itemId}", "r", encoding="utf-8") as file:
#                 pos_item_memory = file.read()
#             with open(f"{MEMORY_BASE_DIR}/item/item.{neg_itemId}", "r", encoding="utf-8") as file:
#                 neg_item_memory = file.read()
#
#         pos_item_row = itemDF[itemDF["item_id:token"] == pos_itemId]
#         pos_item_title = str(pos_item_row["title:token_seq"].values[0]) if len(pos_item_row) > 0 else f"Item {pos_itemId}"
#
#         neg_item_row = itemDF[itemDF["item_id:token"] == neg_itemId]
#         neg_item_title = str(neg_item_row["title:token_seq"].values[0]) if len(neg_item_row) > 0 else f"Item {neg_itemId}"
#
#         if save_negative_samples:
#             interaction_key = f"user_{userId}_pos_{pos_itemId}_round_{round_num}_batch_{batch_idx}"
#             negative_samples_log[interaction_key] = {
#                 "user_id": userId,
#                 "pos_item_id": pos_itemId,
#                 "pos_item_title": pos_item_title,
#                 "neg_item_id": neg_itemId,
#                 "round_number": round_num,
#                 "batch_index": batch_idx
#             }
#
#         user_description = user_memory
#         list_of_item_description = f"title:{neg_item_title.strip()}. description:{neg_item_memory.strip()}\ntitle:{pos_item_title}. description:{pos_item_memory.strip()}"
#         system_prompt = system_prompt_template(user_description, list_of_item_description)
#
#         responseText = await async_client.call_api_async(system_prompt, model)
#         if not responseText:
#             return
#
#         selected_item_title, system_reason = parse_response(responseText)
#
#         pos_similarity = fuzz.ratio(selected_item_title.lower(), pos_item_title.lower())
#         neg_similarity = fuzz.ratio(selected_item_title.lower(), neg_item_title.lower())
#         is_choice_right = pos_similarity > neg_similarity
#
#
#         user_prompt, item_prompt = create_prompts(user_description, list_of_item_description,
#                                                  pos_item_title, neg_item_title,
#                                                  system_reason, is_choice_right)
#
#         user_response = await async_client.call_api_async(user_prompt, model)
#         if user_response:
#             update_user_memory(userId, user_response)
#
#         item_response = await async_client.call_api_async(item_prompt, model)
#         if item_response:
#             update_item_memory(pos_itemId, neg_itemId, item_response, update_neg=update_negative_samples)
#
#         print(f"✅ 用户 {userId} 第{round_num+1}轮完成")
#
#     except Exception as e:
#         print(f"❌ 处理交互时出错: {e}")
