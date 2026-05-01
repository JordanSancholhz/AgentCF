# -*- coding: utf-8 -*-
"""
统一评估脚本 - Multi-modal AgentCF
✅ 实时记录LLM输入输出（jsonl）
✅ 函数级重试 + 批次级重试（轻量）
✅ 优化模糊匹配策略（支持部分标题匹配）
✅ 统计完整匹配vs部分匹配
"""

import math
import os
import random
import numpy as np
import pickle
import json
import asyncio
import re
from tqdm import tqdm
from fuzzywuzzy import fuzz
from datetime import datetime

# ✅ 从config导入所有配置
from config import (
    EVAL_MODE, eval_config, eval_method_name,
    candidate_num, model, test_file, item_file, random_file,
    random_seed, save_ranking_results,
    USE_FIXED_NEGATIVES, EVAL_CANDIDATES_FILE,
    async_evaluation_batch_size, LOG_DIR
)
from dataPrepare import createInterDF, createItemDF, createRandomDF
from prompt import system_prompt_template_evaluation_basic
from request1 import async_client

# ✅ 创建日志目录
LLM_LOG_DIR = f"{LOG_DIR}/llm_interactions"
os.makedirs(LLM_LOG_DIR, exist_ok=True)

# ✅ 全局日志文件
SUCCESS_LOG_FILE = None
FAILURE_LOG_FILE = None

# ============= 通用工具函数 =============

def calculate_dcg(relevance_scores, k):
    """计算DCG@k"""
    dcg = 0.0
    for i in range(min(k, len(relevance_scores))):
        dcg += relevance_scores[i] / math.log2(i + 2)
    return dcg

def calculate_ndcg(relevance_scores, k):
    """计算NDCG@k"""
    dcg_k = calculate_dcg(relevance_scores, k)
    sorted_scores = sorted(relevance_scores, reverse=True)
    idcg_k = calculate_dcg(sorted_scores[:k], k)
    return dcg_k / idcg_k if idcg_k != 0 else 0.0

def load_fixed_eval_candidates():
    """加载预生成的固定评估候选集"""
    if not USE_FIXED_NEGATIVES:
        return None
    
    if not os.path.exists(EVAL_CANDIDATES_FILE):
        print(f"[ERROR] 固定候选集文件不存在: {EVAL_CANDIDATES_FILE}")
        print(f"请先运行: python negative_sampler.py --seed {random_seed} --verify")
        exit(1)
    
    with open(EVAL_CANDIDATES_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"[INFO] 已加载固定评估候选集: {EVAL_CANDIDATES_FILE}")
    print(f"[INFO] 用户数: {data['metadata']['total_users']}")
    
    return data['candidates']

# ============= Embedding相关函数 =============

def cosine_similarity(vec1, vec2):
    """计算余弦相似度"""
    vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)
    vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-8)
    return np.dot(vec1_norm, vec2_norm)

def load_embeddings():
    """加载用户和物品embedding"""
    if not eval_config.get('use_embedding', False):
        return None, None
    
    embedding_dir = eval_config['embedding_dir']
    user_embeddings_path = f"{embedding_dir}/user_embeddings_gme.pkl"
    item_embeddings_path = f"{embedding_dir}/item_embeddings_gme.pkl"
    
    try:
        with open(user_embeddings_path, 'rb') as f:
            user_embeddings = pickle.load(f)
        with open(item_embeddings_path, 'rb') as f:
            item_embeddings = pickle.load(f)
        
        print(f"[INFO] Embedding加载成功: {len(user_embeddings)} 用户, {len(item_embeddings)} 物品")
        return user_embeddings, item_embeddings
    except Exception as e:
        print(f"[ERROR] Embedding加载失败: {e}")
        return None, None

def compute_embedding_ranking(user_id, candidate_list, user_embeddings, item_embeddings):
    """计算基于embedding的排名"""
    if user_id not in user_embeddings:
        return None
    
    for item_id in candidate_list:
        if item_id not in item_embeddings:
            return None
    
    user_emb = user_embeddings[user_id]['embedding']
    similarities = []
    
    for item_id in candidate_list:
        item_emb = item_embeddings[item_id]['embedding']
        sim = cosine_similarity(user_emb, item_emb)
        similarities.append((item_id, sim))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities

# ============= 实时日志 =============

def log_llm_interaction(user_id, llm_input, llm_output, success, failure_reason=None, matched_items=None, unmatched_lines=None):
    """记录LLM交互（实时写入）"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    
    log_entry = {
        "user_id": user_id,
        "timestamp": timestamp,
        "llm_input": llm_input,
        "llm_output": llm_output,
        "success": success,
        "failure_reason": failure_reason,
        "matched_items": matched_items,
        "unmatched_lines": unmatched_lines
    }
    
    log_file = SUCCESS_LOG_FILE if success else FAILURE_LOG_FILE
    
    if log_file:
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"[WARNING] 写入日志失败: {e}")
    
    if not success:
        print(f"[FAILURE] 用户{user_id} | {failure_reason}")

# ============= LLM相关函数（优化后的模糊匹配）=============

async def get_llm_ranking_async(user_id, candidate_list, cdt_item_title_list, itemDF, memory_dir):
    """
    异步获取LLM排名（简化+完整日志+匹配统计）
    ✅ 使用used_indices防止重复匹配（不替换）
    ✅ 2次重试，阈值20
    ✅ 接受≥50%结果
    ✅ 明确区分完整匹配vs部分匹配
    """
    llm_input = None
    llm_output = None
    
    try:
        # 读取用户记忆
        with open(f"{memory_dir}/user/user.{user_id}", "r", encoding="utf-8") as f:
            user_memory = f.read()
        
        # 读取候选物品描述
        cdt_item_memory_list = []
        for item_id in candidate_list:
            try:
                with open(f"{memory_dir}/item/item.{item_id}", "r", encoding="utf-8") as f:
                    cdt_item_memory_list.append(f.read())
            except:
                cdt_item_memory_list.append("No description")
        
        # 构建prompt
        example_list = ''
        for title, memory in zip(cdt_item_title_list, cdt_item_memory_list):
            example_list += f"title:{title.strip()}. description:{memory.strip()}\n"
        
        system_prompt = system_prompt_template_evaluation_basic(user_memory, candidate_num, example_list)
        llm_input = system_prompt
        
        # ✅ 简化重试：2次，宽松阈值20
        best_ranked_items = []
        last_matched_titles = []
        last_unmatched_lines = []
        
        for retry_idx in range(5):  # 只重试2次
            llm_output = await async_client.call_api_async(system_prompt, model)
            if llm_output is None:
                await asyncio.sleep(0.2)
                continue
            
            # 清理输出
            cleaned_output = llm_output.strip()
            cleaned_output = re.sub(r"\n+", "\n", cleaned_output)
            
            if 'Rank:' not in cleaned_output:
                await asyncio.sleep(0.2)
                continue
            
            # ✅ 提取排名部分
            ans_begin = cleaned_output.index('Rank:') + len('Rank:')
            lines = cleaned_output[ans_begin:].strip().split('\n')
            
            ranked_items = []
            matched_titles = []
            unmatched_lines = []
            used_indices = set()  # ✅ 防止重复匹配（关键！）
            
            # ✅ 逐行匹配
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # 提取排名和标题
                match = re.match(r'^(\d+)\.\s*(.+)$', line)
                if not match:
                    continue
                
                temp_title = match.group(2).strip()
                temp_title = re.sub(r'\*\*', '', temp_title).strip()  # 移除markdown
                
                # ✅ 宽松匹配：阈值20
                best_score = 0
                best_idx = -1
                
                for i, title in enumerate(cdt_item_title_list):
                    if i in used_indices:  # ✅ 跳过已使用的候选项
                        continue
                    
                    # 两种相似度取最大
                    sim = max(
                        fuzz.ratio(temp_title.lower(), title.lower()),
                        fuzz.partial_ratio(temp_title.lower(), title.lower())
                    )
                    
                    if sim > best_score:
                        best_score = sim
                        best_idx = i
                
                # ✅ 阈值20，非常宽松
                if best_score > 20 and best_idx >= 0: # 原本是20
                    item_id = candidate_list[best_idx]
                    ranked_items.append(item_id)
                    used_indices.add(best_idx)  # ✅ 标记为已使用
                    matched_titles.append(f"{temp_title[:40]} → {cdt_item_title_list[best_idx][:40]} (sim={best_score})")
                else:
                    unmatched_lines.append(f"{temp_title[:40]} (best_sim={best_score})")
                
                # ✅ 达到目标数量就停止
                if len(ranked_items) >= candidate_num:
                    break
            
            # 保存最佳结果
            if len(ranked_items) > len(best_ranked_items):
                best_ranked_items = ranked_items
                last_matched_titles = matched_titles
                last_unmatched_lines = unmatched_lines
            
            # ✅ 成功：匹配到完整数量 → 记录到成功日志
            if len(ranked_items) >= candidate_num:
                match_info = f"完整匹配：{len(ranked_items)}/{candidate_num}"
                log_llm_interaction(user_id, llm_input, llm_output, True, match_info, matched_titles, None)
                return ranked_items
            
            await asyncio.sleep(0.1)
        
        # ✅ 接受部分结果：>=50% 或 >=5个
        min_acceptable = max(5, candidate_num // 2)
        
        if len(best_ranked_items) >= min_acceptable:
            reason = f"部分匹配（接受）：{len(best_ranked_items)}/{candidate_num}"
            # ✅ 记录到成功日志（因为可以计算NDCG）
            log_llm_interaction(user_id, llm_input, llm_output, True, reason, last_matched_titles, last_unmatched_lines)
            return best_ranked_items  # ✅ 返回部分结果用于计算NDCG
        else:
            reason = f"匹配不足：{len(best_ranked_items)}/{candidate_num}（低于{min_acceptable}）"
            # ✅ 记录到失败日志
            log_llm_interaction(user_id, llm_input, llm_output, False, reason, last_matched_titles, last_unmatched_lines)
            return None
    
    except Exception as e:
        # ✅ 异常也记录到失败日志
        log_llm_interaction(user_id, llm_input, llm_output, False, f"异常: {type(e).__name__}", None, None)
        return None

# ============= 融合函数（仅RRF）=============

def rrf_fusion(embedding_ranking, llm_ranking, candidate_list, rrf_k=60):
    """RRF融合"""
    emb_rank_dict = {item_id: rank + 1 for rank, (item_id, _) in enumerate(embedding_ranking)}
    
    llm_rank_dict = {}
    if llm_ranking:
        for rank, item_id in enumerate(llm_ranking):
            llm_rank_dict[item_id] = rank + 1
    
    rrf_scores = []
    for item_id in candidate_list:
        emb_rank = emb_rank_dict.get(item_id, len(candidate_list) + 1)
        llm_rank = llm_rank_dict.get(item_id, len(candidate_list) + 1)
        
        rrf_score = (1.0 / (rrf_k + emb_rank)) + (1.0 / (rrf_k + llm_rank))
        rrf_scores.append((item_id, rrf_score))
    
    rrf_scores.sort(key=lambda x: x[1], reverse=True)
    return rrf_scores

# ============= 评估函数 =============

async def evaluate_single_user(record, itemDF, random_df, fixed_candidates, user_embeddings, item_embeddings):
    """评估单个用户（支持四种模式）"""
    user_id = str(record["user_id:token"]).strip()
    target_item_id = str(record["item_id:token"]).strip()
    
    # 1️⃣ 获取候选集（优先使用固定候选集）
    if fixed_candidates and user_id in fixed_candidates:
        candidate_data = fixed_candidates[user_id]
        candidate_list = candidate_data['candidates']
        assert candidate_data['target'] == target_item_id, f"Target mismatch for user {user_id}"
    else:
        # 回退到动态生成
        user_row = random_df[random_df['user_id'] == int(user_id)]
        if len(user_row) == 0:
            return None
        
        candidates = user_row['candidates'].values[0]
        valid_candidates = [c for c in candidates if c != target_item_id]
        
        if len(valid_candidates) < (candidate_num - 1):
            return None
        
        neg_samples = random.sample(valid_candidates, candidate_num - 1)
        candidate_list = neg_samples + [target_item_id]
        random.shuffle(candidate_list)
    
    # 2️⃣ 获取候选物品标题
    cdt_item_title_list = []
    for item_id in candidate_list:
        item_row = itemDF[itemDF["item_id:token"] == item_id]
        if len(item_row) > 0:
            cdt_item_title_list.append(str(item_row["title:token_seq"].values[0]))
        else:
            cdt_item_title_list.append(f"Item {item_id}")
    
    # 3️⃣ 根据模式计算排名
    if EVAL_MODE == "embedding":
        embedding_ranking = compute_embedding_ranking(user_id, candidate_list, user_embeddings, item_embeddings)
        if embedding_ranking is None:
            return None
        
        final_ranking = embedding_ranking
        has_llm = False
    
    elif EVAL_MODE in ["basic", "description"]:
        memory_dir = eval_config['memory_dir']
        llm_ranking = await get_llm_ranking_async(user_id, candidate_list, cdt_item_title_list, itemDF, memory_dir)
        if not llm_ranking:
            return None
        final_ranking = [(item_id, 1.0 / (rank + 1)) for rank, item_id in enumerate(llm_ranking)]
        has_llm = True
    
    elif EVAL_MODE == "rrf":
        embedding_ranking = compute_embedding_ranking(user_id, candidate_list, user_embeddings, item_embeddings)
        if embedding_ranking is None:
            return None
        memory_dir = eval_config['memory_dir']
        llm_ranking = await get_llm_ranking_async(user_id, candidate_list, cdt_item_title_list, itemDF, memory_dir)
        rrf_k = eval_config.get('rrf_k', 60)
        final_ranking = rrf_fusion(embedding_ranking, llm_ranking, candidate_list, rrf_k)
        has_llm = llm_ranking is not None
    
    else:
        return None
    
    # 4️⃣ 计算指标
    ranked_items = [item[0] for item in final_ranking]
    target_rank = ranked_items.index(target_item_id) + 1
    
    relevance_scores = [1 if item == target_item_id else 0 for item in ranked_items]
    
    ndcg_10 = calculate_ndcg(relevance_scores, 10)
    ndcg_5 = calculate_ndcg(relevance_scores, 5)
    ndcg_1 = calculate_ndcg(relevance_scores, 1)
    mrr = 1.0 / target_rank
    
    return {
        "user_id": user_id,
        "target_item_id": target_item_id,
        "target_rank": target_rank,
        "ndcg_10": ndcg_10,
        "ndcg_5": ndcg_5,
        "ndcg_1": ndcg_1,
        "mrr": mrr,
        "has_llm": has_llm,
        "final_ranking": ranked_items
    }

async def process_batch(batch, itemDF, random_df, fixed_candidates, user_embeddings, item_embeddings):
    """批次处理（异步并发）"""
    tasks = [
        evaluate_single_user(record, itemDF, random_df, fixed_candidates, user_embeddings, item_embeddings)
        for _, record in batch.iterrows()
    ]
    return await asyncio.gather(*tasks, return_exceptions=True)

# ============= 主函数 =============

async def main_async():
    global SUCCESS_LOG_FILE, FAILURE_LOG_FILE
    
    print("=" * 60)
    print("Multi-modal AgentCF - 统一评估")
    print("=" * 60)
    print(f"评估模式: {EVAL_MODE}")
    print(f"评估方法: {eval_method_name}")
    print(f"使用固定候选集: {'是' if USE_FIXED_NEGATIVES else '否'}")
    
    if EVAL_MODE == "rrf":
        print(f"RRF参数: k={eval_config.get('rrf_k', 60)}")
    
    # ✅ 初始化日志文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    SUCCESS_LOG_FILE = f"{LLM_LOG_DIR}/success_{timestamp}.jsonl"
    FAILURE_LOG_FILE = f"{LLM_LOG_DIR}/failure_{timestamp}.jsonl"
    
    print(f"[LOG] 成功日志: {SUCCESS_LOG_FILE}")
    print(f"[LOG] 失败日志: {FAILURE_LOG_FILE}")
    
    # 写入起始元数据
    for log_file, log_type in [(SUCCESS_LOG_FILE, "success"), (FAILURE_LOG_FILE, "failure")]:
        with open(log_file, 'w', encoding='utf-8') as f:
            metadata = {
                "type": "metadata",
                "log_type": log_type,
                "eval_mode": EVAL_MODE,
                "model": model,
                "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "batch_size": async_evaluation_batch_size
            }
            f.write(json.dumps(metadata, ensure_ascii=False) + '\n')
    
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # 加载固定候选集
    fixed_candidates = load_fixed_eval_candidates()
    
    # 加载embedding（如果需要）
    user_embeddings, item_embeddings = None, None
    if eval_config.get('use_embedding', False):
        user_embeddings, item_embeddings = load_embeddings()
        if not user_embeddings or not item_embeddings:
            print("[ERROR] Embedding加载失败，无法继续")
            return
    
    # 加载数据
    print("\n[INFO] 加载数据...")
    interDF = createInterDF(test_file)
    itemDF = createItemDF(item_file)
    random_df = createRandomDF(random_file)
    print(f"[INFO] 测试数据: {len(interDF)} 条")
    
    # 批次评估
    results = {"ndcg_10": [], "ndcg_5": [], "ndcg_1": [], "mrr": []}
    evaluation_log = {}
    skipped_count = 0
    llm_success_count = 0
    
    total_batches = (len(interDF) + async_evaluation_batch_size - 1) // async_evaluation_batch_size
    
    print(f"\n[INFO] 开始评估（批次大小={async_evaluation_batch_size}）...")
    print(f"[INFO] LLM交互将实时记录到日志文件\n")
    
    for batch_idx in tqdm(range(total_batches), desc="进度"):
        start_idx = batch_idx * async_evaluation_batch_size
        end_idx = min((batch_idx + 1) * async_evaluation_batch_size, len(interDF))
        batch = interDF.iloc[start_idx:end_idx]
        
        batch_results = await process_batch(batch, itemDF, random_df, fixed_candidates, user_embeddings, item_embeddings)
        
        failed_records = []
        for idx, result in enumerate(batch_results):
            if isinstance(result, Exception) or result is None:
                failed_records.append(batch.iloc[idx])
            else:
                results["ndcg_10"].append(result["ndcg_10"])
                results["ndcg_5"].append(result["ndcg_5"])
                results["ndcg_1"].append(result["ndcg_1"])
                results["mrr"].append(result["mrr"])
                
                if result.get("has_llm", False):
                    llm_success_count += 1
                
                evaluation_log[f"user_{result['user_id']}"] = result
        
        # ✅ 批次级重试（最多3轮，轻量）
        retry_count = 0
        while failed_records and retry_count < 20:
            retry_count += 1
            print(f"\n[WARN] 批次{batch_idx+1}有{len(failed_records)}个失败，第{retry_count}次重试...")
            await asyncio.sleep(0.5)
            
            retry_batch = failed_records
            failed_records = []
            
            for record in retry_batch:
                result = await evaluate_single_user(record, itemDF, random_df, fixed_candidates, user_embeddings, item_embeddings)
                
                if isinstance(result, Exception) or result is None:
                    failed_records.append(record)
                else:
                    results["ndcg_10"].append(result["ndcg_10"])
                    results["ndcg_5"].append(result["ndcg_5"])
                    results["ndcg_1"].append(result["ndcg_1"])
                    results["mrr"].append(result["mrr"])
                    
                    if result.get("has_llm", False):
                        llm_success_count += 1
                    
                    evaluation_log[f"user_{result['user_id']}"] = result
        
        if failed_records:
            skipped_count += len(failed_records)
            print(f"[WARN] 批次{batch_idx+1}最终有{len(failed_records)}个用户失败")
    
    # 写入结束元数据
    for log_file in [SUCCESS_LOG_FILE, FAILURE_LOG_FILE]:
        with open(log_file, 'a', encoding='utf-8') as f:
            end_metadata = {
                "type": "end_metadata",
                "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            f.write(json.dumps(end_metadata, ensure_ascii=False) + '\n')
    
    # ✅ 统计匹配类型（读取成功日志）
    full_match_count = 0
    partial_match_count = 0
    
    if os.path.exists(SUCCESS_LOG_FILE) and EVAL_MODE in ["basic", "description", "rrf"]:
        try:
            with open(SUCCESS_LOG_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        if entry.get("type") == "metadata" or entry.get("type") == "end_metadata":
                            continue
                        
                        reason = entry.get("failure_reason", "")
                        if reason and "完整匹配" in reason:
                            full_match_count += 1
                        elif reason and "部分匹配" in reason:
                            partial_match_count += 1
                    except:
                        continue
        except Exception as e:
            print(f"[WARNING] 统计匹配类型失败: {e}")
    
    # 输出结果
    processed_count = len(results["ndcg_10"])
    
    print(f"\n" + "=" * 60)
    print("评估结果:")
    print("=" * 60)
    
    if processed_count > 0:
        print(f"成功: {processed_count}, 跳过: {skipped_count}")
        
        if EVAL_MODE in ["basic", "description", "rrf"]:
            print(f"LLM成功率: {llm_success_count}/{processed_count} ({llm_success_count/processed_count*100:.1f}%)")
            # ✅ 新增：匹配类型统计
            if full_match_count > 0 or partial_match_count > 0:
                print(f"\n匹配类型分布:")
                print(f"  完整匹配: {full_match_count} ({full_match_count/llm_success_count*100:.1f}%)")
                print(f"  部分匹配: {partial_match_count} ({partial_match_count/llm_success_count*100:.1f}%)")
        
        print()
        print(f"NDCG@1:  {np.mean(results['ndcg_1']):.4f}")
        print(f"NDCG@5:  {np.mean(results['ndcg_5']):.4f}")
        print(f"NDCG@10: {np.mean(results['ndcg_10']):.4f}")
        print(f"MRR:     {np.mean(results['mrr']):.4f}")
        
        # 保存结果
        os.makedirs(LOG_DIR, exist_ok=True)
        
        if save_ranking_results:
            log_file = f"{LOG_DIR}/evaluation_{eval_method_name}.json"
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(evaluation_log, f, ensure_ascii=False, indent=2)
            print(f"\n[INFO] 详细日志: {log_file}")
        
        summary = {
            "eval_method": eval_method_name,
            "eval_mode": EVAL_MODE,
            "statistics": {
                "processed": processed_count,
                "skipped": skipped_count,
                "llm_success": llm_success_count if EVAL_MODE != "embedding" else "N/A",
                "full_match": full_match_count if EVAL_MODE in ["basic", "description", "rrf"] else "N/A",
                "partial_match": partial_match_count if EVAL_MODE in ["basic", "description", "rrf"] else "N/A"
            },
            "metrics": {
                "NDCG@1": float(np.mean(results['ndcg_1'])),
                "NDCG@5": float(np.mean(results['ndcg_5'])),
                "NDCG@10": float(np.mean(results['ndcg_10'])),
                "MRR": float(np.mean(results['mrr']))
            }
        }
        
        summary_file = f"{LOG_DIR}/summary_{eval_method_name}.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[INFO] 汇总结果: {summary_file}")
    else:
        print("[ERROR] 无有效结果")
    
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main_async())
