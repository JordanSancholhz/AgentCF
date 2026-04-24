# -*- coding: utf-8 -*-
import asyncio
import random
import time
from typing import Optional, Dict, Any
import httpx
import json

class AsyncChatGLMClient:
    def __init__(self, api_key: str, base_url: str = "https://open.bigmodel.cn/api/paas/v4", max_concurrent: int = 20, proxy: str = None):
        """
        ChatGLM异步客户端

        参数:
            api_key: ChatGLM API密钥
            base_url: ChatGLM API基础URL
            max_concurrent: 最大并发数
            proxy: 代理地址（可选）
        """
        self.api_key = api_key
        self.base_url = base_url
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)

        # 配置HTTP客户端
        client_kwargs = {
            "timeout": httpx.Timeout(120.0, connect=10.0),
            "headers": {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
        }

        if proxy:
            client_kwargs["proxy"] = proxy

        self.http_client = httpx.AsyncClient(**client_kwargs)

    async def call_api_with_metrics(self, prompt: str, model: str = "glm-4-flash", max_retries: int = 3) -> Dict[str, Any]:
        """
        异步调用ChatGLM API，并返回时延统计信息

        返回字段：
        - content: 成功时的文本，失败时为None
        - latency_ms: 本次成功/最终失败的总耗时（毫秒）
        - attempts: 实际尝试次数
        - success: 是否成功
        - error: 最终错误信息
        """
        async with self.semaphore:
            overall_start = time.perf_counter()
            last_error = None

            for attempt in range(max_retries):
                try:
                    # 构造ChatGLM API请求
                    payload = {
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 2000,
                        "temperature": 0.6,
                    }

                    response = await self.http_client.post(
                        f"{self.base_url}/chat/completions",
                        json=payload
                    )

                    # 检查HTTP状态码
                    if response.status_code != 200:
                        last_error = f"HTTP {response.status_code}: {response.text}"
                        print(f"[API ERROR] {last_error}")
                        raise Exception(last_error)

                    # 解析响应
                    result = response.json()

                    if "choices" in result and len(result["choices"]) > 0:
                        content = result["choices"][0]["message"]["content"]
                        if content and content.strip():
                            return {
                                "content": content.strip(),
                                "latency_ms": round((time.perf_counter() - overall_start) * 1000, 3),
                                "attempts": attempt + 1,
                                "success": True,
                                "error": None,
                            }

                    last_error = "ChatGLM返回空内容（choices为空或content为空）"
                    print(f"[API WARN] {last_error}")

                except Exception as e:
                    last_error = f"{type(e).__name__}: {str(e)}"
                    print(f"[API ERROR] 尝试 {attempt+1}/{max_retries}: {last_error}")

                # 退避重试（1s, 2s, 4s, 8s, 16s）+ 抖动，最后一次不等
                if attempt < max_retries - 1:
                    base = 2 ** attempt
                    jitter = random.uniform(0, 0.5)
                    wait_time = base + jitter
                    print(f"   ⏳ 等待 {wait_time:.1f}秒 后重试...")
                    await asyncio.sleep(wait_time)

            return {
                "content": None,
                "latency_ms": round((time.perf_counter() - overall_start) * 1000, 3),
                "attempts": max_retries,
                "success": False,
                "error": last_error,
            }

    async def call_api_async(self, prompt: str, model: str = "glm-4-flash", max_retries: int = 3) -> Optional[str]:
        """
        向后兼容接口：仅返回字符串结果
        """
        result = await self.call_api_with_metrics(prompt, model, max_retries=max_retries)
        return result["content"]

    async def close(self):
        """关闭HTTP客户端"""
        await self.http_client.aclose()

# ✅ 全局异步客户端实例
# 使用说明：
# 1. 将YOUR_CHATGLM_API_KEY替换为你的ChatGLM API密钥
# 2. model参数可选：glm-4, glm-4-plus, glm-4-flash等
# 3. 如需代理，取消注释proxy参数
async_client = AsyncChatGLMClient(
    api_key="YOUR_API_KEY",  # ⚠️ 请替换为你的ChatGLM API密钥
    base_url="https://open.bigmodel.cn/api/paas/v4",
    max_concurrent=5,
    # proxy="http://127.0.0.1:7897"  # 如需代理，取消注释
)