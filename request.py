# -*- coding: utf-8 -*-
import asyncio
import random
import time
from typing import Optional, Dict, Any
from openai import AsyncOpenAI
import httpx

class AsyncOpenAIClient:
    def __init__(self, api_key: str, max_concurrent: int = 20, proxy: str = None):
        self.api_key = api_key
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        # ✅ 支持代理配置
        if proxy:
            http_client = httpx.AsyncClient(
                proxy=proxy,
                timeout=120.0  # ✅ 增加到120秒
            )
            self.client = AsyncOpenAI(
                api_key=api_key,
                http_client=http_client
            )
        else:
            self.client = AsyncOpenAI(api_key=api_key)

    async def call_api_with_metrics(self, prompt: str, model: str, max_retries: int = 3) -> Dict[str, Any]:
        """
        异步调用OpenAI ChatGPT API，并返回时延统计信息
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
                    response = await self.client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=2000,
                        temperature=0.6,
                        timeout=120,  # ✅ 增加到120秒
                    )

                    if response and response.choices and len(response.choices) > 0:
                        content = response.choices[0].message.content
                        if content and content.strip():
                            return {
                                "content": content.strip(),
                                "latency_ms": round((time.perf_counter() - overall_start) * 1000, 3),
                                "attempts": attempt + 1,
                                "success": True,
                                "error": None,
                            }

                    last_error = "chat.completions 返回空内容（choices 为空或 content 为空）"
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

    async def call_api_async(self, prompt: str, model: str, max_retries: int = 3) -> Optional[str]:
        """
        向后兼容接口：仅返回字符串结果
        """
        result = await self.call_api_with_metrics(prompt, model, max_retries=max_retries)
        return result["content"]

# ✅ 全局异步客户端实例（配置代理）
async_client = AsyncOpenAIClient(
    api_key="YOUR_API_KEY",
    max_concurrent=5,  # ✅ 降低并发数
    proxy="http://127.0.0.1:7897"
)
