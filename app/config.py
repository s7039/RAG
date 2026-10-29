"""全局配置。

所有配置都可以通过环境变量或项目根目录下的 ``.env`` 文件覆盖，
环境变量名 = 字段名大写，例如 ``DEEPSEEK_API_KEY``、``CHUNK_SIZE``。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录：.../RAG
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """应用配置（pydantic-settings 会自动读取 .env）。"""

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---------- 应用 ----------
    app_name: str = "健康档案私有知识库 RAG"
    app_version: str = "1.0.0"
    api_prefix: str = "/api"
    cors_origins: str = "*"
    log_level: str = "INFO"

    # ---------- 数据路径 ----------
    # 输入文档目录（健康档案 PDF 放在这里）
    input_dir: str = "input"
    # 向量库持久化目录
    persist_dir: str = "data/chroma"
    collection_name: str = "health_records"

    # ---------- 切分参数 ----------
    chunk_size: int = 600
    chunk_overlap: int = 100
    min_chunk_size: int = 30

    # ---------- 检索参数 ----------
    # hybrid = 向量召回 + BM25 关键词召回融合；vector = 纯向量
    retrieve_mode: str = "hybrid"
    hybrid_alpha: float = 0.5
    # BM25 原始分数的软饱和尺度：越小越严格
    hybrid_kw_scale: float = 6.0
    top_k: int = 4
    fetch_k: int = 20
    use_mmr: bool = False
    mmr_lambda: float = 0.6
    score_threshold: float = 0.0

    # ---------- 生成参数 ----------
    max_context_chars: int = 6000
    llm_temperature: float = 0.2
    llm_max_tokens: int = 1200

    # ---------- 向量化（Embedding）----------
    # auto: 有 OPENAI_API_KEY 时用 OpenAI，否则用本地哈希向量
    embed_provider: str = "auto"
    embed_batch_size: int = 64
    local_embed_dim: int = 1024

    # ---------- OpenAI ----------
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    # ---------- DeepSeek ----------
    # DeepSeek 兼容 OpenAI 协议，但官方不提供 embedding 接口
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_chat_model: str = "deepseek-chat"

    # ---------- 任意 OpenAI 兼容服务（可选）----------
    # 例如通义千问 DashScope、Moonshot、vLLM / Ollama 暴露的兼容接口
    custom_name: str = "custom"
    custom_label: str = "自定义 OpenAI 兼容服务"
    custom_api_key: str = ""
    custom_base_url: str = ""
    custom_chat_model: str = ""
    custom_embedding_model: str = ""

    # ---------- 兜底 ----------
    default_provider: str = "deepseek"
    default_model: str = ""
    # 没有配置任何 Key 时，是否使用离线模拟大模型（方便本地演示）
    allow_mock_llm: bool = True

    # ---------- 路径工具 ----------
    @property
    def input_path(self) -> Path:
        return self._resolve(self.input_dir)

    @property
    def persist_path(self) -> Path:
        return self._resolve(self.persist_dir)

    def _resolve(self, value: str) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = BASE_DIR / path
        return path

    def ensure_dirs(self) -> None:
        self.input_path.mkdir(parents=True, exist_ok=True)
        self.persist_path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """读取配置（带缓存，避免重复解析 .env）。"""
    return Settings()
