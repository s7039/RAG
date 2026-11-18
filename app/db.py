"""PostgreSQL 连接与会话。

``create_engine`` 是惰性的：模块导入时不会真正连接数据库，所以没起 db 容器时
其他接口和测试照样能用；只有真正读写待办 / 会话时才会连。
"""

from __future__ import annotations

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.config import get_settings

_settings = get_settings()
engine = create_engine(
    _settings.postgres_dsn,
    pool_pre_ping=True,
    future=True,
    # 本地没起 db 时不要无限等：连接 3s 超时，Agent 会话日志直接降级
    connect_args={"connect_timeout": 3},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def healthcheck() -> bool:
    with engine.connect() as conn:
        conn.execute(text("select 1"))
    return True
