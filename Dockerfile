FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 先装依赖，利用层缓存
COPY requirements.txt .
# 国内直连 pypi.org 实测只有 ~20KB/s，默认走清华镜像；
# 需要官方源时：docker compose build --build-arg PIP_INDEX_URL=https://pypi.org/simple
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install -r requirements.txt

COPY app ./app
COPY scripts ./scripts
COPY input ./input
COPY run.py pytest.ini ./

# 非 root 运行；data 目录要预先建好并授权，否则挂 volume 后写不进去
RUN useradd -m -u 1000 appuser \
    && mkdir -p /app/data \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
