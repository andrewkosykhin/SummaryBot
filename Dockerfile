# Переменные окружения (передаются при запуске, через --env-file .env или -e):
#   TELEGRAM_TOKEN     — обязательно, токен бота от @BotFather
#   ANTHROPIC_API_KEY  — обязательно, ключ Claude API
#   OPENAI_API_KEY     — необязательно, для расшифровки голосовых и кружочков
#   SUMMARY_TIME       — необязательно, время саммари для новых чатов (по умолчанию 21:00)
#   TZ_NAME            — необязательно, часовой пояс для всех чатов (по умолчанию Europe/Moscow)
#   MODEL              — необязательно, модель саммари (по умолчанию claude-sonnet-5-5)
#   VISION_MODEL       — необязательно, модель для картинок (по умолчанию как MODEL)
#   FAST_MODEL         — необязательно, модель реакций на голосовые (по умолчанию claude-haiku-4-5)
#   STT_MODEL          — необязательно, модель расшифровки (по умолчанию gpt-4o-transcribe)
#   MEMORY_DETAIL      — необязательно, подробность памяти: short | normal | full (по умолчанию normal)
#
# Пример: docker run -d --env-file .env -v "$PWD/data:/data" summarybot
# (папка data должна принадлежать uid 1000 — от него работает бот)

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DB_PATH=/data/messages.db

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY bot.py .

RUN useradd --create-home bot && mkdir -p /data && chown bot:bot /data
USER bot

VOLUME ["/data"]

CMD ["python", "bot.py"]
