# Переменные окружения (передаются при запуске, через .env или -e):
#   TELEGRAM_TOKEN     — обязательно, токен бота от @BotFather
#   ANTHROPIC_API_KEY  — обязательно, ключ Claude API
#   SUMMARY_TIME       — необязательно, время саммари для новых чатов (по умолчанию 21:00)
#   TZ_NAME            — необязательно, часовой пояс для всех чатов (по умолчанию Europe/Moscow)
#   MODEL              — необязательно, модель (по умолчанию claude-sonnet-5-5)
#
# Пример: docker run -d -e TELEGRAM_TOKEN=... -e ANTHROPIC_API_KEY=... -v gopbot-data:/data gopbot

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
