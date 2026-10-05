"""
Телеграм-бот: собирает сообщения в группах/каналах и раз в день присылает
саммари. У каждого чата свой набор сообщений, свой промпт и своё время.

Установка:
    pip install "python-telegram-bot[job-queue]>=21" anthropic

Переменные окружения:
    TELEGRAM_TOKEN      — токен бота от @BotFather
    ANTHROPIC_API_KEY   — ключ Claude API (console.anthropic.com)
    SUMMARY_TIME        — время саммари по умолчанию для новых чатов (21:00)
    TZ_NAME             — часовой пояс для всех чатов (Europe/Moscow)
    MODEL               — модель, по умолчанию claude-sonnet-5-5
    DB_PATH             — путь к базе, по умолчанию messages.db
    VISION_MODEL        — модель для картинок, по умолчанию как MODEL
    FAST_MODEL          — модель для реакций на голосовые, по умолчанию claude-haiku-4-5
    OPENAI_API_KEY      — ключ OpenAI для расшифровки голосовых (без него не расшифровываем)
    STT_MODEL           — модель расшифровки, по умолчанию gpt-4o-transcribe
    MEMORY_DETAIL       — подробность памяти: short | normal | full (по умолчанию normal)
    PHOTO_BATCH_WAIT    — сек. тишины после последнего фото автора, после которых серия
                          фото разбирается одной пачкой (по умолчанию 10)
    MAX_BATCH_PHOTOS    — сколько фото из серии смотреть (по умолчанию 10); альбом
                          разбирается целиком
    MAX_LINKS           — сколько ссылок из сообщения открывать и описывать для саммари
                          (по умолчанию 3, 0 — не открывать)

ВАЖНО:
  * Группы: в @BotFather выполни /setprivacy -> Disable, иначе бот видит
    только команды. После этого удали и заново добавь бота в группу.
  * Каналы: бота нужно сделать админом канала.

Команды в чате (настройки у каждого чата свои, менять может любой участник):
    /menu                — меню: статистика, саммари на сейчас, настройки (время, промпт,
                           пересказ ссылок). В канале: /menu time 21:30, /menu prompt <текст>
    /summary             — саммари на сейчас (только показать: память и итоги дня не трогает)
    /memory              — что бот помнит о чате (итоги дней, недель, месяцев, лет)
    /retry [N]           — перераспознать последние N неудачных голосовых/картинок
                           (или ответь /retry на конкретное сообщение)
"""

import asyncio
import base64
import contextlib
import datetime
import io
import json
import logging
import os
import re
import sqlite3
from zoneinfo import ZoneInfo

import av
import numpy as np
import anthropic
import openai
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from telegram import (
    Bot,
    BotCommand,
    BotCommandScopeAllChatAdministrators,
    BotCommandScopeAllGroupChats,
    BotCommandScopeDefault,
    InlineKeyboardButton,
    ForceReply,
    InlineKeyboardMarkup,
    MessageEntity,
    Update,
)
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    ChatMemberHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

TOKEN = os.environ["TELEGRAM_TOKEN"]
DEFAULT_TIME = os.getenv("SUMMARY_TIME", "21:00")
TZ = ZoneInfo(os.getenv("TZ_NAME", "Europe/Moscow"))
MODEL = os.getenv("MODEL", "claude-sonnet-5-5")
VISION_MODEL = os.getenv("VISION_MODEL", MODEL)
FAST_MODEL = os.getenv("FAST_MODEL", "claude-haiku-4-5")
STT_MODEL = os.getenv("STT_MODEL", "gpt-4o-transcribe")
PHOTO_BATCH_WAIT = float(os.getenv("PHOTO_BATCH_WAIT", "10"))
MAX_BATCH_PHOTOS = max(1, int(os.getenv("MAX_BATCH_PHOTOS", "10")))
MAX_LINKS = max(0, int(os.getenv("MAX_LINKS", "3")))
DB_PATH = os.getenv("DB_PATH", "messages.db")

MAX_TRANSCRIPT_CHARS = 150_000  # чтобы не улететь в лимиты модели
MAX_PROMPT_CHARS = 4000
TG_MSG_LIMIT = 4000
TIME_RE = re.compile(r"^([01]?\d|2[0-3]):([0-5]\d)$")

DEFAULT_PROMPT = (
    "Ты — чёткий пацан с района, гопник. Тебе дают переписку из чата за день. "
    "Сделай саммари: кто о чём базарил, главные темы, срачи и угарные моменты. "
    "Пиши от лица гопника, сочно, с матом и дворовым сленгом, но по делу — "
    "чтобы было понятно, что реально обсуждали. Упоминай участников по именам. "
    "Без реальных оскорблений по национальности, внешности и т.п. — стебёшь, "
    "а не травишь. Объём — до 2000 символов."
)

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=logging.INFO
)
log = logging.getLogger("gopbot")
ai = AsyncAnthropic()  # берёт ANTHROPIC_API_KEY из окружения
stt = AsyncOpenAI() if os.getenv("OPENAI_API_KEY") else None

# подробность нейтральных заметок (facts), которые бот хранит в памяти
DETAIL_LIMITS = {
    "short": {"day": 500, "week": 700, "month": 900, "year": 1200},
    "normal": {"day": 1500, "week": 2000, "month": 2500, "year": 3000},
    "full": {"day": 5000, "week": 6000, "month": 7000, "year": 8000},
}
DETAIL_HINTS = {
    "short": "только главные темы и события",
    "normal": "темы, кто что сказал или решил, договорённости, цифры",
    "full": "максимально подробно: все темы, участники, цитаты ключевых фраз, ссылки, даты",
}
MEMORY_DETAIL = os.getenv("MEMORY_DETAIL", "normal").strip().lower()
if MEMORY_DETAIL not in DETAIL_LIMITS:
    log.warning("MEMORY_DETAIL=%r не знаю, беру normal", MEMORY_DETAIL)
    MEMORY_DETAIL = "normal"


# ---------- база ----------

def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER NOT NULL,
            author TEXT NOT NULL,
            text TEXT NOT NULL,
            ts TEXT NOT NULL
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS chats (
            chat_id INTEGER PRIMARY KEY,
            title TEXT,
            summary_time TEXT NOT NULL,
            prompt TEXT
        )"""
    )
    # память: итоги дней/недель/месяцев/лет. parent_id — в какую свёртку вошла запись
    conn.execute(
        """CREATE TABLE IF NOT EXISTS summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER NOT NULL,
            level TEXT NOT NULL,
            period_start TEXT NOT NULL,
            period_end TEXT NOT NULL,
            facts TEXT NOT NULL,
            post TEXT,
            created_at TEXT NOT NULL,
            parent_id INTEGER
        )"""
    )
    # сообщения других ботов, которые уже записали (через реплаи людей) — чтобы не дублировать
    conn.execute(
        """CREATE TABLE IF NOT EXISTS seen_bot_msgs (
            chat_id INTEGER NOT NULL,
            msg_id INTEGER NOT NULL,
            PRIMARY KEY (chat_id, msg_id)
        )"""
    )
    # миграция: нераспознанное медиа (JSON) и id сообщения в Telegram — для /retry
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(messages)")}
    if "media" not in cols:
        conn.execute("ALTER TABLE messages ADD COLUMN media TEXT")
        conn.execute("ALTER TABLE messages ADD COLUMN tg_msg_id INTEGER")
    # миграция: отвечать ли в чат пересказом ссылок (по умолчанию выкл)
    chat_cols = {r["name"] for r in conn.execute("PRAGMA table_info(chats)")}
    if "link_replies" not in chat_cols:
        conn.execute("ALTER TABLE chats ADD COLUMN link_replies INTEGER NOT NULL DEFAULT 0")
    # миграция: бот сейчас в чате? Выгнали — данные храним, но таймер не ставим
    if "active" not in chat_cols:
        conn.execute("ALTER TABLE chats ADD COLUMN active INTEGER NOT NULL DEFAULT 1")
    return conn


def ensure_chat(chat_id: int, title: str | None) -> tuple[sqlite3.Row, bool]:
    """Возвращает настройки чата; создаёт их при первом появлении."""
    with db() as conn:
        row = conn.execute("SELECT * FROM chats WHERE chat_id = ?", (chat_id,)).fetchone()
        if row:
            return row, False
        conn.execute(
            "INSERT INTO chats (chat_id, title, summary_time) VALUES (?, ?, ?)",
            (chat_id, title, DEFAULT_TIME),
        )
        row = conn.execute("SELECT * FROM chats WHERE chat_id = ?", (chat_id,)).fetchone()
        return row, True


def get_chat(chat_id: int) -> sqlite3.Row | None:
    with db() as conn:
        return conn.execute("SELECT * FROM chats WHERE chat_id = ?", (chat_id,)).fetchone()


def update_chat(chat_id: int, field: str, value):
    assert field in ("summary_time", "prompt", "title", "link_replies", "active")
    with db() as conn:
        conn.execute(f"UPDATE chats SET {field} = ? WHERE chat_id = ?", (value, chat_id))


def all_chats() -> list[sqlite3.Row]:
    with db() as conn:
        return conn.execute("SELECT * FROM chats").fetchall()


def wipe_chat_data(chat_id: int):
    """Стирает накопленное и память чата. Настройки (время, промпт) остаются."""
    with db() as conn:
        conn.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
        conn.execute("DELETE FROM summaries WHERE chat_id = ?", (chat_id,))
        conn.execute("DELETE FROM seen_bot_msgs WHERE chat_id = ?", (chat_id,))


def save_message(
    chat_id: int,
    author: str,
    text: str,
    ts: datetime.datetime,
    media: dict | None = None,
    tg_msg_id: int | None = None,
):
    with db() as conn:
        cur = conn.execute(
            "INSERT INTO messages (chat_id, author, text, ts, media, tg_msg_id)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (chat_id, author, text, ts.isoformat(),
             json.dumps(media) if media else None, tg_msg_id),
        )
        return cur.lastrowid


def append_message_text(row_id: int, extra: str):
    # если саммари уже забрало сообщение — строки нет, и ладно
    with db() as conn:
        conn.execute("UPDATE messages SET text = text || ? WHERE id = ?", (extra, row_id))


def failed_media(chat_id: int, limit: int):
    """Последние нераспознанные медиа, от старых к новым."""
    with db() as conn:
        rows = conn.execute(
            "SELECT id, author, media, tg_msg_id FROM messages"
            " WHERE chat_id = ? AND media IS NOT NULL ORDER BY ts DESC, id DESC LIMIT ?",
            (chat_id, limit),
        ).fetchall()
    return rows[::-1]


def find_media_row(chat_id: int, author: str, ts: datetime.datetime, tg_msg_id: int):
    with db() as conn:
        return conn.execute(
            "SELECT id FROM messages WHERE chat_id = ? AND"
            " (tg_msg_id = ? OR (tg_msg_id IS NULL AND author = ? AND ts = ? AND text LIKE '[%'))"
            " ORDER BY id DESC LIMIT 1",
            (chat_id, tg_msg_id, author, ts.isoformat()),
        ).fetchone()


def set_message_text(row_id: int, text: str, media: dict | None):
    with db() as conn:
        conn.execute(
            "UPDATE messages SET text = ?, media = ? WHERE id = ?",
            (text, json.dumps(media) if media else None, row_id),
        )


def fetch_messages(chat_id: int):
    with db() as conn:
        return conn.execute(
            "SELECT id, author, text, ts FROM messages WHERE chat_id = ? ORDER BY ts, id",
            (chat_id,),
        ).fetchall()


def delete_up_to(chat_id: int, max_id: int):
    with db() as conn:
        conn.execute("DELETE FROM messages WHERE chat_id = ? AND id <= ?", (chat_id, max_id))


def save_summary(chat_id: int, level: str, start: str, end: str, facts: str, post: str) -> int:
    with db() as conn:
        cur = conn.execute(
            "INSERT INTO summaries (chat_id, level, period_start, period_end, facts, post,"
            " created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (chat_id, level, start, end, facts, post,
             datetime.datetime.now(datetime.timezone.utc).isoformat()),
        )
        return cur.lastrowid


def live_summaries(chat_id: int, level: str) -> list[sqlite3.Row]:
    """Записи уровня, которые ещё не вошли в свёртку, от старых к новым."""
    with db() as conn:
        return conn.execute(
            "SELECT * FROM summaries WHERE chat_id = ? AND level = ? AND parent_id IS NULL"
            " ORDER BY period_start, id",
            (chat_id, level),
        ).fetchall()


def mark_compacted(ids: list[int], parent_id: int):
    with db() as conn:
        conn.executemany(
            "UPDATE summaries SET parent_id = ? WHERE id = ?", [(parent_id, i) for i in ids]
        )


# ---------- расписание ----------

def job_name(chat_id: int) -> str:
    return f"summary:{chat_id}"


def schedule_chat(app: Application, chat_id: int):
    """(Пере)ставит ежедневную задачу для чата по его настройкам."""
    for job in app.job_queue.get_jobs_by_name(job_name(chat_id)):
        job.schedule_removal()
    s = get_chat(chat_id)
    if not s:
        return
    hh, mm = map(int, s["summary_time"].split(":"))
    app.job_queue.run_daily(
        daily_job,
        time=datetime.time(hh, mm, tzinfo=TZ),
        chat_id=chat_id,
        name=job_name(chat_id),
    )
    log.info("Чат %s: саммари в %s", chat_id, s["summary_time"])


def unschedule_chat(app: Application, chat_id: int):
    for job in app.job_queue.get_jobs_by_name(job_name(chat_id)):
        job.schedule_removal()


async def daily_job(context: ContextTypes.DEFAULT_TYPE):
    await summarize_chat(context.bot, context.job.chat_id, silent_if_empty=True)
    await run_rollups(context.bot, context.job.chat_id)


# ---------- «печатает…» ----------

@contextlib.asynccontextmanager
async def typing(bot: Bot, chat_id: int):
    """Держит в чате статус «печатает…», пока идёт работа (Telegram гасит его через ~5 с)."""
    async def keep():
        while True:
            try:
                await bot.send_chat_action(chat_id, ChatAction.TYPING)
            except Exception:
                pass  # нет прав или канал — не страшно
            await asyncio.sleep(4)

    task = asyncio.create_task(keep())
    try:
        yield
    finally:
        task.cancel()


# ---------- кончились деньги у ИИ ----------

QUOTA_NOTICE_EVERY = datetime.timedelta(hours=6)
_quota_noticed: dict[tuple[int, str], datetime.datetime] = {}


def quota_provider(exc: Exception) -> str | None:
    """Если ошибка — закончился баланс, возвращает имя провайдера."""
    if isinstance(exc, openai.APIStatusError) and (
        getattr(exc, "code", None) == "insufficient_quota" or "insufficient_quota" in str(exc)
    ):
        return "OpenAI"
    if isinstance(exc, anthropic.APIStatusError) and "credit balance" in str(exc).lower():
        return "Anthropic"
    return None


def quota_notice(chat_id: int, exc: Exception, what: str, always: bool = False) -> str | None:
    """Текст «кончились токены» — не чаще раза в QUOTA_NOTICE_EVERY на чат и провайдера."""
    provider = quota_provider(exc)
    if not provider:
        return None
    now = datetime.datetime.now(datetime.timezone.utc)
    last = _quota_noticed.get((chat_id, provider))
    if not always and last and now - last < QUOTA_NOTICE_EVERY:
        return None
    _quota_noticed[(chat_id, provider)] = now
    return (
        f"🪫 У меня кончились токены в {provider} — {what} пока не могу. "
        "Пополните баланс, потом /retry."
    )


# ---------- логика саммари ----------

def build_transcript(rows) -> str:
    lines = []
    for r in rows:
        t = datetime.datetime.fromisoformat(r["ts"]).astimezone(TZ).strftime("%H:%M")
        lines.append(f"[{t}] {r['author']}: {r['text']}")
    transcript = "\n".join(lines)
    if len(transcript) > MAX_TRANSCRIPT_CHARS:
        transcript = transcript[-MAX_TRANSCRIPT_CHARS:]  # берём самое свежее
    return transcript


LEVEL_NAMES = {"day": "день", "week": "неделя", "month": "месяц", "year": "год"}
# (что сворачиваем, во что, сколько нужно)
ROLLUPS = [("day", "week", 7), ("week", "month", 4), ("month", "year", 12)]
ROLLUP_TITLES = {"week": "🗓 Итоги недели", "month": "📅 Итоги месяца", "year": "🎆 Итоги года"}
ROLLUP_ASK = {
    "week": "Вот итоги по дням. Сделай итог недели",
    "month": "Вот итоги по неделям. Сделай итог месяца",
    "year": "Вот итоги по месяцам. Сделай итог года",
}

FACTS_POST_SCHEMA = {
    "type": "object",
    "properties": {"facts": {"type": "string"}, "post": {"type": "string"}},
    "required": ["facts", "post"],
    "additionalProperties": False,
}


async def facts_and_post(prompt: str, level: str, content: str) -> dict:
    """Один запрос — два текста: нейтральные facts для памяти и post в стиле чата.

    Стиль чата касается только post: в память и в свёртки идут facts, поэтому
    стиль не накапливается от уровня к уровню.
    """
    system = (
        "Ты ведёшь хронику чата и отвечаешь JSON с двумя полями.\n"
        "facts — нейтральные заметки для долговременной памяти: без стиля, мата, оценок "
        f"и шуток. Включи: {DETAIL_HINTS[MEMORY_DETAIL]}. Упоминай участников по именам. "
        f"Уложись примерно в {DETAIL_LIMITS[MEMORY_DETAIL][level]} символов.\n"
        "post — текст для отправки в чат, строго по инструкции стиля ниже. "
        "Инструкция стиля относится ТОЛЬКО к post.\n\n"
        f"Инструкция стиля для post:\n{prompt}"
    )
    resp = await ai.messages.create(
        model=MODEL,
        max_tokens=16000,  # с запасом: часть уходит на размышления модели
        system=system,
        messages=[{"role": "user", "content": content}],
        output_config={"format": {"type": "json_schema", "schema": FACTS_POST_SCHEMA}},
    )
    if resp.stop_reason in ("refusal", "max_tokens"):
        raise RuntimeError(f"модель не дописала ответ: {resp.stop_reason}")
    data = json.loads(next(b.text for b in resp.content if b.type == "text"))
    return {"facts": data["facts"].strip(), "post": data["post"].strip()}


async def make_summary(prompt: str, transcript: str, memory: str = "") -> dict:
    content = f"Вот переписка за день:\n\n{transcript}"
    if memory:
        content = (
            "Память чата (что было раньше) — используй для отсылок и связей, "
            f"но саммари — про сегодняшнюю переписку:\n\n{memory}\n\n---\n\n{content}"
        )
    return await facts_and_post(prompt, "day", content)


async def make_rollup(prompt: str, level: str, items: list[sqlite3.Row]) -> dict:
    notes = "\n\n".join(
        f"[{fmt_period(r['period_start'], r['period_end'])}] {r['facts']}" for r in items
    )
    content = (
        f"{ROLLUP_ASK[level]}: главные темы, события, кто отличился, "
        f"что изменилось за период.\n\n{notes}"
    )
    return await facts_and_post(prompt, level, content)


def fmt_period(start: str, end: str) -> str:
    a, b = datetime.date.fromisoformat(start), datetime.date.fromisoformat(end)
    if a == b:
        return f"{a:%d.%m.%Y}"
    if a.year == b.year:
        return f"{a:%d.%m}–{b:%d.%m.%Y}"
    return f"{a:%d.%m.%Y}–{b:%d.%m.%Y}"


def memory_text(chat_id: int) -> str:
    """Нейтральная память чата: годы → месяцы → недели → дни (только facts)."""
    parts = []
    for level in ("year", "month", "week", "day"):
        for r in live_summaries(chat_id, level):
            period = fmt_period(r["period_start"], r["period_end"])
            parts.append(f"[{LEVEL_NAMES[level]} {period}] {r['facts']}")
    return "\n\n".join(parts)


def local_date(ts: str) -> str:
    return datetime.datetime.fromisoformat(ts).astimezone(TZ).date().isoformat()


async def run_rollups(bot: Bot, chat_id: int):
    """Сворачивает накопленное: 7 дней → неделя, 4 недели → месяц, 12 месяцев → год."""
    s = get_chat(chat_id)
    if not s:
        return
    prompt = s["prompt"] or DEFAULT_PROMPT
    for child, parent, need in ROLLUPS:
        items = live_summaries(chat_id, child)
        # несколько /summary за день — всё равно один день
        count = len({r["period_end"] for r in items}) if child == "day" else len(items)
        if count < need:
            continue
        try:
            async with typing(bot, chat_id):
                res = await make_rollup(prompt, parent, items)
        except Exception as e:
            log.exception("Не смог свернуть %s → %s в чате %s", child, parent, chat_id)
            notice = quota_notice(chat_id, e, f"подводить итоги ({LEVEL_NAMES[parent]})")
            if notice:
                await bot.send_message(chat_id, notice)
            return  # накопленное останется живым — попробуем со следующим таймером
        start = items[0]["period_start"]
        end = max(r["period_end"] for r in items)
        pid = save_summary(chat_id, parent, start, end, res["facts"], res["post"])
        mark_compacted([r["id"] for r in items], pid)
        log.info("Чат %s: %d × %s → %s #%d", chat_id, len(items), child, parent, pid)
        await send_long(
            bot, chat_id, f"{ROLLUP_TITLES[parent]} ({fmt_period(start, end)}):\n\n{res['post']}"
        )


async def send_long(bot: Bot, chat_id: int, text: str):
    for i in range(0, len(text), TG_MSG_LIMIT):
        await bot.send_message(chat_id, text[i : i + TG_MSG_LIMIT])


async def summarize_chat(
    bot: Bot, chat_id: int, silent_if_empty: bool = False, preview: bool = False
):
    """Итоги дня по таймеру: постит, пишет день в память и очищает сообщения.

    preview=True (ручной /summary) — только показывает саммари на сейчас: ничего не
    удаляет и в память не пишет, чтобы ручные вызовы не дробили день.
    """
    s = get_chat(chat_id)
    if not s:
        return
    rows = fetch_messages(chat_id)
    if not rows:
        if not silent_if_empty:
            await bot.send_message(chat_id, "Пока нечего саммарить — сообщений нет.")
        return
    try:
        async with typing(bot, chat_id):
            summary = await make_summary(
                s["prompt"] or DEFAULT_PROMPT, build_transcript(rows), memory_text(chat_id)
            )
    except Exception as e:
        log.exception("Ошибка при запросе к ИИ для чата %s", chat_id)
        provider = quota_provider(e)
        if provider:
            text = (f"🪫 У меня кончились токены в {provider} — саммари не сделать. "
                    "Пополните баланс, сообщения не потеряются.")
        else:
            text = "ИИ не ответил, попробуй позже." if preview else \
                "ИИ не ответил, попробую в следующий раз."
        await bot.send_message(chat_id, text)
        return  # сообщения не удаляем — уйдут в следующее саммари
    if preview:
        await send_long(
            bot, chat_id, f"🗞 Саммари на сейчас ({len(rows)} сообщ.):\n\n{summary['post']}"
        )
        return
    await send_long(bot, chat_id, f"🗞 Итоги дня ({len(rows)} сообщ.):\n\n{summary['post']}")
    save_summary(chat_id, "day", local_date(rows[0]["ts"]), local_date(rows[-1]["ts"]),
                 summary["facts"], summary["post"])
    delete_up_to(chat_id, max(r["id"] for r in rows))


# ---------- проверки ----------

async def guard(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """Общие проверки для команд. Возвращает True, если можно продолжать."""
    chat = update.effective_chat
    msg = update.effective_message
    if chat.type == "private":
        await msg.reply_text("Добавь меня в группу или канал — там и настроим.")
        return False
    _, created = ensure_chat(chat.id, chat.title)
    if created:
        schedule_chat(context.application, chat.id)
    return True


# ---------- хендлеры ----------

def message_author(update: Update) -> str | None:
    msg = update.effective_message
    user = update.effective_user
    if user and user.is_bot:
        return None
    if msg.sender_chat:  # канал, анонимный админ или привязанный канал
        return msg.author_signature or msg.sender_chat.title or "Канал"
    if user:
        return user.full_name or user.username or str(user.id)
    return None


def capture_bot_reply(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Telegram не присылает боту сообщения других ботов. Но если человек ответил
    реплаем на сообщение бота — оно приходит внутри реплая, и его можно записать."""
    msg = update.effective_message
    target = msg.reply_to_message if msg else None
    bot_user = target.from_user if target else None
    if not bot_user or not bot_user.is_bot or bot_user.id == context.bot.id:
        return
    text = (target.text or target.caption or "").strip()
    if not text:
        return
    chat_id = update.effective_chat.id
    with db() as conn:
        cur = conn.execute(
            "INSERT OR IGNORE INTO seen_bot_msgs (chat_id, msg_id) VALUES (?, ?)",
            (chat_id, target.message_id),
        )
        if not cur.rowcount:
            return  # уже записали по прошлому реплаю
    name = bot_user.full_name + (f" (@{bot_user.username})" if bot_user.username else "")
    save_message(chat_id, f"бот {name}", text, target.date, tg_msg_id=target.message_id)


async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    chat = update.effective_chat
    if not msg or not msg.text:
        return
    author = message_author(update)
    if not author:
        return
    _, created = ensure_chat(chat.id, chat.title)
    if created:
        schedule_chat(context.application, chat.id)
    if await on_settings_reply(update, context):
        return  # это настройка, а не реплика в чате
    capture_bot_reply(update, context)
    row_id = save_message(chat.id, author, msg.text, msg.date)
    urls = message_urls(msg)
    if urls:
        # открываем ссылки фоном, чтобы не держать остальные апдейты
        context.application.create_task(
            enrich_links(row_id, urls, chat.id, context.bot, msg.message_id)
        )


# ---------- ссылки ----------

# web_fetch — серверный инструмент Claude: страницу открывает Anthropic, не наш сервер
LINK_TOOL = {"type": "web_fetch_20250910", "name": "web_fetch", "max_uses": 1,
             "max_content_tokens": 8000}


def message_urls(msg) -> list[str]:
    """Ссылки из сообщения: явные и спрятанные под текст, без повторов."""
    if not MAX_LINKS:
        return []
    urls = []
    for ent, text in msg.parse_entities([MessageEntity.URL, MessageEntity.TEXT_LINK]).items():
        url = ent.url if ent.type == MessageEntity.TEXT_LINK else text
        if not re.match(r"^https?://", url, re.I):
            url = "http://" + url
        if url not in urls:
            urls.append(url)
    return urls[:MAX_LINKS]


async def describe_link(url: str) -> str:
    messages = [{"role": "user", "content": f"Открой ссылку и опиши её: {url}"}]
    for _ in range(3):  # pause_turn — сервер просит продолжить
        resp = await ai.messages.create(
            model=FAST_MODEL,
            max_tokens=2000,
            system=(
                "Ты пишешь заметку в хронику чата о присланной ссылке. Всегда сначала открой "
                "её инструментом web_fetch — даже если домен знакомый. Потом коротко и "
                "нейтрально опиши, что там: 3–6 предложения, о чём страница или документ, "
                "без оценок. Если не открылась, пустая или это заглушка (логин, капча, "
                "cookie-баннер) — так и скажи одной фразой. Ответ — только сама заметка: "
                "не обращайся к собеседнику и ничего не предлагай."
            ),
            tools=[LINK_TOOL],
            messages=messages,
        )
        if resp.stop_reason != "pause_turn":
            break
        messages.append({"role": "assistant", "content": resp.content})
    # берём только итоговый текст — после результата web_fetch, без «сейчас открою…»
    blocks = resp.content
    done = [i for i, b in enumerate(blocks) if b.type == "web_fetch_tool_result"]
    tail = blocks[done[-1] + 1 :] if done else blocks
    return "".join(b.text for b in tail if b.type == "text").strip()


async def retell_link(note: str, prompt: str) -> str:
    """Пересказ нейтральной заметки о ссылке в стиле промпта чата — только для ответа в чат."""
    resp = await ai.messages.create(
        model=FAST_MODEL,
        max_tokens=1000,
        system=prompt,
        messages=[{
            "role": "user",
            "content": "Человек скинул в чат ссылку. Вот что на ней (нейтральная заметка). "
            "Перескажи коротко, 2–4 предложения, что там, и отреагируй. Добавь пару "
            f"подходящих эмодзи.\n\n{note}",
        }],
    )
    return "".join(b.text for b in resp.content if b.type == "text").strip()


async def enrich_links(row_id: int, urls: list[str], chat_id: int, bot: Bot, reply_to: int):
    """Открывает ссылки и дописывает их описание к сообщению в базе.

    Если в чате включён пересказ ссылок — ещё и отвечает в чат в стиле промпта
    (с превью ссылки). В базу идёт только нейтральная заметка.
    """
    s = get_chat(chat_id)
    notes = []
    for url in urls:
        try:
            note = await describe_link(url) or ""
        except Exception as e:
            log.info("Ссылка %s в чате %s не открылась: %s", url, chat_id, e)
            note = ""
        notes.append(f"\n[ссылка {url}: {note or 'не открылась'}]")
        if not (note and s and s["link_replies"]):
            continue
        try:
            async with typing(bot, chat_id):
                text = await retell_link(note, s["prompt"] or DEFAULT_PROMPT)
        except Exception as e:
            log.exception("Не смог пересказать ссылку в чате %s", chat_id)
            text = quota_notice(chat_id, e, "пересказывать ссылки")
        if text:
            await send_reply(bot, chat_id, reply_to, f"🔗 {text}\n\n{url}")
    append_message_text(row_id, "".join(notes))


# ---------- картинки и голосовые ----------

IMAGE_TYPES = ("image/jpeg", "image/png", "image/gif", "image/webp")
MAX_IMAGE_BYTES = 5 * 1024 * 1024  # лимит Claude на картинку
MAX_AUDIO_BYTES = 20 * 1024 * 1024  # больше Bot API не отдаёт


async def describe_images(
    images: list[tuple[bytes, str]], prompt: str, caption: str, total: int
) -> dict:
    """Одна картинка или пачка — одним запросом. Возвращает {facts, post}:
    facts — нейтральное описание для базы и саммари, post — реакция в стиле чата.
    """
    if len(images) == 1:
        ask = "Это картинка из чата."
        size = "1–3 предложения"
    else:
        ask = (
            f"Это {len(images)} картинок, которые человек отправил в чат подряд"
            + (f" (из {total}, остальные не показаны)" if total > len(images) else "")
            + ". Разбери их вместе, как одну историю: что их связывает."
        )
        size = "2–5 предложений"
    if caption:
        ask += f"\nПодпись: {caption}"
    system = (
        "Ты смотришь картинки из чата и отвечаешь JSON с двумя полями.\n"
        "facts — нейтральное описание для хроники чата: что изображено, текст на "
        f"картинках, если это мем — в чём шутка. Без стиля, мата и оценок, {size}.\n"
        f"post — реакция для отправки в чат, {size}, с парой подходящих эмодзи, строго "
        "по инструкции стиля ниже. Инструкция стиля относится ТОЛЬКО к post.\n\n"
        f"Инструкция стиля для post:\n{prompt}"
    )
    content = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": base64.standard_b64encode(data).decode(),
            },
        }
        for data, media_type in images
    ]
    resp = await ai.messages.create(
        model=VISION_MODEL,
        max_tokens=4000,  # с запасом: часть уходит на размышления модели
        output_config={
            "effort": "low",
            "format": {"type": "json_schema", "schema": FACTS_POST_SCHEMA},
        },
        system=system,
        messages=[{"role": "user", "content": content + [{"type": "text", "text": ask}]}],
    )
    if resp.stop_reason in ("refusal", "max_tokens"):
        raise RuntimeError(f"модель не дописала ответ: {resp.stop_reason}")
    data = json.loads(next(b.text for b in resp.content if b.type == "text"))
    return {"facts": data["facts"].strip(), "post": data["post"].strip()}


def voice_pitch(data: bytes) -> float | None:
    """Медианный основной тон голоса в Гц (автокорреляция), None если не вышло."""
    sr, n, hop = 16000, 640, 320  # окно 40 мс, шаг 20 мс
    lo, hi = sr // 400, sr // 70  # ищем тон в диапазоне 70–400 Гц
    chunks = []
    with av.open(io.BytesIO(data)) as c:
        rs = av.AudioResampler(format="s16", layout="mono", rate=sr)
        for frame in c.decode(audio=0):
            chunks += [f.to_ndarray().reshape(-1) for f in rs.resample(frame)]
        chunks += [f.to_ndarray().reshape(-1) for f in rs.resample(None)]
    if not chunks:
        return None
    x = np.concatenate(chunks).astype(np.float32) / 32768
    frames = [x[i : i + n] - x[i : i + n].mean() for i in range(0, len(x) - n, hop)]
    if not frames:
        return None
    rms = np.array([np.sqrt((f ** 2).mean()) for f in frames])
    thr = 0.3 * np.percentile(rms, 90)  # тишину и шум не считаем
    f0s = []
    for f, e in zip(frames, rms):
        if e < thr:
            continue
        ac = np.correlate(f, f, "full")[n - 1 :]
        if ac[0] <= 0:
            continue
        ac = ac[: hi + 2] / ac[0]
        peaks = [k for k in range(lo, hi + 1) if ac[k] > ac[k - 1] and ac[k] >= ac[k + 1]]
        if not peaks:
            continue
        best = max(ac[k] for k in peaks)
        if best < 0.5:
            continue  # не голос (шипящие, шум)
        # первый почти-максимальный пик — защита от ошибки на октаву вниз
        lag = next(k for k in peaks if ac[k] >= 0.85 * best)
        f0s.append(sr / lag)
    return float(np.median(f0s)) if len(f0s) >= 10 else None


def voice_gender(pitch: float | None) -> str | None:
    if pitch is None:
        return None
    if pitch < 150:
        return "мужской"
    if pitch > 175:
        return "женский"
    return None


async def react_to_voice(transcript: str, prompt: str, author: str, gender: str | None) -> str:
    if gender:
        who = f"По голосу — {gender} голос, учитывай пол говорящего."
    else:
        who = "Пол по голосу не определён — не угадывай, обращайся нейтрально."
    resp = await ai.messages.create(
        model=FAST_MODEL,
        max_tokens=400,
        system=prompt,
        messages=[{
            "role": "user",
            "content": "Это расшифровка голосового сообщения из чата. Отреагируй на него "
            "в 1–3 предложениях: в чём суть и что ты об этом думаешь. "
            "Добавь пару подходящих эмодзи.\n\n"
            f"Автор в чате: {author}. {who}\n"
            f"Расшифровка: {transcript}",
        }],
    )
    return "".join(b.text for b in resp.content if b.type == "text").strip()


async def transcribe(data: bytes, filename: str) -> str:
    resp = await stt.audio.transcriptions.create(model=STT_MODEL, file=(filename, data))
    return resp.text.strip()


def fmt_duration(sec: int | None) -> str:
    sec = sec or 0
    return f"{sec // 60}:{sec % 60:02d}"


def media_spec(msg) -> dict | None:
    """Что за медиа в сообщении — в виде, который можно сохранить и перепрогнать."""
    spec = {"caption": (msg.caption or "").strip()}
    if msg.sticker:
        return spec | {"kind": "sticker", "emoji": msg.sticker.emoji or ""}
    if msg.photo:
        # самый крупный вариант не больше ~1600px — Claude всё равно ужмёт
        fit = [p for p in msg.photo if max(p.width, p.height) <= 1600] or msg.photo[:1]
        p = fit[-1]
        return spec | {"kind": "image", "file_id": p.file_id, "group": msg.media_group_id,
                       "media_type": "image/jpeg", "size": p.file_size}
    if msg.document and msg.document.mime_type in IMAGE_TYPES:
        d = msg.document
        return spec | {"kind": "image", "file_id": d.file_id, "group": msg.media_group_id,
                       "media_type": d.mime_type, "size": d.file_size}
    audio = msg.voice or msg.video_note
    if audio:
        return spec | {"kind": "voice" if msg.voice else "video_note",
                       "file_id": audio.file_id, "size": audio.file_size,
                       "duration": audio.duration}
    return None


async def process_media(
    spec: dict,
    context: ContextTypes.DEFAULT_TYPE,
    prompt: str,
    chat_id: int,
    author: str,
    always_notify: bool = False,
) -> tuple[str, str | None, bool]:
    """Возвращает (текст для базы, ответ в чат, стоит ли перепробовать позже).

    Если у ИИ кончились деньги — ответ в чат об этом (обычно не чаще раза в 6 часов).
    """
    caption = spec["caption"]
    tail = f" | подпись: {caption}" if caption else ""
    kind = spec["kind"]

    if kind == "sticker":
        return f"[стикер {spec['emoji']}]".replace(" ]", "]"), None, False

    if kind in ("image", "batch"):
        parts = spec["items"] if kind == "batch" else [spec]
        total = len(parts)
        parts = [p for p in parts if not (p["size"] and p["size"] > MAX_IMAGE_BYTES)]
        if not spec.get("album"):  # альбом смотрим целиком, серию — до лимита
            parts = parts[:MAX_BATCH_PHOTOS]
        what = "картинка" if total == 1 else f"{total} фото"
        if not parts:
            return f"[{what}, слишком большие]{tail}", None, False
        seen = f", смотрел {len(parts)} из {total}" if len(parts) < total else ""
        try:
            images = []
            for p in parts:
                f = await context.bot.get_file(p["file_id"])
                images.append((bytes(await f.download_as_bytearray()), p["media_type"]))
            desc = await describe_images(images, prompt, caption, total)
            # в базу — нейтральное описание, в чат — реакция в стиле промпта
            return f"[{what}{seen}: {desc['facts']}]{tail}", f"🖼 {desc['post']}", False
        except Exception as e:
            log.exception("Не смог описать картинки в чате %s", chat_id)
            notice = quota_notice(chat_id, e, "смотреть картинки", always_notify)
            return f"[{what}]{tail}", notice, True

    label = "голосовое" if kind == "voice" else "кружочек"
    head = f"[{label} {fmt_duration(spec['duration'])}"
    if spec["size"] and spec["size"] > MAX_AUDIO_BYTES:
        return f"{head}]{tail}", None, False
    if not stt:
        return f"{head}]{tail}", None, True
    try:
        f = await context.bot.get_file(spec["file_id"])
        name = "voice.ogg" if kind == "voice" else "note.mp4"
        data = bytes(await f.download_as_bytearray())
        text = await transcribe(data, name)
    except Exception as e:
        log.exception("Не смог расшифровать %s в чате %s", label, chat_id)
        notice = quota_notice(chat_id, e, "слушать голосовые", always_notify)
        return f"{head}]{tail}", notice, True
    if not text:
        return f"{head}, тишина]{tail}", None, False
    try:
        pitch = await asyncio.to_thread(voice_pitch, data)
    except Exception:
        log.exception("Не смог оценить тон голоса в чате %s", chat_id)
        pitch = None
    gender = voice_gender(pitch)
    log.info("Голос в чате %s: тон %s Гц → %s", chat_id,
             f"{pitch:.0f}" if pitch else "?", gender or "не понятно")
    if gender:
        head += f", {gender} голос"
    reply = None  # в чат — только реакция, расшифровка идёт лишь в базу
    try:
        reply = await react_to_voice(text, prompt, author, gender) or None
    except Exception as e:
        log.exception("Не смог отреагировать на %s в чате %s", label, chat_id)
        reply = quota_notice(chat_id, e, "отвечать на голосовые", always_notify)
    return f"{head}: {text}]{tail}", reply, False


async def send_reply(bot: Bot, chat_id: int, reply_to: int | None, text: str):
    try:
        for i in range(0, len(text), TG_MSG_LIMIT):
            await bot.send_message(
                chat_id, text[i : i + TG_MSG_LIMIT],
                reply_to_message_id=reply_to, allow_sending_without_reply=True,
            )
    except Exception:
        log.info("Не смог ответить в %s (нет прав писать?)", chat_id)


# фото, которые автор ещё досылает: (chat_id, автор) -> {"msgs": [...], "seq": n}
_photo_batches: dict[tuple[int, str], dict] = {}


async def collect_photo(msg, spec: dict, chat_id: int, author: str) -> tuple | None:
    """Копит фото автора, пока он шлёт их (альбомом или по одной).

    Каждое новое фото продлевает ожидание на PHOTO_BATCH_WAIT. Пачку получает
    тот вызов, после которого новых фото не пришло, — остальные возвращают None.
    """
    key = (chat_id, author)
    batch = _photo_batches.setdefault(key, {"msgs": [], "seq": 0})
    batch["msgs"].append((msg, spec))
    batch["seq"] += 1
    my = batch["seq"]
    await asyncio.sleep(PHOTO_BATCH_WAIT)
    if _photo_batches.get(key) is not batch or batch["seq"] != my:
        return None  # пришли ещё фото — пачку заберёт последний
    del _photo_batches[key]
    msgs = sorted(batch["msgs"], key=lambda x: x[0].message_id)
    if len(msgs) == 1:
        return msgs[0]
    specs = [sp for _, sp in msgs]
    groups = {sp.get("group") for sp in specs}
    batch_spec = {
        "kind": "batch",
        "caption": " / ".join(sp["caption"] for sp in specs if sp["caption"]),
        "items": specs,
        "album": len(groups) == 1 and None not in groups,  # один альбом — смотрим все
    }
    log.info("Чат %s: пачка из %d фото от %s%s", chat_id, len(msgs), author,
             " (альбом)" if batch_spec["album"] else "")
    return msgs[0][0], batch_spec


async def on_media(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    chat = update.effective_chat
    author = message_author(update)
    spec = media_spec(msg) if msg else None
    if not author or not spec:
        return
    s, created = ensure_chat(chat.id, chat.title)
    if created:
        schedule_chat(context.application, chat.id)
    capture_bot_reply(update, context)
    if spec["kind"] == "image":
        got = await collect_photo(msg, spec, chat.id, author)
        if not got:
            return
        msg, spec = got  # дальше работаем с первым сообщением пачки
    # стикеры обрабатываются мгновенно — им «печатает…» не нужен
    quiet = spec["kind"] == "sticker"
    async with (contextlib.nullcontext() if quiet else typing(context.bot, chat.id)):
        text, reply, retry = await process_media(
            spec, context, s["prompt"] or DEFAULT_PROMPT, chat.id, author
        )
    save_message(chat.id, author, text, msg.date,
                 media=spec if retry else None, tg_msg_id=msg.message_id)
    if reply:
        await send_reply(context.bot, chat.id, msg.message_id, reply)


async def cmd_retry(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/retry реплаем на медиа — распознать его; /retry N — последние N неудачных."""
    if not await guard(update, context):
        return
    msg = update.effective_message
    chat = update.effective_chat
    prompt = get_chat(chat.id)["prompt"] or DEFAULT_PROMPT

    target = msg.reply_to_message
    if target:
        spec = media_spec(target)
        if not spec:
            await msg.reply_text("Это не голосовое, не кружочек и не картинка.")
            return
        author = message_author(Update(0, message=target)) or "?"
        async with typing(context.bot, chat.id):
            text, reply, retry = await process_media(
                spec, context, prompt, chat.id, author, always_notify=True
            )
        row = find_media_row(chat.id, author, target.date, target.message_id)
        if row:
            set_message_text(row["id"], text, spec if retry else None)
        else:
            save_message(chat.id, author, text, target.date,
                         media=spec if retry else None, tg_msg_id=target.message_id)
        await send_reply(context.bot, chat.id, target.message_id,
                         reply or "Не вышло распознать, попробуй позже.")
        return

    n = int(context.args[0]) if context.args and context.args[0].isdigit() else 1
    rows = failed_media(chat.id, min(n, 20))
    if not rows:
        await msg.reply_text(
            "Нераспознанных медиа нет. Чтобы перепрогнать конкретное — "
            "ответь /retry на голосовое или картинку."
        )
        return
    ok = 0
    for r in rows:
        spec = json.loads(r["media"])
        async with typing(context.bot, chat.id):
            text, reply, retry = await process_media(
                spec, context, prompt, chat.id, r["author"]
            )
        set_message_text(r["id"], text, spec if retry else None)
        ok += not retry
        if reply:
            await send_reply(context.bot, chat.id, r["tg_msg_id"], reply)
    await msg.reply_text(f"Перепрогнал {len(rows)}, распознал {ok}.")


async def on_my_member(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Бота добавили в чат или удалили из него."""
    chat = update.effective_chat
    status = update.my_chat_member.new_chat_member.status
    if status in ("member", "administrator"):
        _, created = ensure_chat(chat.id, chat.title)
        update_chat(chat.id, "active", 1)
        schedule_chat(context.application, chat.id)  # вернули в чат — таймер снова в деле
        if created:
            try:
                await context.bot.send_message(
                    chat.id,
                    f"Здарова! Буду собирать сообщения и каждый день в {DEFAULT_TIME} "
                    "кидать саммари.\n"
                    "Настроить может любой: /menu",
                )
            except Exception:
                log.info("Не смог поздороваться в %s (нет прав писать?)", chat.id)
    elif status in ("left", "kicked"):
        # память не стираем: вернут бота — продолжит с того же места
        unschedule_chat(context.application, chat.id)
        if get_chat(chat.id):
            update_chat(chat.id, "active", 0)
        log.info("Бота убрали из чата %s, данные сохранены", chat.id)


def apply_time(app: Application, chat_id: int, raw: str) -> str:
    m = TIME_RE.match(raw.strip())
    if not m:
        return "Не понял время. Формат: ЧЧ:ММ, например 21:30."
    value = f"{int(m.group(1)):02d}:{m.group(2)}"
    update_chat(chat_id, "summary_time", value)
    schedule_chat(app, chat_id)
    return f"Ок, саммари теперь каждый день в {value}."


def apply_prompt(chat_id: int, text: str) -> str:
    text = text.strip()
    if not text:
        return "Пустой промпт не годится."
    if len(text) > MAX_PROMPT_CHARS:
        return f"Слишком длинно, максимум {MAX_PROMPT_CHARS} символов."
    update_chat(chat_id, "prompt", text)
    return "Ок, промпт обновлён."


async def cmd_summary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await guard(update, context):
        return
    await summarize_chat(context.bot, update.effective_chat.id, preview=True)


def plural(n: int, one: str, few: str, many: str) -> str:
    if n % 10 == 1 and n % 100 != 11:
        return f"{n} {one}"
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return f"{n} {few}"
    return f"{n} {many}"


def until_summary(summary_time: str) -> str:
    now = datetime.datetime.now(TZ)
    hh, mm = map(int, summary_time.split(":"))
    nxt = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if nxt <= now:
        nxt += datetime.timedelta(days=1)
    mins = int((nxt - now).total_seconds() // 60)
    h, m = divmod(mins, 60)
    return f"{h} ч {m} мин" if h else f"{m} мин"


def stats_text(chat_id: int) -> str:
    """Статистика чата: что накоплено, сколько до саммари, что в памяти."""
    s = get_chat(chat_id)
    rows = fetch_messages(chat_id)
    texts = [r["text"] for r in rows]
    voices = sum(t.startswith(("[голосовое", "[кружочек")) for t in texts)
    images = sum(bool(re.match(r"\[(картинка|\d+ фото)", t)) for t in texts)
    links = sum(t.count("\n[ссылка ") for t in texts)
    extra = [x for x in (
        voices and plural(voices, "голосовое", "голосовых", "голосовых"),
        images and plural(images, "картинка", "картинки", "картинок"),
        links and plural(links, "ссылка", "ссылки", "ссылок"),
    ) if x]
    lines = [
        f"💬 Накоплено сообщений: {len(rows)}" + (f" ({', '.join(extra)})" if extra else ""),
        f"⏳ До саммари: {until_summary(s['summary_time'])} (в {s['summary_time']})",
    ]

    # вся история, включая уже свёрнутое: сколько дней бот знает чат и сколько итогов подвёл
    with db() as conn:
        days, since = conn.execute(
            "SELECT COUNT(DISTINCT period_end), MIN(period_start) FROM summaries"
            " WHERE chat_id = ? AND level = 'day'", (chat_id,),
        ).fetchone()
        done = dict(conn.execute(
            "SELECT level, COUNT(*) FROM summaries WHERE chat_id = ? GROUP BY level", (chat_id,),
        ).fetchall())
    if not days:
        lines.append("🧠 Память: пока пусто")
    else:
        lines.append(
            f"🧠 Помню {plural(days, 'день', 'дня', 'дней')} "
            f"(с {datetime.date.fromisoformat(since):%d.%m.%Y}): "
            f"{plural(done.get('week', 0), 'неделя', 'недели', 'недель')}, "
            f"{plural(done.get('month', 0), 'месяц', 'месяца', 'месяцев')}, "
            f"{plural(done.get('year', 0), 'год', 'года', 'лет')}"
        )
    return "\n".join(lines)


async def cmd_memory(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await guard(update, context):
        return
    chat = update.effective_chat
    s = get_chat(chat.id)
    memory = memory_text(chat.id)
    if not memory:
        await update.effective_message.reply_text("Пока ничего не помню.")
        return
    try:
        # пересказ в стиле чата — только для показа, в память не сохраняется
        async with typing(context.bot, chat.id):
            resp = await ai.messages.create(
                model=MODEL,
                max_tokens=16000,
                system=s["prompt"] or DEFAULT_PROMPT,
                messages=[{
                    "role": "user",
                    "content": "Вот твои заметки о том, что было в этом чате. Расскажи "
                    "участникам, что ты помнишь: от давнего к свежему — год, месяцы, недели, "
                    "последние дни. Ничего не выдумывай сверх заметок.\n\n" + memory,
                }],
            )
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
    except Exception as e:
        log.exception("Не смог пересказать память чата %s", chat.id)
        text = quota_notice(chat.id, e, "вспоминать", always=True) or "ИИ не ответил, попробуй позже."
    await send_long(context.bot, chat.id, f"🧠 Что я помню:\n\n{text}")


# ---------- меню на кнопках ----------

TIME_PRESETS = ["09:00", "12:00", "15:00", "18:00", "20:00", "21:00", "22:00", "23:00"]


# сообщения-запросы бота: ответ на них реплаем меняет настройку
ASK_TIME = "⏰ Ответь на это сообщение временем саммари в формате ЧЧ:ММ, например 21:30."
ASK_PROMPT = "📝 Ответь на это сообщение текстом нового промпта для саммари."


def menu_text(chat_id: int) -> str:
    return "🤖 Меню бота\n" + stats_text(chat_id)


def menu_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🗞 Саммари сейчас", callback_data="m:summary")],
        [InlineKeyboardButton("⚙️ Настройки", callback_data="m:settings")],
        [InlineKeyboardButton("🔄 Обновить", callback_data="m:home")],
    ])


def settings_text(chat_id: int) -> str:
    s = get_chat(chat_id)
    kind = "свой" if s["prompt"] else "по умолчанию"
    return (
        "⚙️ Настройки чата\n"
        f"⏰ Саммари каждый день в {s['summary_time']}\n"
        f"📝 Промпт: {kind}\n"
        f"🔗 Пересказ ссылок в чат: {'вкл' if s['link_replies'] else 'выкл'}"
    )


def settings_kb(chat_id: int) -> InlineKeyboardMarkup:
    s = get_chat(chat_id)
    state = "вкл ✅" if s["link_replies"] else "выкл"
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("⏰ Время", callback_data="m:time"),
            InlineKeyboardButton("📝 Промпт", callback_data="m:prompt"),
        ],
        [InlineKeyboardButton(f"🔗 Пересказ ссылок: {state}", callback_data="m:links")],
        [InlineKeyboardButton("🗑 Удалить данные чата", callback_data="m:wipe")],
        [InlineKeyboardButton("◀️ Назад", callback_data="m:home")],
    ])


def time_kb() -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton(t, callback_data=f"m:settime:{t}") for t in TIME_PRESETS[i:i + 4]]
        for i in range(0, len(TIME_PRESETS), 4)
    ]
    rows.append([InlineKeyboardButton("✏️ Своё время", callback_data="m:asktime")])
    rows.append([InlineKeyboardButton("◀️ Назад", callback_data="m:settings")])
    return InlineKeyboardMarkup(rows)


def prompt_kb(custom: bool) -> InlineKeyboardMarkup:
    rows = [[InlineKeyboardButton("✏️ Новый промпт", callback_data="m:askprompt")]]
    if custom:
        rows.append([InlineKeyboardButton("♻️ Сбросить на стандартный", callback_data="m:resetprompt")])
    rows.append([InlineKeyboardButton("◀️ Назад", callback_data="m:settings")])
    return InlineKeyboardMarkup(rows)


async def cmd_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/menu — меню. В каналах, где не ответить реплаем на бота, можно и так:
    /menu time 21:30, /menu prompt <текст> (или ответом на сообщение с текстом)."""
    if not await guard(update, context):
        return
    chat = update.effective_chat
    msg = update.effective_message
    args = context.args or []
    if args and args[0].lower() in ("time", "время"):
        await msg.reply_text(apply_time(context.application, chat.id, " ".join(args[1:])))
        return
    if args and args[0].lower() in ("prompt", "промпт"):
        parts = msg.text.split(maxsplit=2)  # сохраняем переносы строк
        text = parts[2] if len(parts) > 2 else ""
        if not text.strip() and msg.reply_to_message and msg.reply_to_message.text:
            text = msg.reply_to_message.text
        await msg.reply_text(apply_prompt(chat.id, text))
        return
    await context.bot.send_message(chat.id, menu_text(chat.id), reply_markup=menu_kb())
    if chat.type == "channel":  # чтобы "/menu" не висел постом в канале
        try:
            await msg.delete()
        except Exception:
            pass


async def on_settings_reply(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """Ответ реплаем на запрос бота «пришли время/промпт». True — если это был он."""
    msg = update.effective_message
    target = msg.reply_to_message
    if not target or not target.from_user or target.from_user.id != context.bot.id:
        return False
    chat_id = update.effective_chat.id
    if target.text == ASK_TIME:
        await msg.reply_text(apply_time(context.application, chat_id, msg.text))
    elif target.text == ASK_PROMPT:
        await msg.reply_text(apply_prompt(chat_id, msg.text))
    else:
        return False
    return True


async def on_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    chat = update.effective_chat
    action = q.data.split(":", 2)[1:]
    if not get_chat(chat.id):
        await q.answer("Чат не найден, перезапусти /menu", show_alert=True)
        return

    async def show(text, kb):
        try:
            await q.edit_message_text(text, reply_markup=kb)
        except Exception:  # "message is not modified" и т.п.
            pass

    match action:
        case ["home"]:
            await q.answer()
            await show(menu_text(chat.id), menu_kb())
        case ["summary"]:
            await q.answer("Делаю саммари…")
            await summarize_chat(context.bot, chat.id, preview=True)
            await show(menu_text(chat.id), menu_kb())
        case ["settings"]:
            await q.answer()
            await show(settings_text(chat.id), settings_kb(chat.id))
        case ["time"]:
            await q.answer()
            s = get_chat(chat.id)
            await show(f"⏰ Сейчас: {s['summary_time']}\nВыбери время:", time_kb())
        case ["settime", value] if TIME_RE.match(value):
            update_chat(chat.id, "summary_time", value)
            schedule_chat(context.application, chat.id)
            await q.answer(f"Саммари теперь в {value}")
            await show(settings_text(chat.id), settings_kb(chat.id))
        case ["asktime"]:
            await q.answer()
            await context.bot.send_message(
                chat.id, ASK_TIME, reply_markup=ForceReply(input_field_placeholder="21:30")
            )
        case ["prompt"]:
            await q.answer()
            s = get_chat(chat.id)
            prompt = s["prompt"] or DEFAULT_PROMPT
            if len(prompt) > 1000:
                prompt = prompt[:1000] + "…"
            kind = "свой" if s["prompt"] else "по умолчанию"
            await show(f"📝 Промпт ({kind}):\n{prompt}", prompt_kb(bool(s["prompt"])))
        case ["askprompt"]:
            await q.answer()
            await context.bot.send_message(chat.id, ASK_PROMPT, reply_markup=ForceReply())
        case ["resetprompt"]:
            update_chat(chat.id, "prompt", None)
            await q.answer("Промпт сброшен")
            await show(settings_text(chat.id), settings_kb(chat.id))
        case ["links"]:
            on = 0 if get_chat(chat.id)["link_replies"] else 1
            update_chat(chat.id, "link_replies", on)
            await q.answer("Пересказ ссылок включён" if on else "Пересказ ссылок выключен")
            await show(settings_text(chat.id), settings_kb(chat.id))
        case ["wipe"]:
            await q.answer()
            await show(
                "🗑 Удалить все накопленные сообщения и всю память чата (итоги дней, недель, "
                "месяцев, лет)? Это не отменить. Настройки — время и промпт — останутся.",
                InlineKeyboardMarkup([
                    [InlineKeyboardButton("✅ Да, удалить", callback_data="m:wipe:yes")],
                    [InlineKeyboardButton("◀️ Отмена", callback_data="m:settings")],
                ]),
            )
        case ["wipe", "yes"]:
            wipe_chat_data(chat.id)
            log.info("Чат %s: данные удалены по кнопке (%s)", chat.id, q.from_user.id)
            await q.answer("Данные чата удалены", show_alert=True)
            await show(menu_text(chat.id), menu_kb())
        case _:
            await q.answer()


# ---------- запуск ----------

# меню команд по "/"; в каналах Telegram его не показывает, там вводим руками
GROUP_COMMANDS = [
    BotCommand("menu", "Меню: саммари, статистика, настройки"),
    BotCommand("summary", "Саммари на сейчас"),
    BotCommand("memory", "Что бот помнит о чате"),
    BotCommand("retry", "Перераспознать голосовые/картинки"),
]


async def post_init(app: Application):
    for s in all_chats():
        if s["active"]:
            schedule_chat(app, s["chat_id"])
    # в личке команд нет — бот работает только в группах и каналах
    await app.bot.delete_my_commands(scope=BotCommandScopeDefault())
    await app.bot.set_my_commands(GROUP_COMMANDS, scope=BotCommandScopeAllGroupChats())
    # убираем старый админский скоуп, он перекрывал бы групповой
    await app.bot.delete_my_commands(scope=BotCommandScopeAllChatAdministrators())


def main():
    app = Application.builder().token(TOKEN).post_init(post_init).build()

    # новые сообщения в группах и каналах (без правок)
    new_posts = filters.UpdateType.MESSAGE | filters.UpdateType.CHANNEL_POST
    chat_text = (
        new_posts
        & filters.TEXT
        & ~filters.COMMAND
        & (filters.ChatType.GROUPS | filters.ChatType.CHANNEL)
    )
    app.add_handler(MessageHandler(chat_text, on_message))
    chat_media = (
        new_posts
        & (
            filters.PHOTO
            | filters.Document.IMAGE
            | filters.VOICE
            | filters.VIDEO_NOTE
            | filters.Sticker.ALL
        )
        & (filters.ChatType.GROUPS | filters.ChatType.CHANNEL)
    )
    # block=False: распознавание долгое, не держим остальные апдейты
    app.add_handler(MessageHandler(chat_media, on_media, block=False))
    app.add_handler(ChatMemberHandler(on_my_member, ChatMemberHandler.MY_CHAT_MEMBER))
    app.add_handler(CallbackQueryHandler(on_menu, pattern=r"^m:"))

    commands = {
        "summary": cmd_summary,
        "menu": cmd_menu,
        "retry": cmd_retry,
        "memory": cmd_memory,
        "start": cmd_menu,  # кнопка Start: в группе — меню, в личке — «добавь в группу»
    }
    for name, handler in commands.items():
        app.add_handler(CommandHandler(name, handler, filters=new_posts))

    log.info("Бот запущен")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
