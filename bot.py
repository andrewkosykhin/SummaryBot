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

ВАЖНО:
  * Группы: в @BotFather выполни /setprivacy -> Disable, иначе бот видит
    только команды. После этого удали и заново добавь бота в группу.
  * Каналы: бота нужно сделать админом канала.

Команды в чате (настройки у каждого чата свои, менять может любой участник):
    /menu                — меню на кнопках (удобно в канале)
    /settings            — текущие настройки чата
    /settime 21:30       — время ежедневного саммари
    /setprompt <текст>   — свой промпт (или ответом на сообщение с текстом)
    /summary             — сделать саммари прямо сейчас
    /stats               — сколько сообщений накоплено
    /retry [N]           — перераспознать последние N неудачных голосовых/картинок
                           (или ответь /retry на конкретное сообщение)
"""

import asyncio
import base64
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
    InlineKeyboardMarkup,
    Update,
)
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
    # миграция: нераспознанное медиа (JSON) и id сообщения в Telegram — для /retry
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(messages)")}
    if "media" not in cols:
        conn.execute("ALTER TABLE messages ADD COLUMN media TEXT")
        conn.execute("ALTER TABLE messages ADD COLUMN tg_msg_id INTEGER")
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
    assert field in ("summary_time", "prompt", "title")
    with db() as conn:
        conn.execute(f"UPDATE chats SET {field} = ? WHERE chat_id = ?", (value, chat_id))


def all_chats() -> list[sqlite3.Row]:
    with db() as conn:
        return conn.execute("SELECT * FROM chats").fetchall()


def forget_chat(chat_id: int):
    with db() as conn:
        conn.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
        conn.execute("DELETE FROM chats WHERE chat_id = ?", (chat_id,))


def save_message(
    chat_id: int,
    author: str,
    text: str,
    ts: datetime.datetime,
    media: dict | None = None,
    tg_msg_id: int | None = None,
):
    with db() as conn:
        conn.execute(
            "INSERT INTO messages (chat_id, author, text, ts, media, tg_msg_id)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (chat_id, author, text, ts.isoformat(),
             json.dumps(media) if media else None, tg_msg_id),
        )


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


async def make_summary(prompt: str, transcript: str) -> str:
    resp = await ai.messages.create(
        model=MODEL,
        max_tokens=16000,  # с запасом: часть уходит на размышления модели
        system=prompt,
        messages=[{"role": "user", "content": f"Вот переписка за день:\n\n{transcript}"}],
    )
    return "".join(b.text for b in resp.content if b.type == "text").strip()


async def send_long(bot: Bot, chat_id: int, text: str):
    for i in range(0, len(text), TG_MSG_LIMIT):
        await bot.send_message(chat_id, text[i : i + TG_MSG_LIMIT])


async def summarize_chat(bot: Bot, chat_id: int, silent_if_empty: bool = False):
    s = get_chat(chat_id)
    if not s:
        return
    rows = fetch_messages(chat_id)
    if not rows:
        if not silent_if_empty:
            await bot.send_message(chat_id, "Пока нечего саммарить — сообщений нет.")
        return
    try:
        summary = await make_summary(
            s["prompt"] or DEFAULT_PROMPT, build_transcript(rows)
        )
    except Exception as e:
        log.exception("Ошибка при запросе к ИИ для чата %s", chat_id)
        provider = quota_provider(e)
        if provider:
            text = (f"🪫 У меня кончились токены в {provider} — саммари не сделать. "
                    "Пополните баланс, сообщения не потеряются.")
        else:
            text = "ИИ не ответил, попробую в следующий раз."
        await bot.send_message(chat_id, text)
        return  # сообщения не удаляем — уйдут в следующее саммари
    await send_long(bot, chat_id, f"🗞 Итоги дня ({len(rows)} сообщ.):\n\n{summary}")
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
    save_message(chat.id, author, msg.text, msg.date)


# ---------- картинки и голосовые ----------

IMAGE_TYPES = ("image/jpeg", "image/png", "image/gif", "image/webp")
MAX_IMAGE_BYTES = 5 * 1024 * 1024  # лимит Claude на картинку
MAX_AUDIO_BYTES = 20 * 1024 * 1024  # больше Bot API не отдаёт


async def describe_image(data: bytes, media_type: str, prompt: str, caption: str) -> str:
    ask = (
        "Это картинка из чата. Опиши в 1–3 предложениях, что на ней: что изображено, "
        "текст на картинке, если это мем — в чём шутка. Добавь пару подходящих эмодзи. "
        "Описание пойдёт в дневное саммари."
    )
    if caption:
        ask += f"\nПодпись к картинке: {caption}"
    resp = await ai.messages.create(
        model=VISION_MODEL,
        max_tokens=4000,  # с запасом: часть уходит на размышления модели
        output_config={"effort": "low"},
        system=prompt,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64.standard_b64encode(data).decode(),
                    },
                },
                {"type": "text", "text": ask},
            ],
        }],
    )
    return "".join(b.text for b in resp.content if b.type == "text").strip()


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
        return spec | {"kind": "image", "file_id": p.file_id,
                       "media_type": "image/jpeg", "size": p.file_size}
    if msg.document and msg.document.mime_type in IMAGE_TYPES:
        d = msg.document
        return spec | {"kind": "image", "file_id": d.file_id,
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

    if kind == "image":
        if spec["size"] and spec["size"] > MAX_IMAGE_BYTES:
            return f"[картинка, слишком большая]{tail}", None, False
        try:
            f = await context.bot.get_file(spec["file_id"])
            data = bytes(await f.download_as_bytearray())
            desc = await describe_image(data, spec["media_type"], prompt, caption)
            return f"[картинка: {desc}]{tail}", f"🖼 {desc}", False
        except Exception as e:
            log.exception("Не смог описать картинку в чате %s", chat_id)
            notice = quota_notice(chat_id, e, "смотреть картинки", always_notify)
            return f"[картинка]{tail}", notice, True

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
        if created:
            schedule_chat(context.application, chat.id)
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
        unschedule_chat(context.application, chat.id)
        forget_chat(chat.id)
        log.info("Бота убрали из чата %s, данные удалены", chat.id)


async def cmd_settings(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await guard(update, context):
        return
    s = get_chat(update.effective_chat.id)
    prompt = s["prompt"] or DEFAULT_PROMPT
    if len(prompt) > 500:
        prompt = prompt[:500] + "…"
    kind = "свой" if s["prompt"] else "по умолчанию"
    await update.effective_message.reply_text(
        f"⏰ Время: {s['summary_time']}\n"
        f"📝 Промпт ({kind}):\n{prompt}"
    )


async def cmd_settime(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await guard(update, context):
        return
    msg = update.effective_message
    m = TIME_RE.match(context.args[0]) if context.args else None
    if not m:
        await msg.reply_text("Формат: /settime 21:30")
        return
    value = f"{int(m.group(1)):02d}:{m.group(2)}"
    update_chat(update.effective_chat.id, "summary_time", value)
    schedule_chat(context.application, update.effective_chat.id)
    await msg.reply_text(f"Ок, саммари теперь каждый день в {value}.")


async def cmd_setprompt(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await guard(update, context):
        return
    msg = update.effective_message
    parts = msg.text.split(maxsplit=1)  # сохраняем переносы строк
    text = parts[1].strip() if len(parts) > 1 else ""
    if not text and msg.reply_to_message and msg.reply_to_message.text:
        text = msg.reply_to_message.text.strip()
    if not text:
        await msg.reply_text(
            "Напиши промпт после команды: /setprompt Сделай саммари как пират\n"
            "или ответь командой /setprompt на сообщение с текстом промпта."
        )
        return
    if len(text) > MAX_PROMPT_CHARS:
        await msg.reply_text(f"Слишком длинно, максимум {MAX_PROMPT_CHARS} символов.")
        return
    update_chat(update.effective_chat.id, "prompt", text)
    await msg.reply_text("Ок, промпт обновлён.")


async def cmd_summary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await guard(update, context):
        return
    await summarize_chat(context.bot, update.effective_chat.id)


async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await guard(update, context):
        return
    n = len(fetch_messages(update.effective_chat.id))
    await update.effective_message.reply_text(f"Накоплено сообщений: {n}")


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text(__doc__.split("Команды в чате")[1].strip(" :\n"))


# ---------- меню на кнопках ----------

TIME_PRESETS = ["09:00", "12:00", "15:00", "18:00", "20:00", "21:00", "22:00", "23:00"]


def menu_text(chat_id: int) -> str:
    s = get_chat(chat_id)
    kind = "свой" if s["prompt"] else "по умолчанию"
    n = len(fetch_messages(chat_id))
    return (
        "🤖 Меню бота\n"
        f"⏰ Саммари каждый день в {s['summary_time']}\n"
        f"📝 Промпт: {kind}\n"
        f"💬 Накоплено сообщений: {n}"
    )


def menu_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🗞 Саммари сейчас", callback_data="m:summary")],
        [
            InlineKeyboardButton("⏰ Время", callback_data="m:time"),
            InlineKeyboardButton("📝 Промпт", callback_data="m:prompt"),
        ],
        [InlineKeyboardButton("🔄 Обновить", callback_data="m:home")],
    ])


def time_kb() -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton(t, callback_data=f"m:settime:{t}") for t in TIME_PRESETS[i:i + 4]]
        for i in range(0, len(TIME_PRESETS), 4)
    ]
    rows.append([InlineKeyboardButton("◀️ Назад", callback_data="m:home")])
    return InlineKeyboardMarkup(rows)


def prompt_kb(custom: bool) -> InlineKeyboardMarkup:
    rows = []
    if custom:
        rows.append([InlineKeyboardButton("♻️ Сбросить на стандартный", callback_data="m:resetprompt")])
    rows.append([InlineKeyboardButton("◀️ Назад", callback_data="m:home")])
    return InlineKeyboardMarkup(rows)


async def cmd_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await guard(update, context):
        return
    chat = update.effective_chat
    await context.bot.send_message(chat.id, menu_text(chat.id), reply_markup=menu_kb())
    if chat.type == "channel":  # чтобы "/menu" не висел постом в канале
        try:
            await update.effective_message.delete()
        except Exception:
            pass


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
            await summarize_chat(context.bot, chat.id)
            await show(menu_text(chat.id), menu_kb())
        case ["time"]:
            await q.answer()
            s = get_chat(chat.id)
            await show(
                f"⏰ Сейчас: {s['summary_time']}\nВыбери время или напиши /settime ЧЧ:ММ",
                time_kb(),
            )
        case ["settime", value] if TIME_RE.match(value):
            update_chat(chat.id, "summary_time", value)
            schedule_chat(context.application, chat.id)
            await q.answer(f"Саммари теперь в {value}")
            await show(menu_text(chat.id), menu_kb())
        case ["prompt"]:
            await q.answer()
            s = get_chat(chat.id)
            prompt = s["prompt"] or DEFAULT_PROMPT
            if len(prompt) > 1000:
                prompt = prompt[:1000] + "…"
            kind = "свой" if s["prompt"] else "по умолчанию"
            await show(
                f"📝 Промпт ({kind}):\n{prompt}\n\n"
                "Поменять: /setprompt <текст> (или ответом на сообщение с текстом).",
                prompt_kb(bool(s["prompt"])),
            )
        case ["resetprompt"]:
            update_chat(chat.id, "prompt", None)
            await q.answer("Промпт сброшен")
            await show(menu_text(chat.id), menu_kb())
        case _:
            await q.answer()


# ---------- запуск ----------

# меню команд по "/"; в каналах Telegram его не показывает, там вводим руками
GROUP_COMMANDS = [
    BotCommand("menu", "Меню на кнопках"),
    BotCommand("summary", "Саммари прямо сейчас"),
    BotCommand("settime", "Время ежедневного саммари"),
    BotCommand("setprompt", "Свой промпт для саммари"),
    BotCommand("settings", "Текущие настройки"),
    BotCommand("stats", "Сколько сообщений накоплено"),
    BotCommand("retry", "Перераспознать голосовые/картинки"),
    BotCommand("help", "Справка"),
]


async def post_init(app: Application):
    for s in all_chats():
        schedule_chat(app, s["chat_id"])
    await app.bot.set_my_commands(
        [BotCommand("help", "Справка")], scope=BotCommandScopeDefault()
    )
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
        "settings": cmd_settings,
        "settime": cmd_settime,
        "setprompt": cmd_setprompt,
        "summary": cmd_summary,
        "stats": cmd_stats,
        "help": cmd_help,
        "menu": cmd_menu,
        "retry": cmd_retry,
        "start": cmd_help,
    }
    for name, handler in commands.items():
        app.add_handler(CommandHandler(name, handler, filters=new_posts))

    log.info("Бот запущен")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
