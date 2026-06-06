"""
whatsapp.py — WhatsApp Cloud API send helpers (text, image, video)
Gym WhatsApp Agent
"""

import os
import logging
import httpx
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)

TOKEN = os.getenv("WHATSAPP_TOKEN", "")
PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID", "")
VERSION = os.getenv("GRAPH_API_VERSION", "v25.0")
BASE_URL = f"https://graph.facebook.com/{VERSION}/{PHONE_NUMBER_ID}"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}


def send_text(to: str, message: str) -> dict:
    """Send a plain text WhatsApp message."""
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": message},
    }
    resp = httpx.post(
        f"{BASE_URL}/messages",
        json=payload,
        headers=HEADERS,
        timeout=15,
    )
    if resp.is_error:
        log.error("[WHATSAPP] send_text failed: status=%s body=%s", resp.status_code, resp.text)
    resp.raise_for_status()
    return resp.json()


def upload_media(file_path: str, mime_type: str) -> str:
    """
    Upload a local media file to WhatsApp and return the media_id.
    mime_type examples: 'image/jpeg', 'image/png', 'video/mp4'
    """
    with open(file_path, "rb") as f:
        files = {
            "file": (os.path.basename(file_path), f, mime_type),
            "messaging_product": (None, "whatsapp"),
            "type": (None, mime_type),
        }
        resp = httpx.post(
            f"{BASE_URL}/media",
            files=files,
            headers={"Authorization": f"Bearer {TOKEN}"},
            timeout=30,
        )
    if resp.is_error:
        log.error("[WHATSAPP] upload_media failed: status=%s body=%s", resp.status_code, resp.text)
    resp.raise_for_status()
    data = resp.json()
    return data["id"]


def send_image(to: str, file_path: str, caption: str = "") -> dict:
    """Upload a local image and send it to the recipient."""
    # Determine MIME type from extension
    ext = os.path.splitext(file_path)[1].lower()
    mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
    mime_type = mime_map.get(ext, "image/jpeg")

    media_id = upload_media(file_path, mime_type)
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "image",
        "image": {"id": media_id, "caption": caption},
    }
    resp = httpx.post(
        f"{BASE_URL}/messages",
        json=payload,
        headers=HEADERS,
        timeout=15,
    )
    if resp.is_error:
        log.error("[WHATSAPP] send_image failed: status=%s body=%s", resp.status_code, resp.text)
    resp.raise_for_status()
    return resp.json()


def send_video(to: str, file_path: str, caption: str = "") -> dict:
    """Upload a local video and send it to the recipient."""
    media_id = upload_media(file_path, "video/mp4")
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "video",
        "video": {"id": media_id, "caption": caption},
    }
    resp = httpx.post(
        f"{BASE_URL}/messages",
        json=payload,
        headers=HEADERS,
        timeout=15,
    )
    if resp.is_error:
        log.error("[WHATSAPP] send_video failed: status=%s body=%s", resp.status_code, resp.text)
    resp.raise_for_status()
    return resp.json()
