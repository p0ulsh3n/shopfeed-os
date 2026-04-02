"""Notification Service — FastAPI App + Routes — Section 24."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from datetime import datetime, timezone

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from .schemas import NotificationRequest, NotificationResponse

logger = logging.getLogger(__name__)

app = FastAPI(title="ShopFeed OS — Notification Service", version="1.0.0")

_user_connections: dict[str, set[WebSocket]] = defaultdict(set)
_notification_history: dict[str, list[dict]] = defaultdict(list)


# ──────────────────────────────────────────────────────────────
# In-App WebSocket connections
# ──────────────────────────────────────────────────────────────

@app.websocket("/ws/notifications/{user_id}")
async def notification_ws(websocket: WebSocket, user_id: str):
    """In-app notification WebSocket — receives real-time alerts."""
    await websocket.accept()
    _user_connections[user_id].add(websocket)
    logger.info("Notification WS connected: user=%s", user_id)

    try:
        while True:
            data = await websocket.receive_json()
            if data.get("ack"):
                notif_id = data.get("notification_id")
                logger.debug("Notification acked: %s", notif_id)
    except WebSocketDisconnect:
        _user_connections[user_id].discard(websocket)


# ──────────────────────────────────────────────────────────────
# Send Notification
# ──────────────────────────────────────────────────────────────

@app.post("/api/v1/notifications/send", response_model=NotificationResponse)
async def send_notification(req: NotificationRequest):
    """Send notification through specified channels."""
    import uuid
    notif_id = str(uuid.uuid4())
    delivered: list[str] = []

    if "push" in req.channels:
        success = await _send_push(req.user_id, req.title, req.body, req.data)
        if success:
            delivered.append("push")

    if "in_app" in req.channels:
        success = await _send_in_app(req.user_id, notif_id, req.title, req.body, req.data)
        if success:
            delivered.append("in_app")

    _notification_history[req.user_id].append({
        "id": notif_id,
        "type": req.notification_type,
        "title": req.title,
        "body": req.body,
        "read": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    })

    return NotificationResponse(
        notification_id=notif_id,
        delivered_to=delivered,
        status="sent" if delivered else "failed",
    )


async def _send_push(user_id: str, title: str, body: str, data: dict) -> bool:
    """Send FCM push notification. In production: call Firebase Cloud Messaging API."""
    logger.info("Push sent: user=%s, title=%s", user_id, title)
    return True


async def _send_in_app(user_id: str, notif_id: str, title: str, body: str, data: dict) -> bool:
    """Send via WebSocket to connected clients."""
    connections = _user_connections.get(user_id, set())
    if not connections:
        return False

    payload = {
        "event": "notification",
        "notification_id": notif_id,
        "title": title,
        "body": body,
        "data": data,
    }

    dead: list[WebSocket] = []
    delivered = False

    for ws in connections:
        try:
            await ws.send_json(payload)
            delivered = True
        except Exception:
            dead.append(ws)

    for ws in dead:
        connections.discard(ws)

    return delivered


# ──────────────────────────────────────────────────────────────
# Live Reminders — Section 24 (Scheduled Lives)
# ──────────────────────────────────────────────────────────────

async def schedule_live_reminders(live_id: str, vendor_name: str, scheduled_at: datetime, follower_ids: list[str]):
    """Schedule 3 reminder notifications for a live stream.

    BUG #S8 FIX: Previously calculated the delay for each reminder and
    then just logged it (no actual scheduling). The reminders at J-24h,
    J-1h, and J-15min were never sent.

    Fixed with asyncio.create_task + asyncio.sleep so each reminder fires
    at the right time without blocking the event loop.
    """
    reminders = [
        (24 * 3600, f"Demain en live : {vendor_name}", "Ne manque pas le live de demain !"),
        (3600,      f"Dans 1h : {vendor_name} en live", "Prépare-toi, ça commence bientôt !"),
        (900,       f"C'est dans 15 minutes !",          f"{vendor_name} passe en live très bientôt"),
    ]

    for offset_s, title, body in reminders:
        now = datetime.now(timezone.utc)
        send_at = scheduled_at.timestamp() - offset_s
        delay = send_at - now.timestamp()

        if delay > 0:
            # BUG #S8 FIX: actually schedule with asyncio.create_task
            asyncio.create_task(
                _send_live_reminder_after(
                    delay=delay,
                    live_id=live_id,
                    follower_ids=follower_ids,
                    title=title,
                    body=body,
                )
            )
            logger.info("Live reminder scheduled: live=%s, in=%.0fs, followers=%d", live_id, delay, len(follower_ids))


async def _send_live_reminder_after(
    delay: float,
    live_id: str,
    follower_ids: list[str],
    title: str,
    body: str,
) -> None:
    """Wait `delay` seconds, then send push + in-app to all followers."""
    await asyncio.sleep(delay)
    logger.info("Firing live reminder: live=%s, followers=%d", live_id, len(follower_ids))
    for user_id in follower_ids:
        try:
            await _send_push(user_id, title, body, {"live_id": live_id, "type": "live_reminder"})
            await _send_in_app(
                user_id,
                f"reminder_{live_id}_{int(delay)}",
                title, body,
                {"live_id": live_id, "type": "live_reminder"},
            )
        except Exception as e:
            logger.warning("Reminder delivery failed for user=%s: %s", user_id, e)


@app.get("/api/v1/notifications/{user_id}")
async def get_notifications(user_id: str, unread_only: bool = False):
    """Get notification history for a user."""
    notifs = _notification_history.get(user_id, [])
    if unread_only:
        notifs = [n for n in notifs if not n.get("read")]
    return {"notifications": notifs[-50:], "total": len(notifs)}


# ──────────────────────────────────────────────────────────────
# t.md §6: Pavlovian Conditioned Notifications
# ──────────────────────────────────────────────────────────────

# Pre-built narrative teaser templates
_NARRATIVE_TEASERS = [
    "Ton histoire est en pause… 📖",
    "Tu étais sur le point de découvrir quelque chose…",
    "Le prochain chapitre t'attend 🔥",
    "Ta capsule est presque complète…",
    "Quelque chose de spécial vient d'arriver pour toi",
]

# Social proof templates
_SOCIAL_PROOF_TEMPLATES = [
    "{count} personnes près de toi complètent cette capsule en ce moment",
    "{count} acheteurs dans ta zone viennent de craquer pour ça",
    "Tes voisins n'arrêtent pas de commander ce soir 🛒",
]


@app.post("/api/v1/notifications/schedule-smart")
async def schedule_smart_notification(req: NotificationRequest):
    """t.md §6 — Schedule a notification at the optimal circadian time.

    Uses temporal vulnerability scoring to pick the moment when
    the user's rational filter is lowest (typically late evening).
    The notification arrives when the brain is most receptive.
    """
    import uuid
    from datetime import timedelta

    notif_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)

    # Determine optimal send time based on notification type
    if req.notification_type in ("narrative_teaser", "desire_trigger"):
        # Narrative teasers work best late evening (21h-23h) or pre-wake (7h-8h)
        current_hour = now.hour
        if current_hour < 21:
            # Schedule for tonight
            delay_hours = 21 - current_hour + (30 / 60)  # 21:30
        else:
            # Schedule for tomorrow morning
            delay_hours = (7 - current_hour + 24) % 24 + (15 / 60)  # 7:15

        delay_s = delay_hours * 3600

    elif req.notification_type == "collection_reminder":
        # Collection reminders: send after 4-8 hours of inactivity
        delay_s = 4 * 3600 + (hash(req.user_id) % (4 * 3600))  # 4-8h, user-seeded

    elif req.notification_type == "social_proof_local":
        # Social proof: send during high-traffic windows (18h-22h)
        current_hour = now.hour
        if 18 <= current_hour <= 22:
            delay_s = 0  # Send immediately during peak
        else:
            delay_hours = (18 - current_hour + 24) % 24
            delay_s = delay_hours * 3600
    else:
        delay_s = 0  # Unknown type: send now

    if delay_s > 0:
        asyncio.create_task(
            _send_delayed_notification(
                delay=delay_s,
                user_id=req.user_id,
                notif_id=notif_id,
                title=req.title,
                body=req.body,
                data={**req.data, "type": req.notification_type},
            )
        )
        status = "scheduled"
    else:
        await _send_push(req.user_id, req.title, req.body, req.data)
        await _send_in_app(req.user_id, notif_id, req.title, req.body, req.data)
        status = "sent"

    _notification_history[req.user_id].append({
        "id": notif_id,
        "type": req.notification_type,
        "title": req.title,
        "body": req.body,
        "read": False,
        "created_at": now.isoformat(),
        "scheduled": status == "scheduled",
    })

    return {
        "notification_id": notif_id,
        "status": status,
        "scheduled_delay_s": delay_s if delay_s > 0 else None,
    }


@app.post("/api/v1/notifications/narrative-teaser/{user_id}")
async def send_narrative_teaser(user_id: str, collection_progress: float = 0.0):
    """t.md §6 — Send a narrative-continuation teaser.

    "Ton histoire est en pause" — triggers Zeigarnik tension + FOMO.
    Automatically picks the best teaser template and schedules at
    the optimal circadian time.
    """
    # Pick teaser template based on collection progress
    if collection_progress > 0.3:
        title = "Ta capsule est presque complète…"
        body = f"Tu as {int(collection_progress * 100)}% — il manque si peu pour finir ton histoire 🔥"
    else:
        idx = hash(user_id) % len(_NARRATIVE_TEASERS)
        title = _NARRATIVE_TEASERS[idx]
        body = "Reviens continuer là où tu t'es arrêté"

    req = NotificationRequest(
        user_id=user_id,
        notification_type="narrative_teaser",
        title=title,
        body=body,
        data={"collection_progress": collection_progress},
    )
    return await schedule_smart_notification(req)


@app.post("/api/v1/notifications/social-proof/{user_id}")
async def send_social_proof(user_id: str, nearby_count: int = 12):
    """t.md §6 — "X personnes près de toi complètent cette capsule".

    Local social proof triggers herd instinct + FOMO.
    """
    idx = hash(user_id) % len(_SOCIAL_PROOF_TEMPLATES)
    template = _SOCIAL_PROOF_TEMPLATES[idx]
    body = template.format(count=nearby_count)

    req = NotificationRequest(
        user_id=user_id,
        notification_type="social_proof_local",
        title="📍 Autour de toi",
        body=body,
        data={"nearby_buyers": nearby_count},
    )
    return await schedule_smart_notification(req)


async def _send_delayed_notification(
    delay: float,
    user_id: str,
    notif_id: str,
    title: str,
    body: str,
    data: dict,
) -> None:
    """Wait `delay` seconds, then send push + in-app."""
    await asyncio.sleep(delay)
    logger.info("Firing scheduled notification: user=%s, type=%s", user_id, data.get("type"))
    await _send_push(user_id, title, body, data)
    await _send_in_app(user_id, notif_id, title, body, data)
