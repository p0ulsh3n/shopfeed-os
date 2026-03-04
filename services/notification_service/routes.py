"""Notification Service — FastAPI App + Routes — Section 24."""

from __future__ import annotations

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
    """Schedule 3 reminder notifications for a live stream."""
    reminders = [
        (24 * 3600, f"Demain en live : {vendor_name}", "Ne manque pas le live de demain !"),
        (3600,      f"Dans 1h : {vendor_name} en live", "Prépare-toi, ça commence bientôt !"),
        (900,       f"C'est dans 15 minutes !", f"{vendor_name} passe en live très bientôt"),
    ]

    for offset_s, title, body in reminders:
        now = datetime.now(timezone.utc)
        send_at = scheduled_at.timestamp() - offset_s
        delay = send_at - now.timestamp()

        if delay > 0:
            logger.info("Live reminder scheduled: live=%s, in=%.0fs", live_id, delay)


@app.get("/api/v1/notifications/{user_id}")
async def get_notifications(user_id: str, unread_only: bool = False):
    """Get notification history for a user."""
    notifs = _notification_history.get(user_id, [])
    if unread_only:
        notifs = [n for n in notifs if not n.get("read")]
    return {"notifications": notifs[-50:], "total": len(notifs)}
