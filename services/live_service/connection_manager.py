"""WebSocket Connection Manager for Live Rooms — Section 07."""

from __future__ import annotations

from collections import defaultdict

from fastapi import WebSocket


class LiveConnectionManager:
    """Manages WebSocket connections for live rooms."""

    def __init__(self):
        self._rooms: dict[str, set[WebSocket]] = defaultdict(set)
        self._viewer_counts: dict[str, int] = defaultdict(int)

    async def connect(self, ws: WebSocket, room: str) -> None:
        await ws.accept()
        self._rooms[room].add(ws)
        if "viewer" in room:
            live_id = room.split("_")[1] if "_" in room else room
            self._viewer_counts[live_id] += 1

    def disconnect(self, ws: WebSocket, room: str) -> None:
        self._rooms[room].discard(ws)
        if "viewer" in room:
            live_id = room.split("_")[1] if "_" in room else room
            self._viewer_counts[live_id] = max(0, self._viewer_counts[live_id] - 1)

    async def broadcast(self, room: str, data: dict) -> None:
        dead: list[WebSocket] = []
        for ws in self._rooms.get(room, set()):
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._rooms[room].discard(ws)

    def get_viewer_count(self, live_id: str) -> int:
        return self._viewer_counts.get(live_id, 0)
