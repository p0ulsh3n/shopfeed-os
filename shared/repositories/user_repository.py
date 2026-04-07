"""
Repository — Users & UserProfiles
===================================
Toutes les opérations DB sur les users passent par ici.
AUCUN SQL brut — uniquement SQLAlchemy 2.0 Select/Insert/Update.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from shared.db.models.user import UserORM, UserFollowORM, UserProfileORM


class UserRepository:
    """CRUD repository pour les utilisateurs."""

    # ──────────────────────────── READ ────────────────────────────

    async def get_by_id(self, session: AsyncSession, user_id: uuid.UUID | str) -> UserORM | None:
        uid = uuid.UUID(str(user_id)) if isinstance(user_id, str) else user_id
        result = await session.execute(
            select(UserORM)
            .where(UserORM.id == uid, UserORM.deleted_at.is_(None))
        )
        return result.scalar_one_or_none()

    async def get_by_email(self, session: AsyncSession, email: str) -> UserORM | None:
        result = await session.execute(
            select(UserORM)
            .where(UserORM.email == email, UserORM.deleted_at.is_(None))
        )
        return result.scalar_one_or_none()

    async def get_by_phone(self, session: AsyncSession, phone: str) -> UserORM | None:
        result = await session.execute(
            select(UserORM)
            .where(UserORM.phone == phone, UserORM.deleted_at.is_(None))
        )
        return result.scalar_one_or_none()

    async def get_with_profile(
        self, session: AsyncSession, user_id: uuid.UUID | str
    ) -> UserORM | None:
        uid = uuid.UUID(str(user_id)) if isinstance(user_id, str) else user_id
        result = await session.execute(
            select(UserORM)
            .options(selectinload(UserORM.profile))
            .where(UserORM.id == uid, UserORM.deleted_at.is_(None))
        )
        return result.scalar_one_or_none()

    # ──────────────────────────── WRITE ───────────────────────────

    async def create(self, session: AsyncSession, data: dict[str, Any]) -> UserORM:
        user = UserORM(**data)
        session.add(user)
        await session.flush()  # Obtenir l'ID sans commit
        return user

    async def update(
        self,
        session: AsyncSession,
        user_id: uuid.UUID | str,
        updates: dict[str, Any],
    ) -> UserORM | None:
        uid = uuid.UUID(str(user_id)) if isinstance(user_id, str) else user_id
        await session.execute(
            update(UserORM)
            .where(UserORM.id == uid)
            .values(**updates)
        )
        await session.flush()
        return await self.get_by_id(session, uid)

    async def soft_delete(self, session: AsyncSession, user_id: uuid.UUID | str) -> None:
        uid = uuid.UUID(str(user_id)) if isinstance(user_id, str) else user_id
        await session.execute(
            update(UserORM)
            .where(UserORM.id == uid)
            .values(deleted_at=datetime.now(timezone.utc))
        )
        await session.flush()


class UserProfileRepository:
    """CRUD repository pour les profils ML (Intent Graph)."""

    async def get(self, session: AsyncSession, user_id: uuid.UUID | str) -> UserProfileORM | None:
        uid = uuid.UUID(str(user_id)) if isinstance(user_id, str) else user_id
        result = await session.execute(
            select(UserProfileORM).where(UserProfileORM.user_id == uid)
        )
        return result.scalar_one_or_none()

    async def upsert(
        self, session: AsyncSession, user_id: uuid.UUID | str, data: dict[str, Any]
    ) -> UserProfileORM:
        uid = uuid.UUID(str(user_id)) if isinstance(user_id, str) else user_id
        profile = await self.get(session, uid)
        if profile is None:
            profile = UserProfileORM(user_id=uid, **data)
            session.add(profile)
        else:
            for key, value in data.items():
                setattr(profile, key, value)
        await session.flush()
        return profile


class UserFollowRepository:
    """CRUD repository pour les follows (user → vendor)."""

    async def get_following(
        self, session: AsyncSession, user_id: uuid.UUID | str
    ) -> list[uuid.UUID]:
        uid = uuid.UUID(str(user_id)) if isinstance(user_id, str) else user_id
        result = await session.execute(
            select(UserFollowORM.vendor_id).where(UserFollowORM.user_id == uid)
        )
        return [row[0] for row in result.fetchall()]

    async def follow(
        self,
        session: AsyncSession,
        user_id: uuid.UUID | str,
        vendor_id: uuid.UUID | str,
    ) -> None:
        uid = uuid.UUID(str(user_id)) if isinstance(user_id, str) else user_id
        vid = uuid.UUID(str(vendor_id)) if isinstance(vendor_id, str) else vendor_id
        # Idempotent — ignore si déjà suivi
        existing = await session.execute(
            select(UserFollowORM).where(
                UserFollowORM.user_id == uid,
                UserFollowORM.vendor_id == vid,
            )
        )
        if existing.scalar_one_or_none() is None:
            session.add(UserFollowORM(user_id=uid, vendor_id=vid))
            await session.flush()

    async def unfollow(
        self,
        session: AsyncSession,
        user_id: uuid.UUID | str,
        vendor_id: uuid.UUID | str,
    ) -> None:
        uid = uuid.UUID(str(user_id)) if isinstance(user_id, str) else user_id
        vid = uuid.UUID(str(vendor_id)) if isinstance(vendor_id, str) else vendor_id
        await session.execute(
            delete(UserFollowORM).where(
                UserFollowORM.user_id == uid,
                UserFollowORM.vendor_id == vid,
            )
        )
        await session.flush()
