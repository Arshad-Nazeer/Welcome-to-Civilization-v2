"""
backend/app/db.py
-----------------
MongoDB connection management using Motor (AsyncIOMotorClient).
Provides robust, async database operations for comments history with
graceful in-memory fallback if the database connection is interrupted.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any
import uuid

import motor.motor_asyncio
from pymongo.errors import PyMongoError, ServerSelectionTimeoutError

from .config import settings

logger = logging.getLogger("wtc.db")

_client: motor.motor_asyncio.AsyncIOMotorClient | None = None
_db: motor.motor_asyncio.AsyncIOMotorDatabase | None = None
_is_connected: bool = False

# Fallback in-memory store for resilience if MongoDB is temporarily unreachable
_in_memory_comments: list[dict[str, Any]] = []


async def connect_to_mongo() -> None:
    """Initialize MongoDB connection pool on startup."""
    global _client, _db, _is_connected
    uri = settings.effective_mongodb_uri
    logger.info("Connecting to MongoDB at: %s", uri.split("@")[-1] if "@" in uri else uri)

    try:
        _client = motor.motor_asyncio.AsyncIOMotorClient(
            uri,
            serverSelectionTimeoutMS=settings.MONGODB_TIMEOUT_MS,
        )
        # Test connection with ping
        await _client.admin.command("ping")
        _db = _client[settings.MONGODB_DB_NAME]
        _is_connected = True
        logger.info("Connected to MongoDB successfully! Database: '%s'", settings.MONGODB_DB_NAME)

        # Ensure index on timestamp and action for fast pagination & filtering
        await _db.comments.create_index([("timestamp", -1)])
        await _db.comments.create_index([("headline.action", 1)])
        await _db.comments.create_index([("id", 1)], unique=True)
    except (ServerSelectionTimeoutError, PyMongoError, Exception) as exc:
        _is_connected = False
        logger.warning(
            "MongoDB connection failed: %s. Operating with in-memory fallback store.",
            exc,
        )


async def close_mongo_connection() -> None:
    """Close MongoDB connection pool on shutdown."""
    global _client, _db, _is_connected
    if _client is not None:
        _client.close()
        _is_connected = False
        logger.info("Closed MongoDB connection.")


def is_db_connected() -> bool:
    """Check if MongoDB is actively connected."""
    return _is_connected


async def save_comment(comment_doc: dict[str, Any]) -> str:
    """
    Save comment prediction record to MongoDB 'comments' collection.
    Stores uncensored original text, timestamps, policy version, and per-model results.
    """
    if "id" not in comment_doc or not comment_doc["id"]:
        comment_doc["id"] = uuid.uuid4().hex[:16]

    if "timestamp" not in comment_doc:
        comment_doc["timestamp"] = datetime.now(timezone.utc).isoformat()

    doc_id = comment_doc["id"]

    if _is_connected and _db is not None:
        try:
            # Upsert by id
            await _db.comments.replace_one({"id": doc_id}, comment_doc, upsert=True)
            return doc_id
        except Exception as exc:
            logger.error("Failed to write comment to MongoDB: %s. Saving to in-memory fallback.", exc)

    # In-memory store fallback
    # Remove existing if any, then insert at front (newest first)
    _in_memory_comments[:] = [c for c in _in_memory_comments if c.get("id") != doc_id]
    _in_memory_comments.insert(0, comment_doc)
    return doc_id


async def get_comment_by_id(comment_id: str) -> dict[str, Any] | None:
    """Fetch single comment record by unique ID."""
    if _is_connected and _db is not None:
        try:
            doc = await _db.comments.find_one({"id": comment_id}, {"_id": 0})
            if doc:
                return doc
        except Exception as exc:
            logger.error("Error retrieving comment %s from MongoDB: %s", comment_id, exc)

    # Search in-memory store
    for c in _in_memory_comments:
        if c.get("id") == comment_id:
            # Return copy without Mongo internal fields
            res = dict(c)
            res.pop("_id", None)
            return res
    return None


async def delete_comment_by_id(comment_id: str) -> bool:
    """
    Permanently delete a comment from MongoDB and in-memory store by ID.
    Returns True if a document was deleted, False if not found.
    """
    deleted = False
    if _is_connected and _db is not None:
        try:
            res = await _db.comments.delete_one({"id": comment_id})
            if res.deleted_count > 0:
                deleted = True
        except Exception as exc:
            logger.error("Error deleting comment %s from MongoDB: %s", comment_id, exc)

    # Also remove from in-memory fallback store if present
    global _in_memory_comments
    initial_len = len(_in_memory_comments)
    _in_memory_comments = [c for c in _in_memory_comments if c.get("id") != comment_id]
    if len(_in_memory_comments) < initial_len:
        deleted = True

    return deleted


async def get_comments(
    limit: int = 50,
    skip: int = 0,
    action: str | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """
    Retrieve paginated list of comments, newest first, with optional action filter.
    Returns (items, total_count).
    """
    if _is_connected and _db is not None:
        try:
            query: dict[str, Any] = {}
            if action:
                # Match headline.action (case-insensitive or exact)
                query["headline.action"] = {"$regex": f"^{action}$", "$options": "i"}

            total = await _db.comments.count_documents(query)
            cursor = (
                _db.comments.find(query, {"_id": 0})
                .sort("timestamp", -1)
                .skip(skip)
                .limit(limit)
            )
            items = await cursor.to_list(length=limit)
            return items, total
        except Exception as exc:
            logger.error("Error querying comments from MongoDB: %s. Falling back to in-memory.", exc)

    # In-memory query
    filtered = _in_memory_comments
    if action:
        act_lower = action.lower()
        filtered = [
            c for c in _in_memory_comments
            if c.get("headline", {}).get("action", "").lower() == act_lower
        ]

    total = len(filtered)
    items = [dict(c) for c in filtered[skip : skip + limit]]
    for item in items:
        item.pop("_id", None)
    return items, total
