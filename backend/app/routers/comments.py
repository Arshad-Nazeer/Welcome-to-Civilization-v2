"""
backend/app/routers/comments.py
-------------------------------
Comment history endpoints for retrieving previously scored comments.
Supports pagination and filtering by action badge (Red, Orange, Yellow, Green).
"""

from __future__ import annotations

import math
from typing import Any
from fastapi import APIRouter, HTTPException, Query, status

from ..db import get_comments, get_comment_by_id, delete_comment_by_id
from ..schemas import CommentsListResponse

router = APIRouter(prefix="/api/comments", tags=["Comments History"])


@router.get("", response_model=CommentsListResponse)
async def list_comments(
    page: int = Query(default=1, ge=1, description="Page number starting at 1"),
    limit: int = Query(default=20, ge=1, le=100, description="Items per page"),
    action: str | None = Query(default=None, description="Filter by action: Remove, Hide with warning, Flag for review, Allow"),
) -> CommentsListResponse:
    """
    List past moderated comments sorted newest first.
    Filterable by action.
    """
    skip = (page - 1) * limit
    items, total = await get_comments(limit=limit, skip=skip, action=action)
    total_pages = math.ceil(total / limit) if total > 0 else 1

    return CommentsListResponse(
        items=items,
        total=total,
        page=page,
        limit=limit,
        total_pages=total_pages,
    )


@router.get("/{comment_id}")
async def get_single_comment(comment_id: str) -> dict[str, Any]:
    """Retrieve full details of a single moderated comment by ID."""
    comment = await get_comment_by_id(comment_id)
    if not comment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Comment with ID '{comment_id}' not found.",
        )
    return comment


@router.delete("/{comment_id}")
async def delete_single_comment(comment_id: str) -> dict[str, Any]:
    """Permanently delete a comment from database history."""
    deleted = await delete_comment_by_id(comment_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Comment with ID '{comment_id}' not found.",
        )
    return {
        "success": True,
        "message": f"Comment '{comment_id}' permanently deleted.",
        "id": comment_id,
    }
