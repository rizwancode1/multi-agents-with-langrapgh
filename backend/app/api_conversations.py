"""
Conversation API endpoints.
"""

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import selectinload

from app.auth import require_api_key
from app.db import db_session
from app.models_db import Conversation, ConversationMessage

router = APIRouter(prefix="/conversations", tags=["conversations"])


class MessageResponse(BaseModel):
    id: int
    role: str
    text: str
    status: str | None = None
    time: str
    created_at: str


class ConversationResponse(BaseModel):
    id: int
    title: str
    customer: str
    status: str
    thread_id: str | None = None
    messages: list[MessageResponse]
    created_at: str
    updated_at: str


class CreateConversationRequest(BaseModel):
    title: str
    customer: str = "Unassigned"
    status: str = "Open"


class AddMessageRequest(BaseModel):
    role: str
    text: str
    status: str | None = None


def _message_to_response(msg: ConversationMessage) -> MessageResponse:
    return MessageResponse(
        id=msg.id,
        role=msg.role,
        text=msg.text,
        status=msg.status,
        time=msg.created_at.strftime("%I:%M %p") if msg.created_at else "Just now",
        created_at=msg.created_at.isoformat() if msg.created_at else datetime.now(UTC).isoformat(),
    )


def _conversation_to_response(conv: Conversation) -> ConversationResponse:
    messages = sorted(conv.messages, key=lambda m: m.created_at or datetime.min.replace(tzinfo=timezone.utc))
    return ConversationResponse(
        id=conv.id,
        title=conv.title,
        customer=conv.customer_name,
        status=conv.status,
        thread_id=conv.thread_id,
        messages=[_message_to_response(m) for m in messages],
        created_at=conv.created_at.isoformat() if conv.created_at else datetime.now(UTC).isoformat(),
        updated_at=conv.updated_at.isoformat() if conv.updated_at else datetime.now(UTC).isoformat(),
    )


def get_conversation_thread_id(conversation_id: int) -> str | None:
    """Return the LangGraph thread id for a conversation.

    Persists the mapping on first use so old conversations get backfilled.
    Returns None if the conversation does not exist.
    """
    with db_session() as db:
        conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
        if not conv:
            return None
        if not conv.thread_id:
            conv.thread_id = f"conv-{conversation_id}"
        return conv.thread_id


def get_conversation_messages(conversation_id: int) -> list[dict]:
    """Return all persisted messages for a conversation as {role, text} dicts."""
    with db_session() as db:
        conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
        if not conv:
            return []
        messages = sorted(conv.messages, key=lambda m: m.created_at or datetime.min.replace(tzinfo=timezone.utc))
        return [{"role": m.role, "text": m.text} for m in messages]


def save_conversation_message(
    conversation_id: int,
    role: str,
    text: str,
    status: str | None = None,
) -> MessageResponse | None:
    """Persist a message to a conversation.

    Returns the saved message, or None if the conversation does not exist.
    On the first user message the conversation title is set from the message text.
    """
    with db_session() as db:
        conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
        if not conv:
            return None

        is_first_message = len(conv.messages) == 0
        msg = ConversationMessage(
            conversation_id=conversation_id,
            role=role,
            text=text,
            status=status,
        )
        db.add(msg)
        db.flush()
        db.refresh(msg)

        if is_first_message and role == "user":
            conv.title = text[:50]

        return _message_to_response(msg)


@router.get("/", response_model=list[ConversationResponse])
async def list_conversations(
    limit: int = Query(default=50, ge=1, le=200, description="Max conversations returned"),
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
):
    """List conversations (paginated), newest activity first."""
    with db_session() as db:
        conversations = (
            db.query(Conversation)
            .options(selectinload(Conversation.messages))
            .order_by(Conversation.updated_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )
        return [_conversation_to_response(c) for c in conversations]


@router.post("/", response_model=ConversationResponse)
async def create_conversation(body: CreateConversationRequest, _auth: None = Depends(require_api_key)):
    with db_session() as db:
        conv = Conversation(
            title=body.title,
            customer_name=body.customer,
            status=body.status,
        )
        db.add(conv)
        db.flush()
        conv.thread_id = f"conv-{conv.id}"
        db.refresh(conv)
        return _conversation_to_response(conv)


@router.get("/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(conversation_id: int):
    with db_session() as db:
        conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return _conversation_to_response(conv)


@router.post("/{conversation_id}/messages", response_model=MessageResponse)
async def add_message(conversation_id: int, body: AddMessageRequest, _auth: None = Depends(require_api_key)):
    with db_session() as db:
        conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found")

        msg = ConversationMessage(
            conversation_id=conversation_id,
            role=body.role,
            text=body.text,
            status=body.status,
        )
        db.add(msg)
        db.flush()
        db.refresh(msg)
        return _message_to_response(msg)
