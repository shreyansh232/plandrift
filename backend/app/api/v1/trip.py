"""Trip planning API endpoints.

Thin route handlers that delegate all business logic to
``app.services.trip``.  This module owns only HTTP concerns:
request/response schemas, status codes, and dependency injection.

Conversation flow
-----------------
  POST /trips/start              → Create trip, start clarification
  POST /trips/{id}/clarify       → Answer questions → run feasibility
  POST /trips/{id}/proceed       → Confirm after risk check → assumptions
  POST /trips/{id}/assumptions   → Confirm assumptions → generate plan
  POST /trips/{id}/refine        → Refine the generated plan

CRUD
----
  GET    /trips                  → List user's trips
  GET    /trips/{id}             → Trip detail with latest version
  GET    /trips/{id}/versions    → Trip with full version history
  DELETE /trips/{id}             → Delete trip + versions + session
"""

from typing import Annotated, AsyncGenerator, Callable, Optional
import json
import asyncio
from uuid import UUID

from fastapi import APIRouter, Depends, status, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_db
from app.db.models import User
from app.schemas.trip import (
    AgentResponse,
    TripResponse,
    TripSummary,
    TripWithVersions,
    TripMessageResponse,
)
from app.services import trip as trip_service

router = APIRouter(prefix="/trips", tags=["trips"])


def _chunk_text(text: str, size: int = 8) -> list[str]:
    """Split text into smaller chunks for smoother streaming."""
    return [text[i : i + size] for i in range(0, len(text), size)]


def _make_status_callback(status_queue: asyncio.Queue[str]) -> Callable[[str], None]:
    loop = asyncio.get_running_loop()

    def _cb(message: str) -> None:
        loop.call_soon_threadsafe(status_queue.put_nowait, message)

    return _cb


async def _stream_agent_response(
    response: AgentResponse,
) -> AsyncGenerator[str, None]:
    meta = {
        "trip_id": str(response.trip_id) if response.trip_id else None,
        "version_id": str(response.version_id) if response.version_id else None,
        "phase": response.phase,
        "has_high_risk": response.has_high_risk,
    }
    yield f"event: meta\ndata: {json.dumps(meta)}\n\n"

    # Stream the response with minimal delay for token-like feel
    for chunk in _chunk_text(response.message):
        payload = json.dumps({"text": chunk})
        yield f"event: delta\ndata: {payload}\n\n"
        await asyncio.sleep(0.02)  # 20ms delay for smooth streaming

    yield "event: done\ndata: {}\n\n"


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class StartTripRequest(BaseModel):
    """Kick off a new trip planning conversation."""

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description=(
            "Natural-language travel request, e.g. "
            "'Plan a trip from Mumbai to Japan in March, 7 days, solo'"
        ),
    )


class ClarifyRequest(BaseModel):
    """Submit answers to clarification questions."""

    answers: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Answers to the clarification questions",
    )


class ProceedRequest(BaseModel):
    """Decide whether to continue after a high-risk feasibility check."""

    proceed: bool = Field(
        ..., description="True to proceed despite risks, False to reconsider"
    )


class AssumptionsRequest(BaseModel):
    """Confirm or adjust planning assumptions."""

    confirmed: bool = Field(..., description="Accept assumptions as-is")
    modifications: Optional[str] = Field(
        None, max_length=2000, description="Changes to assumptions"
    )
    additional_interests: Optional[str] = Field(
        None, max_length=1000, description="Extra interests to weave into the plan"
    )


class RefineRequest(BaseModel):
    """Ask for a specific refinement of the generated plan."""

    refinement_type: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="What to refine, e.g. 'make it cheaper' or 'add more hiking'",
    )


# ---------------------------------------------------------------------------
# Conversation endpoints (stateful, phase-by-phase)
# ---------------------------------------------------------------------------


@router.post(
    "/start", response_model=AgentResponse, status_code=status.HTTP_201_CREATED
)
async def start_trip(
    body: StartTripRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> AgentResponse:
    """Start a new trip planning conversation.

    The agent extracts origin / destination from the prompt.  If it can't,
    the response message asks for them and ``trip_id`` will be ``null`` —
    call this endpoint again with a more complete prompt.
    """
    return await trip_service.start_trip_conversation(db, current_user.id, body.prompt)


@router.post("/start/stream")
async def start_trip_stream(
    body: StartTripRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> StreamingResponse:
    # Early rate-limit check for free users to return a proper 429
    await trip_service._enforce_plan_limit(db, current_user.id)

    status_queue: asyncio.Queue[str] = asyncio.Queue()

    async def runner() -> AgentResponse:
        return await trip_service.start_trip_conversation(
            db,
            current_user.id,
            body.prompt,
            on_status=_make_status_callback(status_queue),
        )

    async def generator() -> AsyncGenerator[str, None]:
        task = asyncio.create_task(runner())
        try:
            while not task.done():
                try:
                    status_msg = await asyncio.wait_for(status_queue.get(), timeout=0.2)
                    yield f"event: status\ndata: {json.dumps({'text': status_msg})}\n\n"
                except asyncio.TimeoutError:
                    continue

            if task.exception():
                exc = task.exception()
                if isinstance(exc, HTTPException):
                    error_payload = json.dumps(
                        {"error": exc.detail, "status_code": exc.status_code}
                    )
                else:
                    error_payload = json.dumps({"error": str(exc), "status_code": 500})
                yield f"event: error\ndata: {error_payload}\n\n"
                return

            response = await task
            async for chunk in _stream_agent_response(response):
                yield chunk
        except Exception as e:
            error_payload = json.dumps({"error": str(e), "status_code": 500})
            yield f"event: error\ndata: {error_payload}\n\n"

    return StreamingResponse(generator(), media_type="text/event-stream")


@router.post("/{trip_id}/clarify", response_model=AgentResponse)
async def clarify_trip(
    trip_id: UUID,
    body: ClarifyRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> AgentResponse:
    """Submit answers to the clarification questions.

    Triggers the feasibility analysis.  The ``has_high_risk`` flag in the
    response tells the frontend whether to show a proceed/cancel dialog
    before calling ``/proceed``.
    """
    return await trip_service.submit_clarification(
        db, trip_id, current_user.id, body.answers
    )


@router.post("/{trip_id}/clarify/stream")
async def clarify_trip_stream(
    trip_id: UUID,
    body: ClarifyRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> StreamingResponse:
    status_queue: asyncio.Queue[str] = asyncio.Queue()

    async def runner() -> AgentResponse:
        return await trip_service.submit_clarification(
            db,
            trip_id,
            current_user.id,
            body.answers,
            on_status=_make_status_callback(status_queue),
        )

    async def generator() -> AsyncGenerator[str, None]:
        task = asyncio.create_task(runner())
        try:
            while not task.done():
                try:
                    status_msg = await asyncio.wait_for(status_queue.get(), timeout=0.2)
                    yield f"event: status\ndata: {json.dumps({'text': status_msg})}\n\n"
                except asyncio.TimeoutError:
                    continue

            if task.exception():
                exc = task.exception()
                if isinstance(exc, HTTPException):
                    error_payload = json.dumps(
                        {"error": exc.detail, "status_code": exc.status_code}
                    )
                else:
                    error_payload = json.dumps({"error": str(exc), "status_code": 500})
                yield f"event: error\ndata: {error_payload}\n\n"
                return

            response = await task
            async for chunk in _stream_agent_response(response):
                yield chunk
        except Exception as e:
            error_payload = json.dumps({"error": str(e), "status_code": 500})
            yield f"event: error\ndata: {error_payload}\n\n"

    return StreamingResponse(generator(), media_type="text/event-stream")


@router.post("/{trip_id}/proceed", response_model=AgentResponse)
async def proceed_trip(
    trip_id: UUID,
    body: ProceedRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> AgentResponse:
    """Proceed (or not) after feasibility and generate planning assumptions.

    * If the previous ``/clarify`` response had ``has_high_risk=true``,
      this confirms or rejects the risky trip.
    * If ``has_high_risk`` was ``false``, call with ``proceed=true`` to
      advance to the assumptions phase.
    """
    return await trip_service.proceed_after_feasibility(
        db, trip_id, current_user.id, body.proceed
    )


@router.post("/{trip_id}/proceed/stream")
async def proceed_trip_stream(
    trip_id: UUID,
    body: ProceedRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> StreamingResponse:
    status_queue: asyncio.Queue[str] = asyncio.Queue()

    async def runner() -> AgentResponse:
        return await trip_service.proceed_after_feasibility(
            db,
            trip_id,
            current_user.id,
            body.proceed,
            on_status=_make_status_callback(status_queue),
        )

    async def generator() -> AsyncGenerator[str, None]:
        task = asyncio.create_task(runner())
        try:
            while not task.done():
                try:
                    status_msg = await asyncio.wait_for(status_queue.get(), timeout=0.2)
                    yield f"event: status\ndata: {json.dumps({'text': status_msg})}\n\n"
                except asyncio.TimeoutError:
                    continue

            if task.exception():
                exc = task.exception()
                if isinstance(exc, HTTPException):
                    error_payload = json.dumps(
                        {"error": exc.detail, "status_code": exc.status_code}
                    )
                else:
                    error_payload = json.dumps({"error": str(exc), "status_code": 500})
                yield f"event: error\ndata: {error_payload}\n\n"
                return

            response = await task
            async for chunk in _stream_agent_response(response):
                yield chunk
        except Exception as e:
            error_payload = json.dumps({"error": str(e), "status_code": 500})
            yield f"event: error\ndata: {error_payload}\n\n"

    return StreamingResponse(generator(), media_type="text/event-stream")


@router.post("/{trip_id}/assumptions", response_model=AgentResponse)
async def confirm_assumptions(
    trip_id: UUID,
    body: AssumptionsRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> AgentResponse:
    """Confirm or adjust assumptions, then generate the full itinerary.

    This is the longest-running call — the agent researches current
    prices and builds a day-by-day plan with budget breakdown.
    """
    return await trip_service.confirm_trip_assumptions(
        db,
        trip_id,
        current_user.id,
        body.confirmed,
        modifications=body.modifications,
        additional_interests=body.additional_interests,
    )


@router.post("/{trip_id}/assumptions/stream")
async def confirm_assumptions_stream(
    trip_id: UUID,
    body: AssumptionsRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> StreamingResponse:
    status_queue: asyncio.Queue[str] = asyncio.Queue()

    async def runner() -> AgentResponse:
        return await trip_service.confirm_trip_assumptions(
            db,
            trip_id,
            current_user.id,
            body.confirmed,
            modifications=body.modifications,
            additional_interests=body.additional_interests,
            on_status=_make_status_callback(status_queue),
        )

    async def generator() -> AsyncGenerator[str, None]:
        task = asyncio.create_task(runner())
        try:
            while not task.done():
                try:
                    status_msg = await asyncio.wait_for(status_queue.get(), timeout=0.2)
                    yield f"event: status\ndata: {json.dumps({'text': status_msg})}\n\n"
                except asyncio.TimeoutError:
                    continue

            if task.exception():
                exc = task.exception()
                if isinstance(exc, HTTPException):
                    error_payload = json.dumps(
                        {"error": exc.detail, "status_code": exc.status_code}
                    )
                else:
                    error_payload = json.dumps({"error": str(exc), "status_code": 500})
                yield f"event: error\ndata: {error_payload}\n\n"
                return

            response = await task
            async for chunk in _stream_agent_response(response):
                yield chunk
        except Exception as e:
            error_payload = json.dumps({"error": str(e), "status_code": 500})
            yield f"event: error\ndata: {error_payload}\n\n"

    return StreamingResponse(generator(), media_type="text/event-stream")


@router.post("/{trip_id}/refine", response_model=AgentResponse)
async def refine_trip(
    trip_id: UUID,
    body: RefineRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> AgentResponse:
    """Request a refinement of the generated plan.

    Can be called multiple times. Each call updates the same TripVersion.
    """
    return await trip_service.refine_trip_plan(
        db, trip_id, current_user.id, body.refinement_type
    )


@router.post("/{trip_id}/refine/stream")
async def refine_trip_stream(
    trip_id: UUID,
    body: RefineRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> StreamingResponse:
    status_queue: asyncio.Queue[str] = asyncio.Queue()

    async def runner() -> AgentResponse:
        return await trip_service.refine_trip_plan(
            db,
            trip_id,
            current_user.id,
            body.refinement_type,
            on_status=_make_status_callback(status_queue),
        )

    async def generator() -> AsyncGenerator[str, None]:
        task = asyncio.create_task(runner())
        try:
            while not task.done():
                try:
                    status = await asyncio.wait_for(status_queue.get(), timeout=0.2)
                    payload = json.dumps({"text": status})
                    yield f"event: status\ndata: {payload}\n\n"
                except asyncio.TimeoutError:
                    continue

            # Check if task raised an exception
            if task.exception():
                exc = task.exception()
                if isinstance(exc, HTTPException):
                    error_payload = json.dumps(
                        {"error": exc.detail, "status_code": exc.status_code}
                    )
                else:
                    error_payload = json.dumps({"error": str(exc), "status_code": 500})
                yield f"event: error\ndata: {error_payload}\n\n"
                return

            response = await task
            async for chunk in _stream_agent_response(response):
                yield chunk
        except Exception as e:
            error_payload = json.dumps({"error": str(e), "status_code": 500})
            yield f"event: error\ndata: {error_payload}\n\n"

    return StreamingResponse(generator(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Token Streaming endpoints (character-by-character)
# ---------------------------------------------------------------------------


@router.post("/{trip_id}/clarify/token-stream")
async def clarify_trip_token_stream(
    trip_id: UUID,
    body: ClarifyRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> StreamingResponse:
    """Submit clarification answers with token-by-token streaming."""
    status_queue: asyncio.Queue[str] = asyncio.Queue()
    token_queue: asyncio.Queue[str] = asyncio.Queue[str]()

    async def generator() -> AsyncGenerator[str, None]:
        try:
            # Ownership and validity check
            await trip_service._get_user_trip(db, trip_id, current_user.id)
            agent = trip_service._get_agent(trip_id)
            version = await trip_service._latest_version(db, trip_id)

            # Send initial meta before streaming tokens
            meta = {
                "trip_id": str(trip_id),
                "version_id": str(version.id),
                "phase": agent.state.phase.value,
                "has_high_risk": False,
            }
            yield f"event: meta\ndata: {json.dumps(meta)}\n\n"

            loop = asyncio.get_running_loop()
            agent.on_status = _make_status_callback(status_queue)

            full_response = ""

            def run():
                nonlocal full_response
                for token in agent.process_clarification_stream(body.answers):
                    full_response += token
                    loop.call_soon_threadsafe(token_queue.put_nowait, token)

            task = asyncio.create_task(asyncio.to_thread(run))

            while not task.done() or not status_queue.empty() or not token_queue.empty():
                while not status_queue.empty():
                    status_msg = status_queue.get_nowait()
                    yield f"event: status\ndata: {json.dumps({'text': status_msg})}\n\n"

                while not token_queue.empty():
                    token = token_queue.get_nowait()
                    yield f"event: token\ndata: {json.dumps({'text': token})}\n\n"

                await asyncio.sleep(0.02)

            if task.exception():
                raise task.exception()

            # Finalize and persist
            await trip_service._persist_state(db, version, agent)
            await trip_service._store_message(
                db, trip_id, "user", body.answers, phase=agent.state.phase.value
            )
            await trip_service._store_message(
                db, trip_id, "assistant", full_response, phase=agent.state.phase.value
            )

            # Send FINAL meta with updated phase
            final_meta = {
                "trip_id": str(trip_id),
                "version_id": str(version.id),
                "phase": agent.state.phase.value,
                "has_high_risk": agent.state.phase.value == "feasibility" and agent.state.awaiting_confirmation,
            }
            yield f"event: meta\ndata: {json.dumps(final_meta)}\n\n"

            yield "event: done\ndata: {}\n\n"

        except Exception as e:
            error_payload = json.dumps({"error": str(e), "status_code": 500})
            yield f"event: error\ndata: {error_payload}\n\n"

    return StreamingResponse(generator(), media_type="text/event-stream")


@router.post("/{trip_id}/proceed/token-stream")
async def proceed_trip_token_stream(
    trip_id: UUID,
    body: ProceedRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> StreamingResponse:
    """Handle proceed decision with token-by-token streaming."""
    status_queue: asyncio.Queue[str] = asyncio.Queue()
    token_queue: asyncio.Queue[str] = asyncio.Queue[str]()

    async def generator() -> AsyncGenerator[str, None]:
        try:
            # Ownership and validity check
            await trip_service._get_user_trip(db, trip_id, current_user.id)
            agent = trip_service._get_agent(trip_id)
            version = await trip_service._latest_version(db, trip_id)

            loop = asyncio.get_running_loop()
            agent.on_status = _make_status_callback(status_queue)

            full_response = ""

            def run():
                nonlocal full_response
                # Call this BEFORE sending meta so the phase is updated
                for token in agent.confirm_proceed_stream(body.proceed):
                    full_response += token
                    loop.call_soon_threadsafe(token_queue.put_nowait, token)

            # Send initial meta AFTER update
            meta = {
                "trip_id": str(trip_id),
                "version_id": str(version.id),
                "phase": agent.state.phase.value,
                "has_high_risk": False,
            }
            yield f"event: meta\ndata: {json.dumps(meta)}\n\n"

            task = asyncio.create_task(asyncio.to_thread(run))

            while not task.done() or not status_queue.empty() or not token_queue.empty():
                while not status_queue.empty():
                    status_msg = status_queue.get_nowait()
                    yield f"event: status\ndata: {json.dumps({'text': status_msg})}\n\n"

                while not token_queue.empty():
                    token = token_queue.get_nowait()
                    yield f"event: token\ndata: {json.dumps({'text': token})}\n\n"

                await asyncio.sleep(0.02)

            if task.exception():
                raise task.exception()

            # Finalize and persist
            await trip_service._persist_state(db, version, agent)
            user_message = "Let's proceed anyway." if body.proceed else "Let me reconsider."
            await trip_service._store_message(
                db, trip_id, "user", user_message, phase=agent.state.phase.value
            )
            await trip_service._store_message(
                db, trip_id, "assistant", full_response, phase=agent.state.phase.value
            )

            # Send FINAL meta with updated phase
            final_meta = {
                "trip_id": str(trip_id),
                "version_id": str(version.id),
                "phase": agent.state.phase.value,
                "has_high_risk": False,
            }
            yield f"event: meta\ndata: {json.dumps(final_meta)}\n\n"

            yield "event: done\ndata: {}\n\n"

        except Exception as e:
            error_payload = json.dumps({"error": str(e), "status_code": 500})
            yield f"event: error\ndata: {error_payload}\n\n"

    return StreamingResponse(generator(), media_type="text/event-stream")


@router.post("/{trip_id}/assumptions/token-stream")
async def confirm_assumptions_token_stream(
    trip_id: UUID,
    body: AssumptionsRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> StreamingResponse:
    """Confirm assumptions with token-by-token streaming."""
    status_queue: asyncio.Queue[str] = asyncio.Queue[str]()
    token_queue: asyncio.Queue[str] = asyncio.Queue[str]()

    async def generator() -> AsyncGenerator[str, None]:
        try:
            # Ownership and validity check
            await trip_service._get_user_trip(db, trip_id, current_user.id)
            agent = trip_service._get_agent(trip_id)
            version = await trip_service._latest_version(db, trip_id)

            loop = asyncio.get_running_loop()
            agent.on_status = _make_status_callback(status_queue)

            full_response = ""

            def run():
                nonlocal full_response
                # Call this BEFORE sending meta so the phase is updated
                for token in agent.confirm_assumptions_stream(
                    confirmed=body.confirmed,
                    modifications=body.modifications,
                    additional_interests=body.additional_interests,
                ):
                    full_response += token
                    loop.call_soon_threadsafe(token_queue.put_nowait, token)

            # Send initial meta AFTER update
            meta = {
                "trip_id": str(trip_id),
                "version_id": str(version.id),
                "phase": agent.state.phase.value,
                "has_high_risk": False,
            }
            yield f"event: meta\ndata: {json.dumps(meta)}\n\n"

            task = asyncio.create_task(asyncio.to_thread(run))

            while not task.done() or not status_queue.empty() or not token_queue.empty():
                while not status_queue.empty():
                    status_msg = status_queue.get_nowait()
                    yield f"event: status\ndata: {json.dumps({'text': status_msg})}\n\n"

                while not token_queue.empty():
                    token = token_queue.get_nowait()
                    yield f"event: token\ndata: {json.dumps({'text': token})}\n\n"

                await asyncio.sleep(0.02)

            if task.exception():
                raise task.exception()

            # Finalize and persist
            await trip_service._persist_state(db, version, agent)

            if (
                body.confirmed
                and not body.modifications
                and not body.additional_interests
            ):
                user_message = "Assumptions look good."
            else:
                parts = []
                if body.modifications:
                    parts.append(body.modifications)
                if body.additional_interests:
                    parts.append(body.additional_interests)
                user_message = " ".join(parts).strip() or "Update assumptions."

            await trip_service._store_message(
                db, trip_id, "user", user_message, phase=agent.state.phase.value
            )
            await trip_service._store_message(
                db, trip_id, "assistant", full_response, phase=agent.state.phase.value
            )

            # Send FINAL meta with updated phase
            final_meta = {
                "trip_id": str(trip_id),
                "version_id": str(version.id),
                "phase": agent.state.phase.value,
                "has_high_risk": False,
            }
            yield f"event: meta\ndata: {json.dumps(final_meta)}\n\n"

            yield "event: done\ndata: {}\n\n"

        except Exception as e:
            error_payload = json.dumps({"error": str(e), "status_code": 500})
            yield f"event: error\ndata: {error_payload}\n\n"

    return StreamingResponse(generator(), media_type="text/event-stream")


@router.post("/{trip_id}/refine/token-stream")
async def refine_trip_token_stream(
    trip_id: UUID,
    body: RefineRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> StreamingResponse:
    """Refine plan with token-by-token streaming."""
    status_queue: asyncio.Queue[str] = asyncio.Queue[str]()
    token_queue: asyncio.Queue[str] = asyncio.Queue[str]()

    async def generator() -> AsyncGenerator[str, None]:
        try:
            # Ownership and validity check
            await trip_service._get_user_trip(db, trip_id, current_user.id)
            agent = trip_service._get_agent(trip_id)
            version = await trip_service._latest_version(db, trip_id)

            loop = asyncio.get_running_loop()
            agent.on_status = _make_status_callback(status_queue)

            full_response = ""

            def run():
                nonlocal full_response
                # Call this BEFORE sending meta so the phase is updated
                for token in agent.refine_plan_stream(body.refinement_type):
                    full_response += token
                    loop.call_soon_threadsafe(token_queue.put_nowait, token)

            # Send initial meta AFTER update
            meta = {
                "trip_id": str(trip_id),
                "version_id": str(version.id),
                "phase": agent.state.phase.value,
                "has_high_risk": False,
            }
            yield f"event: meta\ndata: {json.dumps(meta)}\n\n"

            task = asyncio.create_task(asyncio.to_thread(run))

            while not task.done() or not status_queue.empty() or not token_queue.empty():
                while not status_queue.empty():
                    status_msg = status_queue.get_nowait()
                    yield f"event: status\ndata: {json.dumps({'text': status_msg})}\n\n"

                while not token_queue.empty():
                    token = token_queue.get_nowait()
                    yield f"event: token\ndata: {json.dumps({'text': token})}\n\n"

                await asyncio.sleep(0.02)

            if task.exception():
                raise task.exception()

            # Finalize and persist
            await trip_service._persist_state(db, version, agent)
            await trip_service._store_message(
                db, trip_id, "user", body.refinement_type, phase=agent.state.phase.value
            )
            await trip_service._store_message(
                db, trip_id, "assistant", full_response, phase=agent.state.phase.value
            )

            # Send FINAL meta with updated phase
            final_meta = {
                "trip_id": str(trip_id),
                "version_id": str(version.id),
                "phase": agent.state.phase.value,
                "has_high_risk": False,
            }
            yield f"event: meta\ndata: {json.dumps(final_meta)}\n\n"

            yield "event: done\ndata: {}\n\n"

        except Exception as e:
            error_payload = json.dumps({"error": str(e), "status_code": 500})
            yield f"event: error\ndata: {error_payload}\n\n"

    return StreamingResponse(generator(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# CRUD endpoints (stateless, DB-only)
# ---------------------------------------------------------------------------


@router.get("", response_model=list[TripSummary])
async def list_trips(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[TripSummary]:
    """List all trips for the authenticated user (newest first)."""
    return await trip_service.list_user_trips(db, current_user.id)


@router.get("/{trip_id}", response_model=TripResponse)
async def get_trip(
    trip_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> TripResponse:
    """Get a single trip with its latest version data."""
    return await trip_service.get_trip_detail(db, trip_id, current_user.id)


@router.get("/{trip_id}/versions", response_model=TripWithVersions)
async def get_trip_versions(
    trip_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> TripWithVersions:
    """Get a trip with all its version history."""
    return await trip_service.get_trip_version_history(db, trip_id, current_user.id)


@router.get("/{trip_id}/messages", response_model=list[TripMessageResponse])
async def get_trip_messages(
    trip_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[TripMessageResponse]:
    """Get chat messages for a trip conversation."""
    messages = await trip_service.list_trip_messages(db, trip_id, current_user.id)
    return [TripMessageResponse.model_validate(m) for m in messages]


@router.delete("/{trip_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_trip(
    trip_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    """Delete a trip, all its versions, and any live agent session."""
    await trip_service.delete_user_trip(db, trip_id, current_user.id)
