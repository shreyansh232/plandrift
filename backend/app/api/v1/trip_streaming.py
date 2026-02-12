"""Streaming trip API endpoints for token-by-token responses with phase tracking."""

import asyncio
import json
from typing import Annotated, AsyncGenerator, Callable
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.agent.agent import TravelAgent
from app.agent.models import Phase
from app.api.deps import get_current_user
from app.config import get_settings
from app.db.models import User

router = APIRouter(prefix="/trips", tags=["trips-streaming"])
settings = get_settings()


def _make_status_callback(status_queue: asyncio.Queue[str]) -> Callable[[str], None]:
    loop = asyncio.get_running_loop()

    def _cb(message: str) -> None:
        loop.call_soon_threadsafe(status_queue.put_nowait, message)

    return _cb


def _make_token_callback(token_queue: asyncio.Queue[str]) -> Callable[[str], None]:
    loop = asyncio.get_running_loop()

    def _cb(token: str) -> None:
        loop.call_soon_threadsafe(token_queue.put_nowait, token)

    return _cb


@router.post("/start-stream")
async def start_trip_stream(
    request: dict,
    current_user: Annotated[User, Depends(get_current_user)],
) -> StreamingResponse:
    """Start a new trip planning conversation with token streaming.

    Phase: CLARIFICATION
    """
    prompt = request.get("prompt", "")
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    token_queue: asyncio.Queue[str] = asyncio.Queue()
    status_queue: asyncio.Queue[str] = asyncio.Queue()
    phase_queue: asyncio.Queue[str] = asyncio.Queue()

    async def generator() -> AsyncGenerator[str, None]:
        agent = TravelAgent(
            api_key=settings.openrouter_api_key,
            on_status=_make_status_callback(status_queue),
            on_token=_make_token_callback(token_queue),
        )

        yield f"event: meta\ndata: {json.dumps({'phase': 'clarification', 'has_high_risk': False})}\n\n"

        loop = asyncio.get_event_loop()

        def run_stream():
            phase_queue.put_nowait(Phase.CLARIFICATION.value)
            for token in agent.start_stream(prompt):
                phase_queue.put_nowait(agent.state.phase.value)

        task = loop.run_in_executor(None, run_stream)

        try:
            current_phase = Phase.CLARIFICATION.value

            while not task.done():
                try:
                    new_phase = phase_queue.get_nowait()
                    if new_phase != current_phase:
                        current_phase = new_phase
                        yield f"event: meta\ndata: {json.dumps({'phase': current_phase, 'has_high_risk': False})}\n\n"
                except asyncio.QueueEmpty:
                    pass

                try:
                    status_msg = status_queue.get_nowait()
                    yield f"event: status\ndata: {json.dumps({'text': status_msg})}\n\n"
                except asyncio.QueueEmpty:
                    pass

                try:
                    token = token_queue.get_nowait()
                    yield f"event: token\ndata: {json.dumps({'text': token})}\n\n"
                except asyncio.QueueEmpty:
                    await asyncio.sleep(0.01)

            while not token_queue.empty():
                token = token_queue.get_nowait()
                yield f"event: token\ndata: {json.dumps({'text': token})}\n\n"

            yield "event: done\ndata: {}\n\n"

        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(generator(), media_type="text/event-stream")


@router.post("/{trip_id}/clarify-stream")
async def clarify_trip_stream(
    trip_id: UUID,
    request: dict,
    current_user: Annotated[User, Depends(get_current_user)],
) -> StreamingResponse:
    """Submit clarification answers with token streaming.

    Phase: FEASIBILITY
    """
    answers = request.get("answers", "")
    if not answers:
        raise HTTPException(status_code=400, detail="Answers are required")

    token_queue: asyncio.Queue[str] = asyncio.Queue()
    status_queue: asyncio.Queue[str] = asyncio.Queue()
    phase_queue: asyncio.Queue[str] = asyncio.Queue()

    async def generator() -> AsyncGenerator[str, None]:
        agent = TravelAgent(
            api_key=settings.openrouter_api_key,
            on_status=_make_status_callback(status_queue),
            on_token=_make_token_callback(token_queue),
        )

        yield f"event: meta\ndata: {json.dumps({'phase': 'feasibility', 'has_high_risk': False})}\n\n"

        loop = asyncio.get_event_loop()

        def run_stream():
            phase_queue.put_nowait(Phase.FEASIBILITY.value)
            for token in agent.clarify_stream(answers):
                phase_queue.put_nowait(agent.state.phase.value)

        task = loop.run_in_executor(None, run_stream)

        try:
            current_phase = Phase.FEASIBILITY.value

            while not task.done():
                try:
                    new_phase = phase_queue.get_nowait()
                    if new_phase != current_phase:
                        current_phase = new_phase
                        yield f"event: meta\ndata: {json.dumps({'phase': current_phase, 'has_high_risk': False})}\n\n"
                except asyncio.QueueEmpty:
                    pass

                try:
                    status_msg = status_queue.get_nowait()
                    yield f"event: status\ndata: {json.dumps({'text': status_msg})}\n\n"
                except asyncio.QueueEmpty:
                    pass

                try:
                    token = token_queue.get_nowait()
                    yield f"event: token\ndata: {json.dumps({'text': token})}\n\n"
                except asyncio.QueueEmpty:
                    await asyncio.sleep(0.01)

            while not token_queue.empty():
                token = token_queue.get_nowait()
                yield f"event: token\ndata: {json.dumps({'text': token})}\n\n"

            yield "event: done\ndata: {}\n\n"

        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(generator(), media_type="text/event-stream")


@router.post("/{trip_id}/plan-stream")
async def generate_plan_stream(
    trip_id: UUID,
    request: dict,
    current_user: Annotated[User, Depends(get_current_user)],
) -> StreamingResponse:
    """Generate the trip plan with token streaming.

    Phase: PLANNING
    """
    modifications = request.get("modifications")

    token_queue: asyncio.Queue[str] = asyncio.Queue()
    status_queue: asyncio.Queue[str] = asyncio.Queue()
    phase_queue: asyncio.Queue[str] = asyncio.Queue()

    async def generator() -> AsyncGenerator[str, None]:
        agent = TravelAgent(
            api_key=settings.openrouter_api_key,
            on_status=_make_status_callback(status_queue),
            on_token=_make_token_callback(token_queue),
        )

        yield f"event: meta\ndata: {json.dumps({'phase': 'planning', 'has_high_risk': False})}\n\n"

        loop = asyncio.get_event_loop()

        def run_stream():
            phase_queue.put_nowait(Phase.PLANNING.value)
            for token in agent.assumptions_stream(
                confirmed=True, modifications=modifications
            ):
                phase_queue.put_nowait(agent.state.phase.value)

        task = loop.run_in_executor(None, run_stream)

        try:
            current_phase = Phase.PLANNING.value

            while not task.done():
                try:
                    new_phase = phase_queue.get_nowait()
                    if new_phase != current_phase:
                        current_phase = new_phase
                        yield f"event: meta\ndata: {json.dumps({'phase': current_phase, 'has_high_risk': False})}\n\n"
                except asyncio.QueueEmpty:
                    pass

                try:
                    status_msg = status_queue.get_nowait()
                    yield f"event: status\ndata: {json.dumps({'text': status_msg})}\n\n"
                except asyncio.QueueEmpty:
                    pass

                try:
                    token = token_queue.get_nowait()
                    yield f"event: token\ndata: {json.dumps({'text': token})}\n\n"
                except asyncio.QueueEmpty:
                    await asyncio.sleep(0.01)

            while not token_queue.empty():
                token = token_queue.get_nowait()
                yield f"event: token\ndata: {json.dumps({'text': token})}\n\n"

            yield "event: done\ndata: {}\n\n"

        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(generator(), media_type="text/event-stream")


@router.post("/{trip_id}/refine-stream-token")
async def refine_plan_token_stream(
    trip_id: UUID,
    request: dict,
    current_user: Annotated[User, Depends(get_current_user)],
) -> StreamingResponse:
    """Refine the plan with token-by-token streaming.

    Phase: REFINEMENT
    """
    refinement_type = request.get("refinement_type", "")
    if not refinement_type:
        raise HTTPException(status_code=400, detail="Refinement type is required")

    token_queue: asyncio.Queue[str] = asyncio.Queue()
    status_queue: asyncio.Queue[str] = asyncio.Queue()
    phase_queue: asyncio.Queue[str] = asyncio.Queue()

    async def generator() -> AsyncGenerator[str, None]:
        agent = TravelAgent(
            api_key=settings.openrouter_api_key,
            on_status=_make_status_callback(status_queue),
            on_token=_make_token_callback(token_queue),
        )

        yield f"event: meta\ndata: {json.dumps({'phase': 'refinement', 'has_high_risk': False})}\n\n"

        loop = asyncio.get_event_loop()

        def run_stream():
            phase_queue.put_nowait(Phase.REFINEMENT.value)
            for token in agent.refine_stream(refinement_type):
                phase_queue.put_nowait(agent.state.phase.value)

        task = loop.run_in_executor(None, run_stream)

        try:
            current_phase = Phase.REFINEMENT.value

            while not task.done():
                try:
                    new_phase = phase_queue.get_nowait()
                    if new_phase != current_phase:
                        current_phase = new_phase
                        yield f"event: meta\ndata: {json.dumps({'phase': current_phase, 'has_high_risk': False})}\n\n"
                except asyncio.QueueEmpty:
                    pass

                try:
                    status_msg = status_queue.get_nowait()
                    yield f"event: status\ndata: {json.dumps({'text': status_msg})}\n\n"
                except asyncio.QueueEmpty:
                    pass

                try:
                    token = token_queue.get_nowait()
                    yield f"event: token\ndata: {json.dumps({'text': token})}\n\n"
                except asyncio.QueueEmpty:
                    await asyncio.sleep(0.01)

            while not token_queue.empty():
                token = token_queue.get_nowait()
                yield f"event: token\ndata: {json.dumps({'text': token})}\n\n"

            yield "event: done\ndata: {}\n\n"

        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(generator(), media_type="text/event-stream")
