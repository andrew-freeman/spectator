"""FastAPI application exposing the hierarchical reasoning supervisor."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol

from fastapi import Body, Depends, FastAPI, Form, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from app.actor.actor_runner import ActorRunner
from app.critic.critic_runner import CriticRunner
from app.governor.governor_logic import arbitrate
from app.meta.meta_actor_runner import MetaActorRunner, MetaActorOutput
from app.meta.meta_critic_runner import MetaCriticRunner, MetaCriticOutput
from app.meta.meta_governor_logic import evaluate_meta_cycle

from .command_interpreter import CommandInterpreter
from .memory_manager import MemoryManager
from .state_manager import StateManager
from .tool_executor import ToolExecutor

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
COG_PARAMS_PATH = CONFIG_DIR / "cog_params.json"
SYSTEM_LIMITS_PATH = CONFIG_DIR / "system_limits.json"
SENSITIVE_FIELD_TOKENS = ("api_key", "apikey", "token", "secret", "password")


class LLMClient(Protocol):
    def generate(self, prompt: str, *, stop: Optional[Iterable[str]] = None) -> str:
        ...


class CycleRequest(BaseModel):
    objectives: List[str] = Field(..., description="Ordered objectives for this cycle")
    context: Dict[str, Any] = Field(default_factory=dict)
    memory: List[str] = Field(default_factory=list)


class CycleResponse(BaseModel):
    cycle: int
    actor: Dict[str, Any]
    critic: Dict[str, Any]
    governor: Dict[str, Any]
    tool_results: List[Dict[str, Any]]
    meta: Optional[Dict[str, Any]]
    state: Dict[str, Any]


class CommandRequest(BaseModel):
    message: str = Field(..., description="Natural language instruction for the interpreter")


class ReasoningSupervisor:
    """Coordinates actor, critic, governor, and meta layers."""

    def __init__(
        self,
        actor_runner: ActorRunner,
        critic_runner: CriticRunner,
        meta_actor_runner: MetaActorRunner,
        meta_critic_runner: MetaCriticRunner,
        state_manager: StateManager,
        memory_manager: MemoryManager,
        tool_executor: ToolExecutor,
        cog_params: Dict[str, Any],
        system_limits: Dict[str, Any],
        config_path: Path = COG_PARAMS_PATH,
    ) -> None:
        self.actor_runner = actor_runner
        self.critic_runner = critic_runner
        self.meta_actor_runner = meta_actor_runner
        self.meta_critic_runner = meta_critic_runner
        self.state_manager = state_manager
        self.memory_manager = memory_manager
        self.tool_executor = tool_executor
        self.cog_params = cog_params
        self.system_limits = system_limits
        self._config_path = config_path
        self._meta_cycle_index = 0
        self._meta_frequency = int(self.cog_params.get("meta_frequency", 3) or 3)

    def run_cycle(
        self,
        objectives: List[str],
        context: Optional[Dict[str, Any]] = None,
        memory_snippets: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        actor_output = self.actor_runner.run(objectives, context=context or {}, memory_snippets=memory_snippets)
        actor_payload = asdict(actor_output)
        critic_output = self.critic_runner.run(actor_payload)
        decision = arbitrate(actor_output, critic_output)

        tool_results: List[Dict[str, Any]] = []
        if decision.verdict not in {"request_more_data", "defer_to_critic"}:
            tool_results = self.tool_executor.execute_many(decision.tool_calls)

        cycle_record = {
            "objectives": objectives,
            "actor": actor_payload,
            "critic": asdict(critic_output),
            "governor": asdict(decision),
            "tool_results": tool_results,
        }
        self.state_manager.log_cycle(cycle_record)

        meta_summary = self._maybe_run_meta_layer()
        self.state_manager.update_last_cycle({"meta": meta_summary})

        response = {
            "cycle": self.state_manager.cycle_index - 1,
            "actor": actor_payload,
            "critic": asdict(critic_output),
            "governor": asdict(decision),
            "tool_results": tool_results,
            "meta": meta_summary,
            "state": self.state_manager.read(),
        }
        return response

    def _maybe_run_meta_layer(self) -> Optional[Dict[str, Any]]:
        if self.state_manager.cycle_index % self._meta_frequency != 0:
            return None

        history = self.state_manager.history()
        recent_decisions = history[-self._meta_frequency :]
        meta_actor_output = self.meta_actor_runner.run(
            current_params=self.cog_params,
            recent_decisions=recent_decisions,
            meta_cycle=self._meta_cycle_index,
            system_limits=self.system_limits,
        )
        meta_actor_payload = _serialize_meta_actor(meta_actor_output)
        meta_critic_output = self.meta_critic_runner.run(meta_actor_payload, self.cog_params)
        decision = evaluate_meta_cycle(meta_actor_output, meta_critic_output, self.cog_params)

        if decision.decision == "apply":
            self.cog_params = decision.updated_params
            self._persist_cog_params()
        if decision.decision in {"apply", "noop"}:
            self._meta_cycle_index += 1

        summary = {
            "meta_actor": meta_actor_payload,
            "meta_critic": asdict(meta_critic_output),
            "meta_governor": asdict(decision),
        }
        return summary

    def _persist_cog_params(self) -> None:
        try:
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            with self._config_path.open("w", encoding="utf-8") as handle:
                json.dump(self.cog_params, handle, indent=2)
        except OSError as exc:  # pragma: no cover - filesystem safeguard
            LOGGER.warning("Failed to persist cog params: %s", exc)


def _serialize_meta_actor(meta_actor_output: MetaActorOutput) -> Dict[str, Any]:
    adjustments = {
        name: {"delta": value.delta, "justification": value.justification}
        for name, value in meta_actor_output.parameter_adjustments.items()
    }
    return {
        "meta_thoughts": meta_actor_output.meta_thoughts,
        "cognitive_strategy": meta_actor_output.cognitive_strategy,
        "parameter_adjustments": adjustments,
        "meta_improvements": meta_actor_output.meta_improvements,
        "assumptions": meta_actor_output.assumptions,
    }


def load_json_config(path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
    if not path.exists():
        return dict(default)
    try:
        return json.loads(path.read_text(encoding="utf-8") or "{}") or dict(default)
    except json.JSONDecodeError:
        LOGGER.warning("Invalid JSON in %s; using defaults", path)
        return dict(default)


app = FastAPI(title="Self-Reflective LLM Mind")
app.state.supervisor = None
app.state.command_interpreter = None
app.state.state_manager = StateManager()
app.state.memory_manager = MemoryManager()
app.state.tool_executor = ToolExecutor(app.state.state_manager, app.state.memory_manager)
app.state.cog_params = load_json_config(COG_PARAMS_PATH, {"meta_frequency": 3, "debug_enabled": False})
app.state.system_limits = load_json_config(SYSTEM_LIMITS_PATH, {})

templates = Jinja2Templates(directory="app/ui/templates")
app.mount("/static", StaticFiles(directory="app/ui/static"), name="static")


def configure_supervisor(
    *,
    actor_client: LLMClient,
    critic_client: Optional[LLMClient] = None,
    meta_actor_client: Optional[LLMClient] = None,
    meta_critic_client: Optional[LLMClient] = None,
) -> None:
    """Initialise the global supervisor using provided LLM clients."""

    critic_client = critic_client or actor_client
    meta_actor_client = meta_actor_client or actor_client
    meta_critic_client = meta_critic_client or critic_client

    supervisor = ReasoningSupervisor(
        actor_runner=ActorRunner(actor_client),
        critic_runner=CriticRunner(critic_client),
        meta_actor_runner=MetaActorRunner(meta_actor_client),
        meta_critic_runner=MetaCriticRunner(meta_critic_client),
        state_manager=app.state.state_manager,
        memory_manager=app.state.memory_manager,
        tool_executor=app.state.tool_executor,
        cog_params=app.state.cog_params,
        system_limits=app.state.system_limits,
    )
    app.state.supervisor = supervisor
    app.state.command_interpreter = CommandInterpreter(actor_client)


def get_supervisor() -> ReasoningSupervisor:
    supervisor: Optional[ReasoningSupervisor] = getattr(app.state, "supervisor", None)
    if supervisor is None:
        raise HTTPException(status_code=503, detail="Supervisor not configured. Call configure_supervisor first.")
    return supervisor


def get_command_interpreter() -> CommandInterpreter:
    interpreter: Optional[CommandInterpreter] = getattr(app.state, "command_interpreter", None)
    if interpreter is None:
        raise HTTPException(status_code=503, detail="Command interpreter not configured.")
    return interpreter


@app.get("/health")
def healthcheck() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def dashboard(request: Request, supervisor: ReasoningSupervisor = Depends(get_supervisor)):
    state = supervisor.state_manager.read()
    history = supervisor.state_manager.history()
    gpu_temps = app.state.tool_executor._read_gpu_temps()
    return templates.TemplateResponse(
        "dashboard.html",
        {"request": request, "state": state, "history": history, "gpu_temps": gpu_temps},
    )


@app.post("/run-cycle", response_model=CycleResponse)
def run_cycle(request: CycleRequest, supervisor: ReasoningSupervisor = Depends(get_supervisor)) -> CycleResponse:
    result = supervisor.run_cycle(request.objectives, context=request.context, memory_snippets=request.memory)
    return CycleResponse(**result)


@app.post("/hx/run-cycle")
def hx_run_cycle(
    cycle_request: CycleRequest,
    request: Request,
    supervisor: ReasoningSupervisor = Depends(get_supervisor),
):
    result = supervisor.run_cycle(
        cycle_request.objectives,
        context=cycle_request.context,
        memory_snippets=cycle_request.memory,
    )
    return templates.TemplateResponse("cycle_row.html", {"request": request, "result": result})


@app.get("/history")
def get_history(supervisor: ReasoningSupervisor = Depends(get_supervisor)) -> Dict[str, Any]:
    return {"history": supervisor.state_manager.history()}


@app.post("/command")
async def command_endpoint(
    request: Request,
    payload: Optional[CommandRequest] = Body(default=None),
    message: Optional[str] = Form(default=None),
    supervisor: ReasoningSupervisor = Depends(get_supervisor),
    interpreter: CommandInterpreter = Depends(get_command_interpreter),
):
    user_message = (payload.message if payload else None) or message
    if not user_message:
        raise HTTPException(status_code=422, detail="Missing command message")

    structured = interpreter.interpret(user_message)
    result = supervisor.run_cycle(
        structured.get("objectives", []),
        context=structured.get("context", {}),
        memory_snippets=structured.get("memory_snippets", []),
    )

    if request.headers.get("hx-request") == "true":
        return templates.TemplateResponse("cycle_row.html", {"request": request, "result": result})
    return CycleResponse(**result)


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    return any(token in lowered for token in SENSITIVE_FIELD_TOKENS)


def _sanitize_snapshot(data: Any) -> Any:
    if isinstance(data, dict):
        sanitized: Dict[str, Any] = {}
        for key, value in data.items():
            if _is_sensitive_key(key):
                continue
            sanitized[key] = _sanitize_snapshot(value)
        return sanitized
    if isinstance(data, list):
        return [_sanitize_snapshot(item) for item in data]
    return data


@app.get("/debug-system")
def debug_system(supervisor: ReasoningSupervisor = Depends(get_supervisor)) -> Dict[str, Any]:
    history = supervisor.state_manager.history()
    last_cycle = history[-1] if history else {}

    actor_snapshot = _sanitize_snapshot(last_cycle.get("actor", {}))
    critic_snapshot = _sanitize_snapshot(last_cycle.get("critic", {}))
    governor_snapshot = _sanitize_snapshot(last_cycle.get("governor", {}))
    meta_snapshot = _sanitize_snapshot(last_cycle.get("meta")) if last_cycle.get("meta") is not None else None

    state_snapshot = _sanitize_snapshot(supervisor.state_manager.read())
    cog_params_with_flag = dict(supervisor.cog_params)
    cog_params_with_flag.setdefault("debug_enabled", False)
    cog_params_snapshot = _sanitize_snapshot(cog_params_with_flag)

    return {
        "cycle": supervisor.state_manager.cycle_index - 1,
        "actor": actor_snapshot,
        "critic": critic_snapshot,
        "governor": governor_snapshot,
        "meta": meta_snapshot,
        "state": state_snapshot,
        "cog_params": cog_params_snapshot,
    }


@app.on_event("startup")
async def auto_cycle_loop() -> None:
    import asyncio

    supervisor = app.state.supervisor

    if not supervisor:
        LOGGER.warning("Supervisor not configured at startup.")
        return

    async def _loop() -> None:
        while True:
            try:
                supervisor.run_cycle(
                    objectives=["Monitor and stabilize GPU thermal state"],
                    context={},
                    memory_snippets=[],
                )
            except Exception:  # pragma: no cover - runtime safeguard
                LOGGER.exception("Automatic cycle failure")
            await asyncio.sleep(60)

    asyncio.create_task(_loop())


__all__ = ["app", "configure_supervisor", "ReasoningSupervisor"]
