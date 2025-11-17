"""FastAPI application exposing the hierarchical reasoning supervisor."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol

from fastapi import Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from app.actor.actor_runner import PlannerRunner
from app.agent.responder import Responder
from app.core.schemas import GovernorDecision, ReflectionOutput, ToolResult
from app.critic.critic_runner import CriticRunner
from app.governor.governor_logic import arbitrate
from app.memory.episodic_memory import EpisodicMemory
from app.reflection.reflection_runner import ReflectionRunner
from app.state.state_store import GLOBAL_STATE_STORE
from app.history.history_manager import HistoryManager

from .command_interpreter import CommandInterpreter
from .memory_manager import MemoryManager
from .state_manager import StateManager
from .tool_executor import ToolExecutor

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
ROOT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
DATA_DIR = Path(__file__).resolve().parents[2] / "data"

COG_PARAMS_PATH = CONFIG_DIR / "cog_params.json"
SYSTEM_LIMITS_PATH = CONFIG_DIR / "system_limits.json"
POLICY_CONFIG_PATH = ROOT_CONFIG_DIR / "policies.json"
IDENTITY_PATH = ROOT_CONFIG_DIR / "identity.json"
EPISODE_LOG_PATH = DATA_DIR / "episodes" / "episodes.jsonl"

AGENT_IDENTITY = {
    "name": "Spectator",
    "description": "Local autonomous reasoning agent on this machine operating within the user's workstation.",
    "role": "Local autonomous reasoning agent",
    "environment": "User's workstation",
}

SENSITIVE_FIELD_TOKENS = ("api_key", "apikey", "token", "secret", "password")


class LLMClient(Protocol):
    def generate(self, prompt: str, *, stop: Optional[Iterable[str]] = None) -> str:
        ...


class CycleRequest(BaseModel):
    objectives: List[str] = Field(..., description="Ordered objectives for this cycle")
    context: Dict[str, Any] = Field(default_factory=dict)
    memory: List[str] = Field(default_factory=list)


app = FastAPI()
history = HistoryManager()

BASE_DIR = Path(__file__).resolve().parents[2]
TEMPLATE_DIR = BASE_DIR / "app" / "ui" / "templates"
STATIC_DIR = BASE_DIR / "app" / "ui" / "static"

templates = Jinja2Templates(directory=str(TEMPLATE_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

for key in [
    "supervisor",
    "command_interpreter",
    "reflection_runner",
    "planner_runner",
    "critic_runner",
    "state_manager",
    "memory_manager",
    "tool_executor",
    "identity_profile",
    "policy_config",
    "episodic_memory",
    "cog_params",
    "system_limits",
]:
    if not hasattr(app.state, key):
        setattr(app.state, key, None)


class ReasoningSupervisor:
    """Coordinates the v2 reasoning pipeline."""

    def __init__(
        self,
        *,
        reflection_runner: ReflectionRunner,
        planner_runner: PlannerRunner,
        critic_runner: CriticRunner,
        state_manager: StateManager,
        memory_manager: MemoryManager,
        tool_executor: ToolExecutor,
        identity_profile: Dict[str, Any],
        policy_config: Dict[str, Any],
        episodic_memory: Optional[EpisodicMemory] = None,
        cog_params: Optional[Dict[str, Any]] = None,
        system_limits: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.reflection_runner = reflection_runner
        self.planner_runner = planner_runner
        self.critic_runner = critic_runner
        self.state_manager = state_manager
        self.memory_manager = memory_manager
        self.tool_executor = tool_executor
        self.identity_profile = identity_profile or AGENT_IDENTITY
        self.policy_config = policy_config or {}
        self.episodic_memory = episodic_memory
        self.cog_params = cog_params or {}
        self.system_limits = system_limits or {}
        self.responder = Responder()

    def run_user_input(
        self,
        message: str,
        *,
        context: Optional[Dict[str, Any]] = None,
        memory_snippets: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        return self._run_pipeline(
            message,
            context=context,
            memory_snippets=memory_snippets,
        )

    def run_cycle(
        self,
        objectives: List[str],
        context: Optional[Dict[str, Any]] = None,
        memory_snippets: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        text = " ".join(obj.strip() for obj in objectives if obj).strip()
        if not text:
            text = "Autonomous cycle request."
        result = self._run_pipeline(
            text,
            context=context,
            memory_snippets=memory_snippets,
        )
        return result["record"]

    def _run_pipeline(
        self,
        user_message: str,
        *,
        context: Optional[Dict[str, Any]] = None,
        memory_snippets: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        cycle_id = self.state_manager.cycle_index
        reflection = self.reflection_runner.run(user_message)
        if context:
            reflection.context.update(context)
        history.append(
            {
                "type": "reflection",
                "timestamp": time.time(),
                "output": reflection.to_dict(),
            }
        )
        current_state = self.state_manager.read()
        memory_context = self._collect_memory_context(memory_snippets)
        plan = self.planner_runner.run(
            reflection,
            current_state,
            memory_context=memory_context,
        )
        history.append(
            {
                "type": "planner",
                "timestamp": time.time(),
                "output": plan.to_dict(),
            }
        )

        critic_output = self.critic_runner.run(
            reflection=reflection,
            plan=plan,
            tool_results=[],        # no tools executed yet at this stage
            current_state=current_state,
        )

        history.append(
            {
                "type": "critic",
                "timestamp": time.time(),
                "output": critic_output.to_dict(),
            }
        )
        decision = arbitrate(
            plan,
            critic_output,
            mode=reflection.mode,
            context=reflection.context,
        )
        tool_result_objs: List[ToolResult] = []
        if decision.final_tool_calls:
            tool_result_objs = self.tool_executor.execute_many(decision.final_tool_calls)
        tool_results = [
            result.to_dict() if hasattr(result, "to_dict") else dict(result)
            for result in tool_result_objs
        ]
        updated_state = self.state_manager.read()
        cycle_record = {
            "cycle": cycle_id,
            "user_message": user_message,
            "mode": reflection.mode,
            "reflection": reflection.to_dict(),
            "plan": plan.to_dict(),
            "actor": plan.to_dict(),
            "critic": critic_output.to_dict(),
            "governor": decision.to_dict(),
            "tool_results": tool_results,
            "state": updated_state,
        }
        self.state_manager.log_cycle(cycle_record)
        snapshot = {
            "cycle": cycle_id,
            "reflection": cycle_record["reflection"],
            "plan": cycle_record["plan"],
            "critic": cycle_record["critic"],
            "governor": cycle_record["governor"],
            "tool_results": tool_results,
            "state": updated_state,
        }
        GLOBAL_STATE_STORE.save_latest(snapshot)
        GLOBAL_STATE_STORE.append_history(snapshot)
        self._record_episode(cycle_id, reflection, decision, tool_results)
        final_text = self.responder.build(
            mode=reflection.mode,
            reflection=reflection,
            plan=plan,
            decision=decision,
            tool_results=tool_results,
            identity=self.identity_profile,
            policy=self.policy_config,
            original_message=user_message,
            current_state=updated_state,
        )
        history.append(
            {
                "type": "assistant",
                "timestamp": time.time(),
                "message": final_text,
            }
        )
        return {
            "final_text": final_text,
            "cycle": cycle_id,
            "mode": reflection.mode,
            "tool_results": tool_results,
            "governor": decision.to_dict(),
            "plan": plan.to_dict(),
            "critic": critic_output.to_dict(),
            "reflection": reflection.to_dict(),
            "record": cycle_record,
        }

    def _collect_memory_context(self, memory_snippets: Optional[List[str]]) -> List[str]:
        exported = self.memory_manager.export()
        snippets: List[str] = []
        for entry in exported[-5:]:
            if isinstance(entry, dict) and entry.get("content"):
                snippets.append(str(entry["content"]))
        if memory_snippets:
            snippets.extend(memory_snippets)
        return snippets

    def _record_episode(
        self,
        cycle_id: int,
        reflection: ReflectionOutput,
        decision: GovernorDecision,
        tool_results: List[Dict[str, Any]],
    ) -> None:
        if not self.episodic_memory:
            return
        actions = [
            f"{call.name}:{json.dumps(call.arguments, sort_keys=True)}"
            for call in decision.final_tool_calls
        ]
        readings = {
            result.get("tool"): result.get("result")
            for result in tool_results
            if result.get("status") == "ok"
        }
        episode = {
            "cycle": cycle_id,
            "objectives": [reflection.goal],
            "actions": actions,
            "readings": readings,
            "outcome": decision.verdict,
            "rationale": decision.rationale,
            "notes": reflection.reflection_notes,
            "tool_results": tool_results,
        }
        try:
            self.episodic_memory.append_episode(episode)
        except OSError:
            LOGGER.warning("Failed to persist episode for cycle %s", cycle_id)


def _load_json_or_empty(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            return data
    except Exception:
        LOGGER.warning("Failed to load JSON from %s", path)
    return {}


def configure_supervisor(
    *,
    actor_client: LLMClient,
    critic_client: Optional[LLMClient] = None,
) -> None:
    """Initialise the global supervisor using provided LLM clients."""

    critic_client = critic_client or actor_client

    identity_profile = _load_json_or_empty(IDENTITY_PATH) or dict(AGENT_IDENTITY)
    policy_config = _load_json_or_empty(POLICY_CONFIG_PATH) or {}
    cog_params = _load_json_or_empty(COG_PARAMS_PATH)
    system_limits = _load_json_or_empty(SYSTEM_LIMITS_PATH)

    state_manager = StateManager()
    memory_manager = MemoryManager()

    episodic_memory: Optional[EpisodicMemory] = None
    try:
        episodic_memory = EpisodicMemory(EPISODE_LOG_PATH)
    except Exception:
        LOGGER.warning("Failed to initialise episodic memory at %s", EPISODE_LOG_PATH)

    tool_executor = ToolExecutor(
        state_manager=state_manager,
        memory_manager=memory_manager,
        policy_config=policy_config,
        system_limits=system_limits,
        history_manager=history,
    )
    reflection_runner = ReflectionRunner(actor_client, identity_profile=identity_profile)
    planner_runner = PlannerRunner(actor_client, identity=identity_profile, policy=policy_config)
    critic_runner = CriticRunner(critic_client, identity=identity_profile, policy=policy_config)

    supervisor = ReasoningSupervisor(
        reflection_runner=reflection_runner,
        planner_runner=planner_runner,
        critic_runner=critic_runner,
        state_manager=state_manager,
        memory_manager=memory_manager,
        tool_executor=tool_executor,
        identity_profile=identity_profile,
        policy_config=policy_config,
        episodic_memory=episodic_memory,
        cog_params=cog_params,
        system_limits=system_limits,
    )

    app.state.supervisor = supervisor
    app.state.command_interpreter = CommandInterpreter(actor_client)
    app.state.reflection_runner = reflection_runner
    app.state.planner_runner = planner_runner
    app.state.critic_runner = critic_runner
    app.state.state_manager = state_manager
    app.state.memory_manager = memory_manager
    app.state.tool_executor = tool_executor
    app.state.identity_profile = identity_profile
    app.state.policy_config = policy_config
    app.state.episodic_memory = episodic_memory
    app.state.cog_params = cog_params
    app.state.system_limits = system_limits


def get_supervisor() -> ReasoningSupervisor:
    supervisor: Optional[ReasoningSupervisor] = getattr(app.state, "supervisor", None)
    if supervisor is None:
        raise HTTPException(
            status_code=503,
            detail="Supervisor not configured. Call configure_supervisor first.",
        )
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
def root_redirect() -> RedirectResponse:
    return RedirectResponse(url="/chat", status_code=307)


@app.get("/chat")
def chat_page(request: Request, supervisor: ReasoningSupervisor = Depends(get_supervisor)):
    cycle_history = supervisor.state_manager.history()
    session_history = history.load()
    return templates.TemplateResponse(
        "chat.html",
        {
            "request": request,
            "history": session_history,
            "cycle_history": cycle_history,
            "active_tab": "chat",
        },
    )


@app.get("/cycles")
def cycles_page(request: Request, supervisor: ReasoningSupervisor = Depends(get_supervisor)):
    state = supervisor.state_manager.read()
    history = supervisor.state_manager.history()
    gpu_temps = supervisor.tool_executor._read_gpu_temps()
    episodes = supervisor.episodic_memory.tail(5) if supervisor.episodic_memory else []
    return templates.TemplateResponse(
        "cycles.html",
        {
            "request": request,
            "state": state,
            "history": history,
            "gpu_temps": gpu_temps,
            "episodes": episodes,
            "active_tab": "cycles",
        },
    )


@app.post("/run-cycle")
def run_cycle(
    request: CycleRequest,
    supervisor: ReasoningSupervisor = Depends(get_supervisor),
) -> Dict[str, Any]:
    return supervisor.run_cycle(
        request.objectives,
        context=request.context,
        memory_snippets=request.memory,
    )


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
    return templates.TemplateResponse(
        "cycle_row.html",
        {"request": request, "result": result, "state_snapshot": result.get("state", {})},
    )


@app.get("/history")
def ui_history(request: Request):
    return templates.TemplateResponse("history.html", {"request": request})


@app.get("/ui/history/list")
def ui_history_list():
    history = GLOBAL_STATE_STORE.load_history(50)
    return templates.TemplateResponse("history_list.html", {"history": history})


@app.post("/command")
async def command(
    request: Request,
    message: Optional[str] = Form(None),
    supervisor: ReasoningSupervisor = Depends(get_supervisor),
    interpreter: CommandInterpreter = Depends(get_command_interpreter),
):
    if message is None:
        try:
            body = await request.json()
            message = body.get("message")
        except Exception:  # pragma: no cover - malformed body safeguard
            message = None

    if not message:
        raise HTTPException(400, "No command message provided")

    LOGGER.info("Received command message: %s", message)
    structured = interpreter.interpret(message)
    structured_context = dict(structured.get("context", {}))
    if structured.get("force_action"):
        structured_context["force_action"] = True
    cycle_result = supervisor.run_cycle(
        structured.get("objectives", []),
        context=structured_context,
        memory_snippets=structured.get("memory_snippets", []),
    )
    return templates.TemplateResponse(
        "cycle_row.html",
        {"request": request, "result": cycle_result, "state_snapshot": cycle_result.get("state", {})},
    )


@app.post("/api/chat")
async def chat_api(
    request: Request,
    supervisor: ReasoningSupervisor = Depends(get_supervisor),
):
    message = None
    try:
        form = await request.form()
        message = form.get("message")
    except Exception:
        message = None

    if not message:
        try:
            raw = await request.json()
            message = raw.get("message")
        except Exception:
            pass

    if not message:
        raise HTTPException(400, "No chat message provided")

    history.append(
        {
            "type": "user",
            "timestamp": time.time(),
            "message": message,
        }
    )
    agent_output = supervisor.run_user_input(message)
    final_message = agent_output.get("final_text") or "I processed your request."
    return templates.TemplateResponse(
        "chat_message_agent.html",
        {"request": request, "message": final_message},
    )


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


def _build_debug_snapshot(supervisor: ReasoningSupervisor) -> Dict[str, Any]:
    history = supervisor.state_manager.history()
    last_cycle = history[-1] if history else {}

    reflection_snapshot = _sanitize_snapshot(last_cycle.get("reflection", {})) if last_cycle else {}
    actor_snapshot = _sanitize_snapshot(last_cycle.get("actor", {}))
    critic_snapshot = _sanitize_snapshot(last_cycle.get("critic", {}))
    governor_snapshot = _sanitize_snapshot(last_cycle.get("governor", {}))
    state_snapshot = _sanitize_snapshot(supervisor.state_manager.read())
    cog_params_with_flag = dict(supervisor.cog_params)
    cog_params_with_flag.setdefault("debug_enabled", False)
    cog_params_snapshot = _sanitize_snapshot(cog_params_with_flag)

    return {
        "cycle": supervisor.state_manager.cycle_index - 1,
        "reflection": reflection_snapshot,
        "actor": actor_snapshot,
        "critic": critic_snapshot,
        "governor": governor_snapshot,
        "state": state_snapshot,
        "cog_params": cog_params_snapshot,
    }


@app.get("/debug-system")
def debug_system(
    request: Request,
    supervisor: ReasoningSupervisor = Depends(get_supervisor),
):
    snapshot = _build_debug_snapshot(supervisor)
    return templates.TemplateResponse(
        "debug.html",
        {
            "request": request,
            "snapshot": snapshot,
            "history": history.load(),
            "active_tab": "debug",
        },
    )


@app.get("/api/debug-system")
def debug_system_api(supervisor: ReasoningSupervisor = Depends(get_supervisor)) -> Dict[str, Any]:
    return _build_debug_snapshot(supervisor)


@app.get("/api/state/latest")
def api_state_latest():
    return GLOBAL_STATE_STORE.load_latest() or {}


@app.get("/api/state/history")
def api_state_history(limit: int = 50):
    return GLOBAL_STATE_STORE.load_history(limit)


@app.on_event("startup")
async def auto_cycle_loop() -> None:
    import asyncio

    try:
        supervisor = get_supervisor()
    except HTTPException:
        LOGGER.warning("Supervisor not configured at startup; auto-cycle disabled.")
        return

    async def _loop() -> None:
        while True:
            try:
                result = supervisor.run_cycle(
                    objectives=["Monitor and stabilize GPU thermal state"],
                    context={},
                    memory_snippets=[],
                )
                verdict = result.get("governor", {}).get("verdict") if isinstance(result, dict) else None
                if verdict == "query_mode":
                    LOGGER.warning("Auto-cycle produced query_mode verdict; skipping result.")
            except Exception:  # pragma: no cover - runtime safeguard
                LOGGER.exception("Automatic cycle failure")
            await asyncio.sleep(60)

    asyncio.create_task(_loop())


__all__ = ["app", "configure_supervisor", "ReasoningSupervisor"]
