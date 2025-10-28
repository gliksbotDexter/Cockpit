"""
FastAPI Application for Dexter‑Gliksbot UI Bridge (patched)

This module exposes REST and WebSocket endpoints used by the Dexter cockpit.
It extends the upstream implementation with deeper health checks and
a unified configuration loader.  The goal of these additions is to
make the application more robust on fresh installations by validating
critical dependencies (Redis, Tesseract, disk space, etc.) at runtime
and surfacing actionable error messages to the user instead of
cryptic `ImportError` traces.  The health checks are performed lazily
to avoid introducing hard dependencies on optional packages.  See the
`perform_deep_health_checks` function for details.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Local imports – note: relative imports used to avoid sys.path issues
from ..core.triple_bus import TripleBusSystem, get_global_triple_bus, MainTopic
from ..core.policy_overlay import CompositeDenyPolicy
from ..api.websocket_manager import WebSocketManager
from ..api.connection_manager import ClientSubscription
from ..api.config_routes import router as config_router
from ..api.time_machine_routes import router as time_machine_router, set_time_machine
from ..brain.time_machine import TimeMachine
from ..brain.memory import BrainDB
from ..agents.bsm import BSM
from ..agents.dexter_orchestrator import DexterOrchestrator
from ..agents.action_executor import ActionExecutor
from ..agents.chatdock import ChatDockAgent
from ..agents.providers import get_provider
from ..speech import TextToSpeech, SpeechToText
from ..tools.windows.dock_manager import DockManager
from ..api.windows_routes import router as windows_router

# Logger setup
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Global state – initialized in lifespan()
# ----------------------------------------------------------------------
triple_bus: Optional[TripleBusSystem] = None
ws_manager: Optional[WebSocketManager] = None
time_machine: Optional[TimeMachine] = None
bsm: Optional[BSM] = None
dexter: Optional[DexterOrchestrator] = None
action_executor: Optional[ActionExecutor] = None
config_data: Optional[Dict[str, Any]] = None
dock_manager: Optional[DockManager] = None


_runtime_lock = asyncio.Lock()

async def _ensure_runtime() -> WebSocketManager:
    """Lazily initialize core services for endpoints that run outside lifespan."""
    global triple_bus, ws_manager, config_data
    async with _runtime_lock:
        if config_data is None:
            config_data = load_config()
        if triple_bus is None:
            triple_bus = get_global_triple_bus()
            await triple_bus.start_all()
        if ws_manager is None:
            ws_manager = WebSocketManager(triple_bus)
            await ws_manager.start()
    return ws_manager


def load_config() -> Dict[str, Any]:
    """
    Load configuration from a unified YAML file (configs/dexter_config.yml).

    If the file is missing, a minimal default configuration is returned.  The
    default intentionally contains only the fields required for the UI bridge
    to boot – end users should provide a proper configuration file for
    production deployments.  Logging a warning makes this clear.
    """
    config_path = Path(__file__).resolve().parents[2] / "configs" / "dexter_config.yml"
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
    else:
        logger.warning(f"Unified config file not found: {config_path}, falling back to defaults")
    # Minimal fallback: two agents and a provider pointing to local Ollama
    return {
        "version": "1.0",
        "environment": "development",
        "agents": {
            "dexter-orchestrator": {
                "label": "Dexter Central Orchestrator",
                "provider": "ollama",
                "model": "qwen2.5:3b-instruct",
                "endpoint": "http://127.0.0.1:11434",
                "temperature": 0.15,
            },
            "bsm-brain": {
                "label": "Brain State Model (BSM)",
                "provider": "ollama",
                "model": "qwen2.5:3b-instruct",
                "endpoint": "http://127.0.0.1:11434",
                "temperature": 0.3,
            },
        },
        "providers": {
            "ollama": {
                "type": "ollama",
                "endpoint": "http://127.0.0.1:11434",
                "models": ["qwen2.5:3b-instruct"],
                "timeout": 60,
            }
        },
        "deny_list": {
            "global": {},
            "agents": {},
        },
    }


def _resolve_env_placeholders(val: Optional[str]) -> Optional[str]:
    """Resolve shell-style ${VAR:-default} placeholders in a string using os.environ.

    Returns the input unchanged if no placeholder is present or val is None.
    """
    if not isinstance(val, str) or "${" not in val:
        return val
    import re, os
    pattern = re.compile(r"\$\{([A-Z0-9_]+)(?::-(.*?)|)\}")
    def repl(m):
        var = m.group(1)
        default = m.group(2)
        return os.environ.get(var, default or "")
    return pattern.sub(repl, val)


def get_agent_provider(agent_config: Dict[str, Any], config: Optional[Dict[str, Any]] = None):
    """
    Create an LLM provider from agent configuration.

    This helper extracts provider parameters (endpoint, API key, etc.) and
    delegates to `get_provider` from dexter_autonomy.agents.providers.
    It returns a provider instance or falls back to Ollama if creation
    fails.  The fallback ensures the system boots even when optional
    providers are misconfigured.
    """
    # Choose provider: agent override -> default_provider -> ollama fallback
    default_provider = (config or {}).get("default_provider") if isinstance(config, dict) else None
    provider_name = agent_config.get("provider") or default_provider or "ollama"
    model = agent_config.get("model")
    endpoint = agent_config.get("endpoint")
    api_key_env = agent_config.get("api_key_env")
    temperature = agent_config.get("temperature", 0.7)
    # Additional optional parameters
    # NOTE: Only timeout is passed to provider init. Other params like system_prompt, 
    # thinking_enabled, thinking_budget, top_p are used during chat calls, not init.
    kwargs: Dict[str, Any] = {}
    if "timeout" in agent_config:
        kwargs["timeout"] = agent_config["timeout"]
    # If a preset exists in config.providers, use it for endpoint/api_key_env defaults
    if isinstance(config, dict):
        providers_cfg = config.get("providers", {}) or {}
        preset = providers_cfg.get(provider_name, {})
        endpoint = endpoint or preset.get("endpoint")
        api_key_env = api_key_env or preset.get("api_key_env")
        model = model or preset.get("default_model")
        if "timeout" not in kwargs and "timeout" in preset:
            kwargs["timeout"] = preset.get("timeout")
        # Pass through Azure api_version if present (for future use by caller)
        if provider_name == "azure" and "api_version" in preset:
            kwargs.setdefault("api_version", preset.get("api_version"))
    # Resolve any environment placeholders in endpoint
    endpoint = _resolve_env_placeholders(endpoint)
    try:
        provider = get_provider(
            provider_name=provider_name,
            endpoint=endpoint,
            api_key_env=api_key_env,
            model=model,
            **kwargs,
        )
        logger.info(f"Created provider {provider_name} for model {model}")
        return provider
    except Exception as e:
        logger.error(f"Failed to create provider {provider_name}: {e}")
        logger.warning("Falling back to Ollama provider")
        return get_provider(
            provider_name="ollama",
            endpoint="http://127.0.0.1:11434",
            model="qwen2.5:3b-instruct",
        )


def perform_deep_health_checks() -> Dict[str, Any]:
    """
    Perform comprehensive health checks for critical system dependencies.

    The returned dictionary contains per-component status and diagnostic
    information.  Each check is wrapped in a try/except to avoid raising
    exceptions and to capture meaningful error messages.  If a module is
    optional and not installed (e.g., pytesseract), the status is marked
    as 'missing' rather than 'error'.  Clients can use this data to
    instruct the user how to install missing dependencies.
    """
    results: Dict[str, Any] = {}
    # Check Tesseract OCR
    try:
        import pytesseract  # type: ignore
        try:
            version = str(pytesseract.get_tesseract_version())
            results['tesseract'] = {"status": "ok", "version": version}
        except Exception as ex:
            results['tesseract'] = {"status": "error", "error": str(ex)}
    except ImportError:
        results['tesseract'] = {"status": "missing", "error": "pytesseract not installed"}
    # Check Redis connectivity
    if os.getenv('DEXTER_ENABLE_REDIS', '0').lower() not in {'1', 'true', 'yes'}:
        results['redis'] = {"status": "disabled"}
    else:
        try:
            import redis  # type: ignore
            try:
                r = redis.Redis(host=os.getenv('DEXTER_REDIS_HOST', 'localhost'),
                                port=int(os.getenv('DEXTER_REDIS_PORT', '6379')),
                                db=int(os.getenv('DEXTER_REDIS_DB', '0')))
                r.ping()
                results['redis'] = {"status": "ok"}
            except Exception as ex:
                results['redis'] = {"status": "error", "error": str(ex)}
        except ImportError:
            results['redis'] = {"status": "missing", "error": "redis-py not installed"}
    # Check disk space
    try:
        total, used, free = shutil.disk_usage(Path('.'))
        results['disk'] = {
            "status": "ok",
            "total_bytes": total,
            "free_bytes": free,
        }
    except Exception as ex:
        results['disk'] = {"status": "error", "error": str(ex)}
    # Check brain database accessibility
    try:
        db_path = Path('./data/brain.db')
        if db_path.exists():
            try:
                conn = sqlite3.connect(db_path)
                conn.execute('SELECT 1')
                conn.close()
                results['brain_db'] = {"status": "ok"}
            except Exception as ex:
                results['brain_db'] = {"status": "error", "error": str(ex)}
        else:
            results['brain_db'] = {"status": "missing", "error": f"Database not found at {db_path}"}
    except Exception as ex:
        results['brain_db'] = {"status": "error", "error": str(ex)}
    # Check TripleBus status (if started)
    try:
        results['triple_bus'] = {
            "status": "running" if triple_bus and triple_bus.main._started else "stopped",
            "main": bool(triple_bus and triple_bus.main._started),
            "collab": bool(triple_bus and triple_bus.collab._started),
            "private_buses": len(triple_bus._private_buses) if triple_bus else 0,
        }
    except Exception as ex:
        results['triple_bus'] = {"status": "error", "error": str(ex)}
    return results


# ----------------------------------------------------------------------
# Pydantic models for request/response
# ----------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str = Field(..., description="Message to send to the agent")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="LLM temperature override")


class ChatResponse(BaseModel):
    response: str
    agent: str
    timestamp: float
    metadata: Optional[Dict[str, Any]] = None


# ----------------------------------------------------------------------
# Webhook subscription model
# ----------------------------------------------------------------------
class WebhookSubscription(BaseModel):
    url: str = Field(..., description="Callback URL to receive events")


# Global list of webhook URLs
webhook_subscriptions: list[str] = []


# ----------------------------------------------------------------------
# Application lifecycle
# ----------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize and tear down global services on startup/shutdown.

    This function replaces the upstream lifespan with additional logging and
    configuration loading.  It runs asynchronously and yields when the app
    is ready to accept requests.  Upon shutdown, it attempts to stop
    services gracefully.
    """
    global triple_bus, ws_manager, time_machine, bsm, dexter, action_executor, config_data, dock_manager
    logger.info("Starting Dexter UI Bridge (patched)...")
    # Load configuration
    config_data = load_config()
    logger.info(f"Loaded configuration environment: {config_data.get('environment', 'unknown')}")
    # Initialize TripleBus and permissive policy
    policy = CompositeDenyPolicy(profile={}, overlay=None)
    triple_bus = get_global_triple_bus()
    await triple_bus.start_all()
    logger.info("TripleBus system started")
    dock_manager = DockManager(triple_bus)
    app.state.dock_manager = dock_manager
    logger.info("DockManager ready")
    # Initialize TimeMachine
    time_machine = TimeMachine(
        timeline_db=str(Path(config_data.get("memory", {}).get("timeline_db", "./data/timeline.db"))),
        snapshot_db=str(Path(config_data.get("memory", {}).get("snapshot_db", "./data/snapshots.db"))),
        snapshot_dir=str(Path(config_data.get("memory", {}).get("snapshot_dir", "./data/snapshots"))),
        auto_snapshot_interval=300.0,
        retention_days=30,
        enable_compression=True,
    )
    await time_machine.start()
    set_time_machine(time_machine)
    logger.info("TimeMachine started")
    # Initialize Brain
    brain_db_path = str(Path(config_data.get("memory", {}).get("brain_db", "./data/brain.db")))
    brain = BrainDB(db_path=brain_db_path)
    logger.info("Brain initialized")
    # Initialize ActionExecutor
    action_executor = ActionExecutor(triple_bus, policy, None)
    logger.info("ActionExecutor initialized")
    # Extract agent configs
    agents_config = config_data.get("agents", {})
    dexter_cfg = agents_config.get("dexter-orchestrator", {})
    bsm_cfg = agents_config.get("bsm-brain", {})
    # Create provider instances
    logger.info("Creating providers...")
    dexter_provider = get_agent_provider(dexter_cfg, config_data)
    bsm_provider = get_agent_provider(bsm_cfg, config_data)
    dexter_model = (
        getattr(dexter_provider, "model", None)
        or getattr(dexter_provider, "default_model", None)
        or dexter_cfg.get("model")
        or "qwen2.5:3b-instruct"
    )
    bsm_model = (
        getattr(bsm_provider, "model", None)
        or getattr(bsm_provider, "default_model", None)
        or bsm_cfg.get("model")
        or "qwen2.5:3b-instruct"
    )
    dexter_endpoint = getattr(dexter_provider, 'endpoint', 'http://127.0.0.1:11434')
    bsm_endpoint = getattr(bsm_provider, 'endpoint', 'http://127.0.0.1:11434')
    # Initialize BSM
    bsm = BSM(
        buses=triple_bus,
        brain=brain,
        model=bsm_model,
        host=bsm_endpoint,
        temperature=bsm_cfg.get("temperature", 0.3),
        time_machine=time_machine,
        # Persist to configured directory
        persistence_dir=str(Path(config_data.get("memory", {}).get("bsm_dir", "./data/bsm"))),
    )
    await bsm.start()
    logger.info("BSM started")
    # Initialize ChatDock agent
    chatdock = ChatDockAgent(triple_bus, policy, action_executor, bsm, None)
    logger.info("ChatDock initialized")
    # Initialize Dexter orchestrator
    dexter = DexterOrchestrator(
        buses=triple_bus,
        policy=policy,
        brain=brain,
        executor=action_executor,
        bsm=bsm,
        chatdock=chatdock,
        config={
            "slots": {
                "dexter-orchestrator": {
                    "endpoint": dexter_endpoint,
                    "api_key_env": dexter_cfg.get("api_key_env", ""),
                    "model": dexter_model,
                    "system_prompt": dexter_cfg.get("system_prompt", "You are Dexter."),
                    "ollama_options": {},
                }
            }
        },
    )
    logger.info("Dexter Orchestrator initialized")
    # Store providers and config on app state for endpoints
    app.state.dexter_provider = dexter_provider
    app.state.bsm_provider = bsm_provider
    app.state.config = config_data
    app.state.speech = config_data.get("speech", {})
    app.state.speech_state = {"agent_mute": {}}  # runtime toggle map
    # Initialize WebSocketManager
    ws_manager = WebSocketManager(triple_bus)
    await ws_manager.start()
    logger.info("WebSocketManager started")
    logger.info("UI Bridge ready")
    try:
        yield
    finally:
        logger.info("Shutting down UI Bridge...")
        if bsm:
            await bsm.stop()
            logger.info("BSM stopped")
        if time_machine:
            await time_machine.stop()
            logger.info("TimeMachine stopped")
        if ws_manager:
            await ws_manager.stop()
            logger.info("WebSocketManager stopped")
        if triple_bus:
            await triple_bus.stop_all()
            logger.info("TripleBus stopped")
        logger.info("UI Bridge shutdown complete")


# ----------------------------------------------------------------------
# FastAPI app setup
# ----------------------------------------------------------------------
app = FastAPI(
    title="Dexter UI Bridge API (patched)",
    description="Real‑time streaming for Dexter Cockpit UI", version="1.0.0", lifespan=lifespan
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(config_router)
app.include_router(time_machine_router)
app.include_router(windows_router)


# ----------------------------------------------------------------------
# Health endpoints
# ----------------------------------------------------------------------
@app.get("/health")
@app.get("/healthz")
async def health_check() -> JSONResponse:
    """
    Extended health check endpoint.

    In addition to the basic status of internal components, this
    endpoint invokes `perform_deep_health_checks` to verify critical
    external dependencies.  The result includes the base health status
    and a `dependencies` section for deeper diagnostics.
    """
    try:
        manager = await _ensure_runtime()
        bus_stats = triple_bus.get_stats() if triple_bus else {}
        conn_stats = manager.connection_manager.get_stats()
        base_status = {
            "status": "ok",
            "components": {
                "triple_bus": {
                    "status": "running" if bus_stats else "stopped",
                    "stats": bus_stats,
                },
                "websocket_manager": {
                    "status": "running" if manager._started else "stopped",
                    "active_connections": conn_stats.get("active_connections", 0),
                    "event_history_size": conn_stats.get("history_size", 0),
                },
            },
            "dependencies": perform_deep_health_checks(),
        }
        return JSONResponse(content=base_status, status_code=200)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=503)


# ----------------------------------------------------------------------
# Agent slot endpoint for legacy cockpit
# ----------------------------------------------------------------------
@app.get("/slots")
async def get_agent_slots() -> JSONResponse:
    """
    Return agent slot configurations formatted for the Cockpit UI.

    The UI expects a simple mapping of slot IDs to details (label,
    description, endpoint, model, API key env, temperature, etc.).  If
    configuration is not loaded, a 500 error is returned.
    """
    try:
        if not config_data:
            return JSONResponse(content={"error": "Configuration not loaded"}, status_code=500)
        agents_cfg = config_data.get("agents", {})
        slots: Dict[str, Any] = {}
        for agent_id, agent_cfg in agents_cfg.items():
            slots[agent_id] = {
                "label": agent_cfg.get("label", agent_id),
                "description": agent_cfg.get("description", ""),
                "endpoint": agent_cfg.get("endpoint", "http://127.0.0.1:8765"),
                "model": agent_cfg.get("model", ""),
                "deployment": agent_cfg.get("deployment", ""),
                "api_key_env": agent_cfg.get("api_key_env", ""),
                "temperature": agent_cfg.get("temperature", 0.7),
                "system_prompt": agent_cfg.get("system_prompt", ""),
                "provider": agent_cfg.get("provider", "ollama"),
            }
        return JSONResponse(content={"slots": slots, "default_slot": "dexter-orchestrator"})
    except Exception as e:
        logger.error(f"Failed to get agent slots: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


# ----------------------------------------------------------------------
# Chat endpoints (Dexter and BSM)
# ----------------------------------------------------------------------
@app.post("/dexter/chat", response_model=ChatResponse)
async def chat_with_dexter(request: ChatRequest) -> ChatResponse:
    if not dexter:
        raise HTTPException(status_code=503, detail="Dexter not initialized")
    try:
        logger.info(f"Dexter chat request: {request.message[:100]}...")
        provider = app.state.dexter_provider
        dexter_cfg = app.state.config.get("agents", {}).get("dexter-orchestrator", {})
        system_prompt = dexter_cfg.get("system_prompt", "You are Dexter.")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.message},
        ]
        loop = asyncio.get_running_loop()
        # Optional model/deployment overrides per agent
        model_override = dexter_cfg.get("model")
        deployment = dexter_cfg.get("deployment")
        response_obj = await loop.run_in_executor(
            None,
            lambda: provider.chat(
                messages=messages,
                temperature=request.temperature or dexter_cfg.get("temperature", 0.15),
                model=model_override,
                **({"deployment": deployment} if deployment else {})
            ),
        )
        response_text = response_obj.content
        # Publish events to TripleBus so BSM observes conversation
        await triple_bus.main.publish(MainTopic.USER_INPUT, {
            "source": "api",
            "message": request.message,
            "timestamp": time.time(),
        })
        await triple_bus.main.publish(MainTopic.TRACE, {
            "agent": "dexter",
            "message": response_text,
            "timestamp": time.time(),
            "provider": provider.name,
            "model": response_obj.model,
        })
        # Update BSM with the conversation
        if bsm and bsm._started:
            try:
                bsm.observe(request.message)
                bsm.observe(response_text)
            except Exception as obs_err:
                logger.warning(f"Failed to observe Dexter chat: {obs_err}")
        # Optionally speak the response via TTS (non-blocking)
        try:
            speech_cfg = getattr(app.state, 'speech', {}).get('tts', {}) if hasattr(app.state, 'speech') else {}
            mute_map = getattr(app.state, 'speech_state', {}).get('agent_mute', {}) if hasattr(app.state, 'speech_state') else {}
            agent_muted = bool(mute_map.get('dexter-orchestrator', False))
            if speech_cfg and speech_cfg.get('enabled') and speech_cfg.get('auto_speak_responses') and not agent_muted:
                tts = TextToSpeech(speech_cfg)
                loop.run_in_executor(None, lambda: tts.speak(response_text, block=True))
        except Exception as speak_err:
            logger.debug(f"TTS speak skipped: {speak_err}")
        logger.info(f"Dexter responded ({provider.name}/{response_obj.model}): {response_text[:100]}...")
        return ChatResponse(
            response=response_text,
            agent="dexter",
            timestamp=time.time(),
            metadata={"provider": provider.name, "model": response_obj.model, "usage": response_obj.usage},
        )
    except Exception as e:
        logger.error(f"Dexter chat failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Dexter chat failed: {str(e)}")


@app.post("/bsm/chat", response_model=ChatResponse)
async def chat_with_bsm(request: ChatRequest) -> ChatResponse:
    if not bsm:
        raise HTTPException(status_code=503, detail="BSM not initialized")
    try:
        logger.info(f"BSM chat request: {request.message[:100]}...")
        provider = app.state.bsm_provider
        bsm_cfg = app.state.config.get("agents", {}).get("bsm-brain", {})
        system_prompt = bsm_cfg.get("system_prompt", "You are BSM, the Brain/State Model.")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.message},
        ]
        loop = asyncio.get_running_loop()
        # Optional model/deployment overrides per agent
        model_override = bsm_cfg.get("model")
        deployment = bsm_cfg.get("deployment")
        response_obj = await loop.run_in_executor(
            None,
            lambda: provider.chat(
                messages=messages,
                temperature=request.temperature or bsm_cfg.get("temperature", 0.3),
                model=model_override,
                **({"deployment": deployment} if deployment else {})
            ),
        )
        response_text = response_obj.content
        # Publish events to TripleBus so BSM observes itself
        await triple_bus.main.publish(MainTopic.USER_INPUT, {
            "source": "api",
            "target": "bsm",
            "message": request.message,
            "timestamp": time.time(),
        })
        await triple_bus.main.publish(MainTopic.TRACE, {
            "agent": "bsm",
            "message": response_text,
            "timestamp": time.time(),
            "provider": provider.name,
            "model": response_obj.model,
            "observations": getattr(bsm, '_observation_count', 0),
            "context_provided": getattr(bsm, '_context_provided_count', 0),
        })
        # Update BSM with conversation
        if bsm and bsm._started:
            try:
                bsm.observe(request.message)
                bsm.observe(response_text)
            except Exception as obs_err:
                logger.warning(f"Failed to observe BSM chat: {obs_err}")
        # Optionally speak the response via TTS
        try:
            speech_cfg = getattr(app.state, 'speech', {}).get('tts', {}) if hasattr(app.state, 'speech') else {}
            mute_map = getattr(app.state, 'speech_state', {}).get('agent_mute', {}) if hasattr(app.state, 'speech_state') else {}
            agent_muted = bool(mute_map.get('bsm-brain', False))
            if speech_cfg and speech_cfg.get('enabled') and speech_cfg.get('auto_speak_responses') and not agent_muted:
                tts = TextToSpeech(speech_cfg)
                loop.run_in_executor(None, lambda: tts.speak(response_text, block=True))
        except Exception as speak_err:
            logger.debug(f"TTS speak skipped: {speak_err}")
        logger.info(f"BSM responded ({provider.name}/{response_obj.model}): {response_text[:100]}...")
        return ChatResponse(
            response=response_text,
            agent="bsm",
            timestamp=time.time(),
            metadata={
                "provider": provider.name,
                "model": response_obj.model,
                "observations": getattr(bsm, '_observation_count', 0),
                "context_provided": getattr(bsm, '_context_provided_count', 0),
                "usage": response_obj.usage,
            },
        )
    except Exception as e:
        logger.error(f"BSM chat failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"BSM chat failed: {str(e)}")


# ----------------------------------------------------------------------
# Speech endpoints (TTS/STT)
# ----------------------------------------------------------------------
class TTSRequest(BaseModel):
    text: str = Field(..., description="Text to speak")
    backend: Optional[str] = Field(None, description="Override TTS backend")
    voice: Optional[str] = Field(None, description="Override voice")


@app.post("/speech/tts")
async def speak_text(req: TTSRequest) -> JSONResponse:
    cfg = getattr(app.state, 'speech', {}).get('tts', {}) if hasattr(app.state, 'speech') else {}
    if req.backend:
        cfg = {**cfg, 'backend': req.backend}
    if req.voice:
        cfg = {**cfg, 'voice': req.voice}
    try:
        tts = TextToSpeech(cfg)
        loop = asyncio.get_running_loop()
        ok = await loop.run_in_executor(None, lambda: tts.speak(req.text, block=True))
        return JSONResponse(content={"ok": bool(ok)}, status_code=200 if ok else 500)
    except Exception as e:
        logger.error(f"TTS error: {e}")
        return JSONResponse(content={"ok": False, "error": str(e)}, status_code=500)


class TranscribeFileRequest(BaseModel):
    file_path: str = Field(..., description="Path to audio file to transcribe")
    backend: Optional[str] = Field(None, description="Override STT backend")
    language: Optional[str] = Field(None, description="Language code, e.g., en-US")


@app.post("/speech/transcribe_file")
async def transcribe_file(req: TranscribeFileRequest) -> JSONResponse:
    cfg = getattr(app.state, 'speech', {}).get('stt', {}) if hasattr(app.state, 'speech') else {}
    if req.backend:
        cfg = {**cfg, 'backend': req.backend}
    if req.language:
        cfg = {**cfg, 'language': req.language}
    try:
        stt = SpeechToText(cfg)
        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(None, lambda: stt.transcribe_file(req.file_path))
        if text is None:
            return JSONResponse(content={"ok": False, "text": None}, status_code=500)
        return JSONResponse(content={"ok": True, "text": text}, status_code=200)
    except Exception as e:
        logger.error(f"STT error: {e}")
        return JSONResponse(content={"ok": False, "error": str(e)}, status_code=500)


@app.get("/speech/voices")
async def list_tts_voices() -> JSONResponse:
    cfg = getattr(app.state, 'speech', {}).get('tts', {}) if hasattr(app.state, 'speech') else {}
    try:
        tts = TextToSpeech(cfg)
        # Capture printed voices by redirecting stdout
        import io, sys
        buf = io.StringIO()
        stdout = sys.stdout
        try:
            sys.stdout = buf
            tts.list_voices()
        finally:
            sys.stdout = stdout
        lines = [l for l in buf.getvalue().splitlines() if l.strip()]
        return JSONResponse(content={"voices": lines}, status_code=200)
    except Exception as e:
        logger.error(f"List voices failed: {e}")
        return JSONResponse(content={"voices": [], "error": str(e)}, status_code=500)


class AgentMuteRequest(BaseModel):
    agent_id: str = Field(..., description="Agent identifier (e.g., dexter-orchestrator)")
    mute: bool = Field(..., description="True to mute, False to unmute")


@app.post("/speech/mute_agent")
async def mute_agent(req: AgentMuteRequest) -> JSONResponse:
    try:
        if not hasattr(app.state, 'speech_state'):
            app.state.speech_state = {"agent_mute": {}}
        app.state.speech_state.setdefault('agent_mute', {})[req.agent_id] = bool(req.mute)
        return JSONResponse(content={"ok": True, "agent_id": req.agent_id, "mute": bool(req.mute)}, status_code=200)
    except Exception as e:
        logger.error(f"Mute toggle failed: {e}")
        return JSONResponse(content={"ok": False, "error": str(e)}, status_code=500)


@app.get("/speech/mute_map")
async def get_mute_map() -> JSONResponse:
    try:
        mute_map = getattr(app.state, 'speech_state', {}).get('agent_mute', {}) if hasattr(app.state, 'speech_state') else {}
        return JSONResponse(content={"agent_mute": mute_map}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"agent_mute": {}, "error": str(e)}, status_code=500)


# ----------------------------------------------------------------------
# Data inspection endpoints
# ----------------------------------------------------------------------
@app.get("/patterns")
async def get_patterns() -> JSONResponse:
    """
    Return the current pattern frequency table from the BSM.

    The response contains a mapping of token → count.  A 503 error is
    returned if the BSM has not been initialized yet.
    """
    if not bsm:
        raise HTTPException(status_code=503, detail="BSM not initialized")
    try:
        return JSONResponse(content={"patterns": bsm.get_patterns()}, status_code=200)
    except Exception as e:
        logger.error(f"Failed to retrieve patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/knowledge")
async def get_knowledge_graph() -> JSONResponse:
    """
    Return the current co‑occurrence knowledge graph from the BSM.

    The response contains an adjacency list mapping each token to its
    neighbouring tokens and co‑occurrence counts.  A 503 error is
    returned if the BSM has not been initialized.
    """
    if not bsm:
        raise HTTPException(status_code=503, detail="BSM not initialized")
    try:
        return JSONResponse(content={"knowledge": bsm.get_knowledge_graph()}, status_code=200)
    except Exception as e:
        logger.error(f"Failed to retrieve knowledge graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------------------------------------------------
# Webhook subscription endpoint
# ----------------------------------------------------------------------
@app.post("/webhooks/subscribe")
async def subscribe_webhook(subscription: WebhookSubscription) -> JSONResponse:
    """
    Register a webhook callback to receive future events.

    This is a simplified stub: the subscription URL is stored in an
    in‑memory list.  Event delivery is not implemented in this
    patched version.  Clients may poll the /patterns and /knowledge
    endpoints to observe changes.
    """
    webhook_subscriptions.append(subscription.url)
    logger.info(f"Webhook subscribed: {subscription.url}")
    return JSONResponse(content={"status": "subscribed", "url": subscription.url}, status_code=201)


# ----------------------------------------------------------------------
# Webhook management endpoints
# ----------------------------------------------------------------------
@app.get("/webhooks")
async def list_webhooks() -> JSONResponse:
    """
    Return the list of currently subscribed webhook URLs.
    """
    return JSONResponse(content={"webhooks": webhook_subscriptions}, status_code=200)


@app.delete("/webhooks/{encoded_url}")
async def unsubscribe_webhook(encoded_url: str) -> JSONResponse:
    """
    Unsubscribe the given webhook URL.

    The URL is URL‑encoded to allow special characters.  If the URL is
    not found in the subscription list, a 404 error is returned.
    """
    from urllib.parse import unquote
    url = unquote(encoded_url)
    if url not in webhook_subscriptions:
        raise HTTPException(status_code=404, detail="Webhook not subscribed")
    webhook_subscriptions.remove(url)
    logger.info(f"Webhook unsubscribed: {url}")
    return JSONResponse(content={"status": "unsubscribed", "url": url}, status_code=200)


# ----------------------------------------------------------------------
# Docked window automation endpoints
# ----------------------------------------------------------------------
class TypeInWindowRequest(BaseModel):
    hwnd: int = Field(..., description="Window handle to type into")
    text: str = Field(..., description="Text to type")
    activate_first: bool = Field(default=True, description="Whether to activate window before typing")

@app.post("/automation/type_in_window")
async def type_in_window(request: TypeInWindowRequest) -> JSONResponse:
    """
    Type text into a specific window by HWND.
    
    This endpoint first activates the target window (if activate_first=True),
    then types the text. This is essential for typing into docked windows
    that may not have keyboard focus.
    """
    if not action_executor:
        raise HTTPException(status_code=503, detail="ActionExecutor not initialized")
    
    try:
        # First activate the window if requested
        if request.activate_first:
            activate_result = await action_executor.execute_intent("activate_window", {"hwnd": request.hwnd})
            if activate_result.get("status") != "ok":
                return JSONResponse(
                    content={
                        "status": "error",
                        "reason": f"Failed to activate window: {activate_result.get('reason', 'unknown')}",
                        "data": activate_result
                    },
                    status_code=400
                )
        
        # Then type the text
        type_result = await action_executor.execute_intent("type_text", {"text": request.text})
        
        if type_result.get("status") == "ok":
            return JSONResponse(
                content={
                    "status": "success",
                    "hwnd": request.hwnd,
                    "text_length": len(request.text)
                },
                status_code=200
            )
        else:
            return JSONResponse(
                content={
                    "status": "error",
                    "reason": type_result.get("reason", "type_text failed"),
                    "data": type_result
                },
                status_code=400
            )
    except Exception as e:
        logger.error(f"Failed to type in window {request.hwnd}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------------------------------------------------
# BSM persistence and context endpoints
# ----------------------------------------------------------------------
@app.post("/bsm/save")
async def save_bsm_state() -> JSONResponse:
    """
    Persist the BSM's patterns and knowledge graph to disk.
    """
    if not bsm:
        raise HTTPException(status_code=503, detail="BSM not initialized")
    try:
        bsm.save_state()
        return JSONResponse(content={"status": "saved"}, status_code=200)
    except Exception as e:
        logger.error(f"Failed to save BSM state: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/bsm/load")
async def load_bsm_state() -> JSONResponse:
    """
    Load the BSM's patterns and knowledge graph from disk.
    """
    if not bsm:
        raise HTTPException(status_code=503, detail="BSM not initialized")
    try:
        bsm.load_state()
        return JSONResponse(content={"status": "loaded"}, status_code=200)
    except Exception as e:
        logger.error(f"Failed to load BSM state: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/bsm/context")
async def get_bsm_context(max_items: int = Query(10, ge=1, le=50)) -> JSONResponse:
    """
    Return a summary of the BSM's most frequent tokens.

    Clients can adjust `max_items` to limit the length of the summary.
    """
    if not bsm:
        raise HTTPException(status_code=503, detail="BSM not initialized")
    try:
        summary = bsm.provide_context(max_items)
        return JSONResponse(content={"context": summary}, status_code=200)
    except Exception as e:
        logger.error(f"Failed to generate BSM context: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------------------------------------------------
# Pattern and knowledge graph queries
# ----------------------------------------------------------------------
@app.get("/patterns/top")
async def get_top_patterns(n: int = Query(10, ge=1, le=100)) -> JSONResponse:
    """
    Return the top `n` tokens by frequency.
    """
    if not bsm:
        raise HTTPException(status_code=503, detail="BSM not initialized")
    try:
        patterns = bsm.get_patterns()
        top = sorted(patterns.items(), key=lambda kv: kv[1], reverse=True)[:n]
        return JSONResponse(content={"patterns": top}, status_code=200)
    except Exception as e:
        logger.error(f"Failed to retrieve top patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/knowledge/node/{token}")
async def get_graph_node(token: str) -> JSONResponse:
    """
    Return the neighbours and weights for a single token in the knowledge graph.
    """
    if not bsm:
        raise HTTPException(status_code=503, detail="BSM not initialized")
    try:
        graph = bsm.get_knowledge_graph()
        node = token.lower()
        neighbours = graph.get(node, {})
        return JSONResponse(content={node: neighbours}, status_code=200)
    except Exception as e:
        logger.error(f"Failed to retrieve node {token} from knowledge graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------------------------------------------------
# Root endpoint
# ----------------------------------------------------------------------

@app.get("/ws/health")
async def websocket_health() -> Dict[str, Any]:
    manager = await _ensure_runtime()
    stats = manager.get_stats()
    connection_stats = manager.connection_manager.get_stats()
    return {"status": "healthy", "websocket_manager": stats, "connection_manager": connection_stats}

@app.get("/stats")
async def system_stats() -> Dict[str, Any]:
    manager = await _ensure_runtime()
    bus_stats = triple_bus.get_stats() if triple_bus else {}
    return {
        "triple_bus": bus_stats,
        "websocket_manager": manager.get_stats(),
        "connection_manager": manager.connection_manager.get_stats(),
    }

@app.get("/debug/connections")
async def debug_connections() -> Dict[str, Any]:
    manager = await _ensure_runtime()
    conn_stats = manager.connection_manager.get_stats()
    return {
        "total_connections": conn_stats.get("total_connections", 0),
        "active_connections": conn_stats.get("active_connections", 0),
        "messages_dropped": conn_stats.get("messages_dropped", 0),
        "messages_sent": conn_stats.get("messages_sent", 0),
        "connections": conn_stats.get("clients", []),
    }

@app.get("/debug/cache")
async def debug_cache() -> Dict[str, Any]:
    manager = await _ensure_runtime()
    snapshot = manager.get_state_snapshot()
    return {
        "agents": snapshot.get("agents", []),
        "missions": snapshot.get("missions", []),
        "system_metrics": snapshot.get("system_metrics", {}),
        "cache_stats": snapshot.get("cache_stats", {}),
    }

@app.get("/debug/cache")
async def debug_cache() -> Dict[str, Any]:
    if ws_manager is None:
        raise HTTPException(status_code=503, detail="WebSocket manager not initialized")
    snapshot = ws_manager.get_state_snapshot()
    return {
        "agents": snapshot.get("agents", []),
        "missions": snapshot.get("missions", []),
        "system_metrics": snapshot.get("system_metrics", {}),
        "cache_stats": snapshot.get("cache_stats", {}),
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Main WebSocket endpoint for Cockpit UI connection."""
    manager = await _ensure_runtime()
    
    # Generate unique client ID for this connection
    client_id = f"cockpit-{uuid.uuid4().hex[:8]}"
    
    # Get client metadata from headers
    metadata = {
        "user_agent": websocket.headers.get("user-agent", "unknown"),
        "host": websocket.headers.get("host", "unknown"),
    }
    
    # Connect the client
    await manager.connection_manager.connect(client_id, websocket, metadata=metadata)
    logger.info(f"WebSocket client connected: {client_id}")
    
    try:
        while True:
            data = await websocket.receive_text()
            # Handle incoming messages from client if needed
            logger.debug(f"Received WebSocket message from {client_id}: {data}")
    except WebSocketDisconnect:
        await manager.connection_manager.disconnect(client_id)
        logger.info(f"WebSocket client disconnected: {client_id}")

@app.get("/")
async def root() -> Dict[str, Any]:
    return {
        "name": "Dexter UI Bridge API",
        "version": "1.0.0",
        "description": "Real‑time streaming for Dexter Cockpit UI",
        "endpoints": {
            "health": "GET /health, /healthz",
            "slots": "GET /slots",
            "dexter_chat": "POST /dexter/chat",
            "bsm_chat": "POST /bsm/chat",
            "agents": "GET /agents",
            "agent_config": "GET /agents/{agent_id}/config",
            "update_agent": "PUT /agents/{agent_id}/config",
            "create_agent": "POST /agents",
            "delete_agent": "DELETE /agents/{agent_id}",
            "send_agent_msg": "POST /agents/{agent_id}/message",
            "agent_websocket": "WebSocket /ws/agent/{agent_id}",
            "patterns": "GET /patterns",
            "patterns_top": "GET /patterns/top?n=10",
            "knowledge": "GET /knowledge",
            "knowledge_node": "GET /knowledge/node/{token}",
            "ws": "WebSocket /ws",
            "ws_health": "GET /ws/health",
            "stats": "GET /stats",
            "debug_connections": "GET /debug/connections",
            "debug_cache": "GET /debug/cache",
            "bsm_context": "GET /bsm/context",
            "bsm_save": "POST /bsm/save",
            "bsm_load": "POST /bsm/load",
            "type_in_window": "POST /automation/type_in_window",
            "webhook_subscribe": "POST /webhooks/subscribe",
            "webhooks": "GET /webhooks",
            "webhook_unsubscribe": "DELETE /webhooks/{encoded_url}",
        },
        "documentation": "/docs",
    }


# ----------------------------------------------------------------------
# Agent Management Endpoints (for AgentSlotControl UI)
# ----------------------------------------------------------------------

class AgentConfigRequest(BaseModel):
    """Request model for agent configuration"""
    display_name: str = Field(..., description="Human-friendly agent name")
    provider: str = Field(..., description="Provider name (azure, openai, nvidia, ollama)")
    model: str = Field(..., description="Model name")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="LLM temperature")
    top_p: float = Field(1.0, ge=0.0, le=1.0, description="Top-p sampling")
    system_prompt: Optional[str] = Field(None, description="System prompt for the agent")


class AgentMessageRequest(BaseModel):
    """Request model for sending message to agent's private bus"""
    message: str = Field(..., description="Message to send to the agent")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


@app.get("/agents")
async def list_agents() -> Dict[str, Any]:
    """
    List all configured agents from dexter_config.yml
    
    Returns agent IDs, labels, descriptions, providers, and current status.
    Used by MainWindow to dynamically create agent slot tabs.
    """
    try:
        agents_config = config_data.get("agents", {})
        agents_list = []
        
        for agent_id, agent_cfg in agents_config.items():
            agents_list.append({
                "id": agent_id,
                "label": agent_cfg.get("label", agent_id),
                "description": agent_cfg.get("description", ""),
                "provider": agent_cfg.get("provider", config_data.get("default_provider", "azure")),
                "model": agent_cfg.get("model", ""),
                "temperature": agent_cfg.get("temperature", 0.7),
                "top_p": agent_cfg.get("top_p", 1.0),
                "system_prompt": agent_cfg.get("system_prompt", ""),
                "status": "idle"  # TODO: Get actual status from agent instance
            })
        
        return {
            "agents": agents_list,
            "count": len(agents_list)
        }
    except Exception as e:
        logger.error(f"Error listing agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agents/{agent_id}/config")
async def get_agent_config(agent_id: str) -> Dict[str, Any]:
    """
    Get configuration for a specific agent
    
    Used by AgentSlotControl to populate Configuration tab on load.
    """
    try:
        agents_config = config_data.get("agents", {})
        
        if agent_id not in agents_config:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
        
        agent_cfg = agents_config[agent_id]
        
        return {
            "id": agent_id,
            "label": agent_cfg.get("label", agent_id),
            "description": agent_cfg.get("description", ""),
            "provider": agent_cfg.get("provider", config_data.get("default_provider", "azure")),
            "model": agent_cfg.get("model", ""),
            "deployment": agent_cfg.get("deployment", ""),
            "temperature": agent_cfg.get("temperature", 0.7),
            "top_p": agent_cfg.get("top_p", 1.0),
            "system_prompt": agent_cfg.get("system_prompt", ""),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/agents/{agent_id}/config")
async def update_agent_config(agent_id: str, config_req: AgentConfigRequest) -> Dict[str, Any]:
    """
    Update agent configuration and persist to dexter_config.yml
    
    Called when user clicks "Save" in AgentSlotControl Configuration tab.
    """
    try:
        import yaml
        
        config_path = Path("./configs/dexter_config.yml")
        
        # Load current config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update agent config
        if "agents" not in config:
            config["agents"] = {}
        
        config["agents"][agent_id] = {
            "label": config_req.display_name,
            "provider": config_req.provider,
            "model": config_req.model,
            "temperature": config_req.temperature,
            "top_p": config_req.top_p,
        }
        
        if config_req.system_prompt:
            config["agents"][agent_id]["system_prompt"] = config_req.system_prompt
        
        # Write back to file
        with open(config_path, 'w') as f:
            yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
        
        # Reload config into memory
        global config_data
        config_data = load_config()
        
        logger.info(f"Updated configuration for agent '{agent_id}'")
        
        return {"status": "success", "message": f"Agent '{agent_id}' configuration updated"}
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Configuration file not found")
    except Exception as e:
        logger.error(f"Error updating agent config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agents")
async def create_agent(config_req: AgentConfigRequest) -> Dict[str, Any]:
    """
    Create a new agent and add to configuration
    
    Called when user clicks "Spawn New" in AgentSlotControl Configuration tab.
    Generates a new agent ID, adds to config, and returns the new agent details.
    """
    try:
        import yaml
        import re
        
        # Generate agent ID from display name
        agent_id = re.sub(r'[^a-z0-9-]', '', config_req.display_name.lower().replace(' ', '-'))
        
        config_path = Path("./configs/dexter_config.yml")
        
        # Load current config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check if agent already exists
        if "agents" not in config:
            config["agents"] = {}
        
        if agent_id in config["agents"]:
            # Make ID unique
            counter = 1
            base_id = agent_id
            while f"{base_id}-{counter}" in config["agents"]:
                counter += 1
            agent_id = f"{base_id}-{counter}"
        
        # Add new agent config
        config["agents"][agent_id] = {
            "label": config_req.display_name,
            "provider": config_req.provider,
            "model": config_req.model,
            "temperature": config_req.temperature,
            "top_p": config_req.top_p,
        }
        
        if config_req.system_prompt:
            config["agents"][agent_id]["system_prompt"] = config_req.system_prompt
        
        # Write back to file
        with open(config_path, 'w') as f:
            yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
        
        # Reload config into memory
        global config_data
        config_data = load_config()
        
        logger.info(f"Created new agent '{agent_id}'")
        
        # TODO: Instantiate the agent and add to triple_bus
        
        return {
            "status": "success",
            "agent_id": agent_id,
            "message": f"Agent '{config_req.display_name}' created successfully"
        }
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Configuration file not found")
    except Exception as e:
        logger.error(f"Error creating agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/agents/{agent_id}")
async def delete_agent(agent_id: str) -> Dict[str, Any]:
    """
    Delete an agent from configuration
    
    Called when user clicks "Delete" in AgentSlotControl Configuration tab.
    Removes from config and stops the agent instance.
    """
    try:
        import yaml
        
        # Prevent deletion of system agents
        if agent_id in ["dexter-orchestrator", "bsm-brain"]:
            raise HTTPException(status_code=403, detail=f"Cannot delete system agent '{agent_id}'")
        
        config_path = Path("./configs/dexter_config.yml")
        
        # Load current config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Remove agent
        if "agents" not in config or agent_id not in config["agents"]:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
        
        del config["agents"][agent_id]
        
        # Write back to file
        with open(config_path, 'w') as f:
            yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
        
        # Reload config into memory
        global config_data
        config_data = load_config()
        
        logger.info(f"Deleted agent '{agent_id}'")
        
        # TODO: Stop agent instance and remove from triple_bus
        
        return {"status": "success", "message": f"Agent '{agent_id}' deleted successfully"}
    except HTTPException:
        raise
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Configuration file not found")
    except Exception as e:
        logger.error(f"Error deleting agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agents/{agent_id}/message")
async def send_agent_message(agent_id: str, msg_req: AgentMessageRequest) -> Dict[str, Any]:
    """
    Send a message to an agent's private bus
    
    Called when user sends a message in AgentSlotControl Chat tab.
    Publishes message to the agent's private bus and returns immediately.
    Response will come via WebSocket.
    """
    try:
        if not triple_bus:
            raise HTTPException(status_code=503, detail="Triple bus not initialized")
        
        # Get private bus for this agent
        private_bus = triple_bus.get_private(agent_id)
        
        if not private_bus:
            raise HTTPException(status_code=404, detail=f"Private bus for agent '{agent_id}' not found")
        
        # Publish message to private bus
        await private_bus.publish({
            "type": "user_message",
            "from": "user",
            "to": agent_id,
            "message": msg_req.message,
            "metadata": msg_req.metadata or {},
            "timestamp": datetime.now().isoformat()
        })
        
        logger.info(f"Sent message to agent '{agent_id}' private bus")
        
        return {
            "status": "success",
            "message": "Message sent to agent",
            "agent_id": agent_id
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error sending message to agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/agent/{agent_id}")
async def agent_websocket(websocket: WebSocket, agent_id: str):
    """
    WebSocket endpoint for real-time agent chat
    
    Connects AgentSlotControl Chat tab to the agent's private bus.
    Bidirectional: client sends messages, receives agent responses.
    """
    await websocket.accept()
    client_id = f"agent-slot-{agent_id}-{id(websocket)}"
    
    logger.info(f"Agent WebSocket connected: {client_id} for agent '{agent_id}'")
    
    try:
        if not triple_bus:
            await websocket.close(code=1011, reason="Triple bus not initialized")
            return
        
        # Get private bus for this agent
        private_bus = triple_bus.get_private(agent_id)
        
        if not private_bus:
            await websocket.close(code=1008, reason=f"Private bus for agent '{agent_id}' not found")
            return
        
        # Subscribe to private bus events
        async def handle_private_event(event: Dict[str, Any]):
            """Forward private bus events to WebSocket client"""
            try:
                await websocket.send_json({
                    "type": "agent_message",
                    "agent_id": agent_id,
                    "event": event
                })
            except Exception as e:
                logger.error(f"Error sending to WebSocket: {e}")
        
        private_bus.subscribe(handle_private_event)
        
        # Listen for client messages
        while True:
            data = await websocket.receive_json()
            
            # Handle incoming message from client
            if data.get("type") == "user_message":
                await private_bus.publish({
                    "type": "user_message",
                    "from": "user",
                    "to": agent_id,
                    "message": data.get("message", ""),
                    "metadata": data.get("metadata", {}),
                    "timestamp": datetime.now().isoformat()
                })
    
    except WebSocketDisconnect:
        logger.info(f"Agent WebSocket disconnected: {client_id}")
    except Exception as e:
        logger.error(f"Agent WebSocket error: {e}")
        try:
            await websocket.close(code=1011, reason=str(e))
        except:
            pass
