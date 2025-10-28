"""
Collaboration Manager for Dexter-Gliksbot

Orchestrates multi-agent collaboration on the COLLAB bus. Dexter uses this to coordinate
idle agents through democratic workflows: observation → proposals → refinement → voting → consensus.

Architecture designed collaboratively by Claude (Anthropic) and Comet Assistant (Perplexity).

Key Features:
- Weighted voting with confidence thresholds (prevents "popular but uncertain" proposals)
- Dynamic timeouts (adapts to priority, agent count, historical performance)
- Observable state transitions (BSM tracks everything via COLLAB bus broadcasts)
- Nested collaborations (max depth 2 to prevent runaway recursion)
- Graceful degradation (handles partial participation and agent failures)
- Dexter intervention (tie-breaker, quality gatekeeper, ambiguity resolver)
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .triple_bus import TripleBusSystem, CollabTopic, get_global_triple_bus


class CollaborationPhase(str, Enum):
    """Phases of collaboration workflow"""
    OBSERVATION = "observation"          # Dexter broadcasts understanding
    PROPOSAL_COLLECTION = "proposals"    # Agents propose solutions
    REFINEMENT = "refinement"            # Improve proposals
    CRITIQUE = "critique"                # Identify issues
    VOTING = "voting"                    # Democratic decision
    CONSENSUS = "consensus"              # Finalize and execute
    PAUSED = "paused"                    # Temporarily suspended
    ABORTED = "aborted"                  # Cancelled/failed


class CollaborationPriority(str, Enum):
    """Priority levels for collaborations"""
    CRITICAL = "critical"    # User safety, data loss prevention
    HIGH = "high"           # User waiting, UX degradation
    NORMAL = "normal"       # Standard operations
    LOW = "low"            # Background optimization

# Base timeouts for each phase (Comet's baseline)
PHASE_TIMEOUTS = {
    CollaborationPhase.PROPOSAL_COLLECTION: 5.0,  # 5 seconds for proposals
    CollaborationPhase.REFINEMENT: 3.0,           # 3 seconds for refinements
    CollaborationPhase.CRITIQUE: 2.5,             # 2.5 seconds for critiques
    CollaborationPhase.VOTING: 2.0,               # 2 seconds for votes
}

# Priority multipliers for timeouts (Comet's design)
PRIORITY_MULTIPLIERS = {
    CollaborationPriority.CRITICAL: 0.5,    # Cut timeouts in half
    CollaborationPriority.HIGH: 0.75,       # 75% of normal
    CollaborationPriority.NORMAL: 1.0,      # Baseline
    CollaborationPriority.LOW: 1.5,         # 50% more time
}

# Nesting limits (prevent infinite recursion)
MAX_NESTING_DEPTH = 2  # Root = 0, child = 1, grandchild = 2


@dataclass
class CollaborationSession:
    """
    Stateful collaboration session.
    
    Observable via COLLAB bus - BSM tracks history, Dexter can intervene.
    """
    session_id: str
    observation: Dict[str, Any]
    invited_agents: List[str]
    phase: CollaborationPhase
    priority: CollaborationPriority
    
    # Nesting support (for sub-collaborations)
    parent_session_id: Optional[str] = None
    child_sessions: List[str] = field(default_factory=list)
    nesting_level: int = 0
    
    # Workflow data
    proposals: List[Dict[str, Any]] = field(default_factory=list)
    refinements: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)  # proposal_id -> refinements
    critiques: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)     # proposal_id -> critiques
    votes: List[Dict[str, Any]] = field(default_factory=list)
    consensus: Optional[Dict[str, Any]] = None
    
    # State management
    created_at: float = field(default_factory=time.time)
    phase_changed_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    
    # Interruption handling
    paused: bool = False
    pause_reason: Optional[str] = None
    queued_commands: List[Dict[str, Any]] = field(default_factory=list)
    
    # Observability (for BSM learning)
    phase_history: List[Tuple[CollaborationPhase, float]] = field(default_factory=list)
    
    # Timeout monitoring
    timeout_task: Optional[asyncio.Task] = None


class TimeoutStrategy:
    """
    Dynamic timeout calculation (Comet's multi-factor design).
    
    Factors:
    1. Priority (critical tasks move fast)
    2. Agent count (more agents = more time, capped at 2x)
    3. Historical performance (adaptive based on reality)
    """
    
    def __init__(self):
        self.historical_response_times: Dict[str, List[float]] = {}  # agent_id -> response times
    
    def calculate_timeout(
        self,
        phase: CollaborationPhase,
        priority: CollaborationPriority,
        agent_count: int,
        session_context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate dynamic timeout for phase"""
        base_timeout = PHASE_TIMEOUTS.get(phase, 3.0)
        
        # Factor 1: Task priority
        priority_multiplier = PRIORITY_MULTIPLIERS[priority]
        
        # Factor 2: Number of active agents (linear scaling, capped at 2x)
        scale_factor = min(1.0 + (agent_count - 1) * 0.2, 2.0)
        
        # Factor 3: Historical performance (adaptive)
        if session_context and "invited_agents" in session_context:
            avg_response_times = []
            for agent_id in session_context["invited_agents"]:
                if agent_id in self.historical_response_times:
                    recent_times = self.historical_response_times[agent_id][-10:]  # Last 10 responses
                    if recent_times:
                        avg_response_times.append(sum(recent_times) / len(recent_times))
            
            if avg_response_times:
                avg_agent_response = sum(avg_response_times) / len(avg_response_times)
                adaptive_factor = avg_agent_response / base_timeout
                adaptive_factor = max(0.5, min(adaptive_factor, 2.0))  # Cap at 0.5x - 2x
            else:
                adaptive_factor = 1.0
        else:
            adaptive_factor = 1.0
        
        final_timeout = base_timeout * priority_multiplier * scale_factor * adaptive_factor
        return max(1.0, final_timeout)  # Minimum 1 second
    
    def record_response_time(self, agent_id: str, response_time: float):
        """Record agent response time for future adaptive calculations"""
        if agent_id not in self.historical_response_times:
            self.historical_response_times[agent_id] = []
        
        self.historical_response_times[agent_id].append(response_time)
        
        # Keep only last 50 responses per agent (prevent memory bloat)
        if len(self.historical_response_times[agent_id]) > 50:
            self.historical_response_times[agent_id] = self.historical_response_times[agent_id][-50:]


class CollaborationManager:
    """
    Orchestrates multi-agent collaboration on COLLAB bus.
    
    Used by Dexter to coordinate idle agents through democratic workflows.
    Observable by BSM for learning collaboration patterns.
    """
    
    def __init__(self, buses: Optional[TripleBusSystem] = None):
        from ..configs import get_global_config
        self.buses = buses or get_global_triple_bus()
        self.active_sessions: Dict[str, CollaborationSession] = {}
        self.session_lock = asyncio.Lock()
        self.timeout_strategy = TimeoutStrategy()
        self.config_manager = get_global_config()

        # Agent performance tracking (for selection hints, not penalties)
        self.agent_response_rates: Dict[str, Dict[str, int]] = {}  # agent_id -> {responded, invited}
    
    # ========== SESSION LIFECYCLE ==========
    
    async def start_collaboration(
        self,
        observation: Dict[str, Any],
        invited_agents: List[str],
        priority: CollaborationPriority = CollaborationPriority.NORMAL,
        parent_session_id: Optional[str] = None
    ) -> str:
        """
        Start new collaboration session.
        
        Returns:
            session_id for tracking
        
        Raises:
            ValueError if nesting depth exceeded
        """
        async with self.session_lock:
            # Check nesting depth
            nesting_level = 0
            if parent_session_id:
                parent = self.active_sessions.get(parent_session_id)
                if not parent:
                    raise ValueError(f"Parent session {parent_session_id} not found")
                if parent.nesting_level >= MAX_NESTING_DEPTH:
                    raise ValueError(f"Max nesting depth ({MAX_NESTING_DEPTH}) exceeded")
                nesting_level = parent.nesting_level + 1
            
            # Create session
            session = CollaborationSession(
                session_id=str(uuid.uuid4()),
                observation=observation,
                invited_agents=invited_agents,
                phase=CollaborationPhase.OBSERVATION,
                priority=priority,
                parent_session_id=parent_session_id,
                nesting_level=nesting_level
            )
            
            # Link parent-child relationship
            if parent_session_id:
                self.active_sessions[parent_session_id].child_sessions.append(session.session_id)
            
            self.active_sessions[session.session_id] = session
            
            # Broadcast collaboration started (observable by BSM)
            await self.buses.collab.publish(CollabTopic.OBSERVATION, {
                "event": "collaboration_started",
                "session_id": session.session_id,
                "observation": observation,
                "invited_agents": invited_agents,
                "priority": priority.value,
                "nesting_level": nesting_level,
                "timestamp": session.created_at
            })
            
            return session.session_id
    
    async def _transition_phase(
        self,
        session_id: str,
        new_phase: CollaborationPhase
    ):
        """
        Transition to new phase and broadcast to COLLAB bus (Comet's observable pattern).
        
        BSM observes these transitions to learn collaboration patterns.
        """
        session = self.active_sessions[session_id]
        old_phase = session.phase
        
        # Cancel existing timeout task if any
        if session.timeout_task and not session.timeout_task.done():
            session.timeout_task.cancel()
        
        # Update phase
        session.phase = new_phase
        session.phase_changed_at = time.time()
        session.phase_history.append((new_phase, session.phase_changed_at))
        
        # Broadcast phase transition (OBSERVABLE)
        await self.buses.collab.publish(CollabTopic.OBSERVATION, {
            "event": "phase_transition",
            "session_id": session_id,
            "old_phase": old_phase.value,
            "new_phase": new_phase.value,
            "timestamp": session.phase_changed_at,
            "nesting_level": session.nesting_level
        })
    
    async def _abort_collaboration(
        self,
        session_id: str,
        reason: str
    ):
        """Abort collaboration immediately"""
        if session_id not in self.active_sessions:
            return
        
        session = self.active_sessions[session_id]
        
        # Cancel timeout task
        if session.timeout_task and not session.timeout_task.done():
            session.timeout_task.cancel()
        
        # Mark as aborted
        await self._transition_phase(session_id, CollaborationPhase.ABORTED)
        session.completed_at = time.time()
        
        # Broadcast abortion
        await self.buses.collab.publish(CollabTopic.OBSERVATION, {
            "event": "collaboration_aborted",
            "session_id": session_id,
            "reason": reason,
            "timestamp": session.completed_at
        })
        
        # Clean up
        async with self.session_lock:
            del self.active_sessions[session_id]
    
    # ========== WORKFLOW ORCHESTRATION ==========
    
    async def broadcast_observation(
        self,
        session_id: str,
        observation: Dict[str, Any]
    ):
        """
        Phase 1: Dexter broadcasts understanding to idle agents.
        
        This starts the collaboration workflow.
        """
        session = self.active_sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        # Transition to observation phase
        await self._transition_phase(session_id, CollaborationPhase.OBSERVATION)
        
        # Broadcast observation to COLLAB bus
        await self.buses.collab.publish(CollabTopic.OBSERVATION, {
            "session_id": session_id,
            "observation": observation,
            "invited_agents": session.invited_agents,
            "priority": session.priority.value,
            "timestamp": time.time()
        })
        
        # Automatically move to proposal collection
        await asyncio.sleep(0.5)  # Give agents time to process observation
        await self._transition_phase(session_id, CollaborationPhase.PROPOSAL_COLLECTION)
    
    async def collect_proposals(
        self,
        session_id: str
    ) -> List[Dict[str, Any]]:
        """
        Phase 2: Collect proposals from invited agents.
        
        Uses Comet's asyncio.gather() pattern with return_exceptions=True for robustness.
        Handles partial participation gracefully.
        """
        session = self.active_sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        # Calculate dynamic timeout
        timeout = self.timeout_strategy.calculate_timeout(
            phase=CollaborationPhase.PROPOSAL_COLLECTION,
            priority=session.priority,
            agent_count=len(session.invited_agents),
            session_context={"invited_agents": session.invited_agents}
        )
        
        # Start timeout monitoring with 75% warning (Comet's suggestion)
        session.timeout_task = asyncio.create_task(
            self._monitor_phase_timeout(session_id, CollaborationPhase.PROPOSAL_COLLECTION, timeout)
        )
        
        # Broadcast proposal request
        await self.buses.collab.publish(CollabTopic.PROPOSAL, {
            "event": "proposal_request",
            "session_id": session_id,
            "timeout": timeout,
            "invited_agents": session.invited_agents,
            "timestamp": time.time()
        })
        
        # Collect proposals with timeout
        proposals = []
        responses_received = []
        
        try:
            # Create listeners for each agent
            proposal_queue = asyncio.Queue()
            
            async def proposal_listener(msg):
                if msg.get("session_id") == session_id and msg.get("event") == "proposal_response":
                    await proposal_queue.put(msg)
            
            self.buses.collab.subscribe(CollabTopic.PROPOSAL, proposal_listener)
            
            try:
                # Wait for responses with timeout
                start_time = time.time()
                while time.time() - start_time < timeout:
                    remaining = timeout - (time.time() - start_time)
                    if remaining <= 0:
                        break
                    
                    try:
                        response = await asyncio.wait_for(proposal_queue.get(), timeout=remaining)
                        proposals.append(response["proposal"])
                        responses_received.append(response.get("from_agent"))
                        
                        # Record response time for adaptive timeouts
                        response_time = time.time() - start_time
                        self.timeout_strategy.record_response_time(
                            response.get("from_agent", "unknown"),
                            response_time
                        )
                        
                        # All agents responded?
                        if len(responses_received) >= len(session.invited_agents):
                            break
                    except asyncio.TimeoutError:
                        break
            finally:
                self.buses.collab.unsubscribe(CollabTopic.PROPOSAL, proposal_listener)
        
        finally:
            # Cancel timeout monitor
            if session.timeout_task and not session.timeout_task.done():
                session.timeout_task.cancel()
        
        # Handle partial participation
        participation_rate = len(responses_received) / len(session.invited_agents) if session.invited_agents else 0
        
        # Track response rates for future agent selection
        for agent_id in session.invited_agents:
            if agent_id not in self.agent_response_rates:
                self.agent_response_rates[agent_id] = {"responded": 0, "invited": 0}
            
            self.agent_response_rates[agent_id]["invited"] += 1
            if agent_id in responses_received:
                self.agent_response_rates[agent_id]["responded"] += 1
        
        # Store proposals in session
        session.proposals = proposals
        
        # Broadcast collection result
        await self.buses.collab.publish(CollabTopic.OBSERVATION, {
            "event": "proposals_collected",
            "session_id": session_id,
            "proposal_count": len(proposals),
            "participation_rate": participation_rate,
            "timestamp": time.time()
        })
        
        return proposals
    
    async def _monitor_phase_timeout(
        self,
        session_id: str,
        phase: CollaborationPhase,
        timeout: float
    ):
        """
        Monitor phase timeout with 75% warning (Comet's design).
        
        Broadcasts warning at 75% elapsed, then handles timeout if reached.
        """
        warning_time = timeout * 0.75
        
        try:
            # Sleep until warning
            await asyncio.sleep(warning_time)
            
            # Broadcast warning
            await self.buses.collab.publish(CollabTopic.OBSERVATION, {
                "event": "deadline_warning",
                "session_id": session_id,
                "phase": phase.value,
                "remaining_seconds": timeout - warning_time,
                "timestamp": time.time()
            })
            
            # Wait remaining time
            await asyncio.sleep(timeout - warning_time)
            
            # Timeout reached - handle it
            await self._handle_phase_timeout(session_id, phase)
            
        except asyncio.CancelledError:
            # Phase completed early, cancel timeout
            pass
    
    async def _handle_phase_timeout(
        self,
        session_id: str,
        phase: CollaborationPhase
    ):
        """Handle timeout for a phase"""
        # Just log it - collection methods handle partial participation
        await self.buses.collab.publish(CollabTopic.OBSERVATION, {
            "event": "phase_timeout",
            "session_id": session_id,
            "phase": phase.value,
            "timestamp": time.time()
        })
    
    # ========== CONSENSUS & VOTING ==========
    
    async def call_vote(
        self,
        session_id: str,
        proposals: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Phase 4: Call for votes on proposals.
        
        Each agent votes with proposal_id + confidence (0.0-1.0).
        """
        session = self.active_sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        await self._transition_phase(session_id, CollaborationPhase.VOTING)
        
        # Calculate timeout
        timeout = self.timeout_strategy.calculate_timeout(
            phase=CollaborationPhase.VOTING,
            priority=session.priority,
            agent_count=len(session.invited_agents)
        )
        
        # Broadcast vote request
        await self.buses.collab.publish(CollabTopic.VOTE_REQUEST, {
            "session_id": session_id,
            "proposals": proposals,
            "timeout": timeout,
            "timestamp": time.time()
        })
        
        # Collect votes (similar pattern to proposals)
        votes = []
        vote_queue = asyncio.Queue()
        
        async def vote_listener(msg):
            if msg.get("session_id") == session_id:
                await vote_queue.put(msg)
        
        self.buses.collab.subscribe(CollabTopic.VOTE_RESPONSE, vote_listener)
        
        try:
            start_time = time.time()
            while time.time() - start_time < timeout:
                remaining = timeout - (time.time() - start_time)
                if remaining <= 0:
                    break
                
                try:
                    vote = await asyncio.wait_for(vote_queue.get(), timeout=remaining)
                    votes.append(vote)
                    
                    if len(votes) >= len(session.invited_agents):
                        break
                except asyncio.TimeoutError:
                    break
        finally:
            self.buses.collab.unsubscribe(CollabTopic.VOTE_RESPONSE, vote_listener)
        
        session.votes = votes
        return votes
    
    async def reach_consensus(
        self,
        session_id: str
    ) -> Dict[str, Any]:
        """
        Phase 5: Reach consensus based on weighted voting.
        
        Uses confidence thresholds (Comet's insight) to prevent "popular but uncertain" proposals.
        Dexter acts as tie-breaker and quality gatekeeper.
        """
        session = self.active_sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        await self._transition_phase(session_id, CollaborationPhase.CONSENSUS)
        
        # Tally weighted votes
        vote_results: Dict[str, Dict[str, Any]] = {}
        total_weight = 0.0
        
        for vote in session.votes:
            proposal_id = vote.get("proposal_id")
            voter_confidence = vote.get("confidence", 0.5)
            
            if proposal_id not in vote_results:
                vote_results[proposal_id] = {
                    "votes": 0,
                    "weight": 0.0,
                    "voters": []
                }
            
            vote_results[proposal_id]["votes"] += 1
            vote_results[proposal_id]["weight"] += voter_confidence
            vote_results[proposal_id]["voters"].append(vote.get("from_agent"))
            total_weight += voter_confidence
        
        # Normalize weights to percentages
        for proposal_id in vote_results:
            vote_results[proposal_id]["weight_percent"] = (
                vote_results[proposal_id]["weight"] / total_weight if total_weight > 0 else 0
            )
        
        # Find winner
        winner_id = max(vote_results.keys(), key=lambda pid: vote_results[pid]["weight"]) if vote_results else None
        
        if not winner_id:
            # No votes - Dexter decides
            consensus_result = {
                "status": "no_consensus",
                "reason": "No votes received",
                "dexter_decision": "required"
            }
        else:
            # Find the actual proposal
            winner_proposal = next((p for p in session.proposals if p.get("id") == winner_id), None)
            
            if not winner_proposal:
                consensus_result = {
                    "status": "error",
                    "reason": "Winner proposal not found"
                }
            else:
                # Validate consensus using Comet's thresholds
                valid, reason = await self._validate_consensus(winner_proposal, vote_results[winner_id], len(session.votes))
                
                if valid:
                    consensus_result = {
                        "status": "consensus_reached",
                        "winner": winner_proposal,
                        "vote_results": vote_results,
                        "confidence": winner_proposal.get("confidence", 0.5),
                        "support_percent": vote_results[winner_id]["weight_percent"]
                    }
                else:
                    # Consensus invalid - Dexter intervenes
                    consensus_result = {
                        "status": "consensus_failed",
                        "reason": reason,
                        "winner": winner_proposal,
                        "vote_results": vote_results,
                        "dexter_decision": "required"
                    }
        
        session.consensus = consensus_result
        session.completed_at = time.time()
        
        # Broadcast consensus
        await self.buses.collab.publish(CollabTopic.CONSENSUS, {
            "session_id": session_id,
            "consensus": consensus_result,
            "timestamp": session.completed_at
        })
        
        return consensus_result
    
    async def _validate_consensus(
        self,
        winner: Dict[str, Any],
        vote_result: Dict[str, Any],
        total_voters: int
    ) -> Tuple[bool, str]:
        """
        Validate winning proposal meets quality thresholds (Comet's design).
        
        Returns:
            (valid, reason)
        """
        # Check: Enough votes?
        winning_threshold = self.config_manager.get("collaboration.consensus.winning_threshold", 0.6)
        if vote_result["weight_percent"] < winning_threshold:
            return False, f"Insufficient votes ({vote_result['weight_percent']:.1%} < {winning_threshold:.1%})"
        
        # Check: Confident enough? (Comet's key insight)
        confidence_floor = self.config_manager.get("collaboration.consensus.confidence_floor", 0.7)
        if winner.get("confidence", 0.0) < confidence_floor:
            return False, f"Low confidence ({winner.get('confidence', 0.0):.2f} < {confidence_floor})"
        
        # Check: Enough participants?
        min_participants = self.config_manager.get("collaboration.consensus.min_participants", 2)
        if total_voters < min_participants:
            return False, f"Too few participants ({total_voters} < {min_participants})"
        
        return True, "Consensus valid"
    
    # ========== INTERRUPTION HANDLING ==========
    
    async def handle_user_interrupt(
        self,
        session_id: str,
        priority: CollaborationPriority,
        command: Dict[str, Any]
    ):
        """
        Handle user command during active collaboration (our 3-tier design).
        
        - CRITICAL: Abort immediately
        - HIGH: Pause and save state
        - NORMAL/LOW: Queue for after consensus
        """
        session = self.active_sessions.get(session_id)
        if not session:
            return
        
        if priority == CollaborationPriority.CRITICAL:
            # ABORT: Cancel immediately
            await self._abort_collaboration(session_id, reason="critical_user_interrupt")
        
        elif priority == CollaborationPriority.HIGH:
            # PAUSE: Save state
            session.paused = True
            session.pause_reason = "high_priority_command"
            
            await self._transition_phase(session_id, CollaborationPhase.PAUSED)
            
            await self.buses.collab.publish(CollabTopic.OBSERVATION, {
                "event": "collaboration_paused",
                "session_id": session_id,
                "reason": session.pause_reason,
                "timestamp": time.time()
            })
        
        else:
            # QUEUE: Let collaboration finish
            session.queued_commands.append(command)
    
    # ========== OBSERVABILITY & STATS ==========
    
    def get_session_stats(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a session (for BSM learning)"""
        session = self.active_sessions.get(session_id)
        if not session:
            return None
        
        duration = (session.completed_at or time.time()) - session.created_at
        
        return {
            "session_id": session_id,
            "phase": session.phase.value,
            "priority": session.priority.value,
            "nesting_level": session.nesting_level,
            "duration_seconds": duration,
            "proposal_count": len(session.proposals),
            "vote_count": len(session.votes),
            "consensus_reached": session.consensus is not None,
            "phase_count": len(session.phase_history),
            "paused": session.paused
        }
    
    def get_agent_performance_hints(self) -> Dict[str, float]:
        """
        Get agent response rates (hints for selection, not penalties).
        
        Comet's wisdom: Use for time-critical collaborations but don't exclude anyone.
        """
        hints = {}
        for agent_id, stats in self.agent_response_rates.items():
            if stats["invited"] > 0:
                hints[agent_id] = stats["responded"] / stats["invited"]
        return hints
