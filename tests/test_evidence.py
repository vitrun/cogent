"""Test Evidence-based traceability system using pytest."""

from cogent.model import Evidence, AgentState
from cogent.agent import AgentResult


def test_evidence_creation():
    """Test basic Evidence creation."""
    evidence = Evidence("start")
    assert evidence.action == "start"
    assert evidence.parent_id is None


def test_evidence_hierarchy():
    """Test parent-child relationships."""
    parent = Evidence("parent")
    parent.children.clear()  # Ensure clean state

    child = parent.child("child")

    assert child.action == "child"
    assert child.parent_id == parent.step_id
    assert len(parent.children) == 1
    assert parent.children[0] is child


def test_file_provenance():
    """Test 'who created this file' scenario."""
    evidence = Evidence("root")

    # Create file with author
    evidence.child("file.write", info={"author": "market_analyst", "timestamp": "now"})

    # Query who created the file
    creators = list(evidence.find_all(action="file.write"))

    assert len(creators) == 1
    assert creators[0].info["author"] == "market_analyst"


def test_side_effect_tracking():
    """Test external side effects are traceable."""
    evidence = Evidence("start")

    # Tool operations - create but don't need variables
    evidence.child("tool.search", info={"tool": "search_api"})
    evidence.child("tool.analyze", info={"tool": "ai_model"})

    # File operations
    evidence.child("file.write", info={"size": "1MB"})

    # Query operations - find_all uses **kwargs, not lambda
    file_operations = list(evidence.find_all(action="file.write"))
    tool_operations = [e for e in list(evidence.find_all()) if e.action.startswith("tool.")]

    assert len(file_operations) == 1
    assert len(tool_operations) == 2


def test_action_chaining():
    """Test evidence chains naturally."""
    react = Evidence("start")

    # ReAct pattern simulation
    think = react.child("reason", info={"phase": "thinking"})

    search = think.child("tool.search", info={"tool": "intelligence"})

    observe = search.child("observe", info={"source": "reliable"})

    # Build act step but don't use act variable
    observe.child("act", info={"action": "buy"})

    # Verify chain exists
    assert len(react.children) == 1
    assert react.children[0].action == "reason"
    assert len(react.children[0].children) == 1
    assert react.children[0].children[0].action == "tool.search"


def test_contextual_state():
    """Test ContextualState base functionality."""

    class TestState(AgentState):
        def model_copy(self, *, update=None, deep=False):
            """Simple copy implementation for testing."""
            new_state = TestState(task="test_task")
            new_state.evidence = self.evidence
            new_state.history = self.history
            if update:
                for key, value in update.items():
                    setattr(new_state, key, value)
            return new_state

    state = TestState(task="test_task")
    state.evidence = Evidence("test")

    # Verify state has evidence
    assert state.evidence.action == "test"
    assert state.task == "test_task"


def test_all_traceability_requirements():
    """Test comprehensive traceability workflow."""
    evidence = Evidence("task")

    # Tool operations
    evidence.child("tool.search", info={"tool": "intelligence", "cost": 0.1})

    # File operations
    evidence.child("file.write", info={"author": "ai_analyst"})

    # Human interaction
    evidence.child("need_human_confirmation", info={"user": "reviewer"})

    # Sub-agent creation
    evidence.child("spawn_subagent", info={"parent": "main_agent"})

    # Verify results
    all_actions = list(evidence.find_all())
    assert len(all_actions) == 5  # root + search + report + human + delegate

    # File provenance check
    report_creators = [
        e for e in all_actions if e.action == "file.write" and e.info.get("author") == "ai_analyst"
    ]
    assert len(report_creators) == 1


def test_evidence_monad_basic():
    """Test EvidenceMonad basic functionality."""
    # Create traced state
    traced_state = AgentState()
    traced_state.evidence = Evidence("start")

    # Create monad
    monad = AgentResult(traced_state, "test_value")

    assert monad.valid
    assert monad.value == "test_value"
    assert monad.state.evidence.action == "start"


def test_evidence_monad_tracing():
    """Test evidence monad tracing."""
    traced_state = AgentState()
    traced_state.evidence = Evidence("start")

    monad = AgentResult(traced_state, "value")

    # Add trace
    traced = monad.trace("tool.call", input_data={"tool": "search"})

    assert traced.value == "value"  # Value unchanged
    assert traced.state.evidence.action == "tool.call"


def test_traced_agent_state():
    """Test AgentState functionality."""
    traced_state = AgentState()
    traced_state.evidence = Evidence("start")

    # Add evidence through state
    new_state = traced_state.with_evidence(
        "tool.call", input_data={"tool": "search", "query": "test"}, info={"type": "lookup"}
    )

    assert new_state is not traced_state  # New state instance
    assert new_state.evidence.action == "tool.call"
    # Check the info structure
    assert "type" in new_state.evidence.info
    assert new_state.evidence.info["type"] == "lookup"


def test_workflow_integration():
    """Test evidence in a realistic monadic workflow."""
    import asyncio
    from cogent.agent import Agent

    # Initialize traced state
    initial_state = AgentState(task="test_workflow")
    initial_state.evidence = Evidence("workflow_start")

    async def process_market_data(state, _):
        new_state = state.with_evidence(
            "tool.market_api",
            input_data={"query": "market_data"},
            info={"api": "market_data_v1", "cost": 0.05},
        )
        return AgentResult(new_state, "processed_data", valid=True)

    async def apply_ml_model(state, data):
        new_state = state.with_evidence(
            "ml_model.apply",
            input_data=data,
            output_data={"trend": "bullish", "confidence": 0.85},
            info={"model": "trend_predictor_v2"},
        )
        return AgentResult(new_state, "analysis_results", valid=True)

    async def write_report(state, data):
        new_state = state.with_evidence(
            "file.write",
            input_data=data,
            output_data={"file": "market_report.pdf", "size": "1.2MB"},
            info={"author": "ai_analyst", "template": "market_analysis"},
        )
        return AgentResult(new_state, "report_generated", valid=True)

    async def run_workflow():
        # Execute workflow
        workflow = (
            Agent.start(initial_state, "initial_data")
            .then(process_market_data)
            .then(apply_ml_model)
            .then(write_report)
        )
        return await workflow.run()

    result = asyncio.run(run_workflow())

    # Verify final evidence
    final_evidence = result.state.evidence
    assert final_evidence is not None

    # Verify the final step evidence
    assert final_evidence.action == "file.write"
    assert "author" in final_evidence.info
    assert final_evidence.info["author"] == "ai_analyst"
    assert "template" in final_evidence.info
    assert final_evidence.info["template"] == "market_analysis"
