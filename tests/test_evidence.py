"""Test Evidence-based traceability system using pytest."""

from cogent.agents import ReActState
from cogent.kernel import Control, Result
from cogent.runtime.trace.evidence import Evidence
from fakes import make_fake_env


def test_evidence_creation():
    """Test basic Evidence creation."""
    evidence = Evidence("start")
    assert evidence.action == "start"
    assert evidence.parent_id is None


def test_evidence_hierarchy():
    """Test parent-child relationships."""
    parent = Evidence("parent")

    # With immutable design, child() returns a new Evidence object
    parent_with_child = parent.child("child")
    child = parent_with_child.children[0]

    assert child.action == "child"
    assert child.parent_id == parent.step_id
    assert len(parent_with_child.children) == 1
    assert parent_with_child.children[0] is child


def test_file_provenance():
    """Test 'who created this file' scenario."""
    evidence = Evidence("root")

    # Create file with author - using immutable design
    evidence_with_file = evidence.child("file.write", info={"author": "market_analyst", "timestamp": "now"})

    # Query who created the file
    creators = list(evidence_with_file.find_all(action="file.write"))

    assert len(creators) == 1
    assert creators[0].info["author"] == "market_analyst"


def test_side_effect_tracking():
    """Test external side effects are traceable."""
    evidence = Evidence("start")

    # Tool operations - using immutable design
    evidence = evidence.child("tool.search", info={"tool": "search_api"})
    evidence = evidence.child("tool.analyze", info={"tool": "ai_model"})

    # File operations
    evidence = evidence.child("file.write", info={"size": "1MB"})

    # Query operations - find_all uses **kwargs, not lambda
    file_operations = list(evidence.find_all(action="file.write"))
    tool_operations = [e for e in list(evidence.find_all()) if e.action.startswith("tool.")]

    assert len(file_operations) == 1
    assert len(tool_operations) == 2


def test_action_chaining():
    """Test evidence chains naturally."""
    react = Evidence("start")

    # ReAct pattern simulation - using immutable design
    # think is a NEW Evidence object with "reason" as a child
    react_with_think = react.child("reason", info={"phase": "thinking"})

    # search is a NEW Evidence object with "tool.search" as a child of the "reason" child
    react_with_search = react_with_think.child("tool.search", info={"tool": "intelligence"})

    # observe is a NEW Evidence object with "observe" as a child of the "tool.search" child
    react_with_observe = react_with_search.child("observe", info={"source": "reliable"})

    # observe_with_act is a NEW Evidence object with "act" as a child of the "observe" child
    react_with_act = react_with_observe.child("act", info={"action": "buy"})

    # Verify chain exists
    # Original react has no children (immutable)
    assert len(react.children) == 0
    
    # react_with_think should have one child (reason)
    assert len(react_with_think.children) == 1
    assert react_with_think.children[0].action == "reason"
    
    # react_with_search should have two children (reason and tool.search)
    assert len(react_with_search.children) == 2
    assert "reason" in [child.action for child in react_with_search.children]
    assert "tool.search" in [child.action for child in react_with_search.children]
    
    # react_with_observe should have three children
    assert len(react_with_observe.children) == 3
    assert "observe" in [child.action for child in react_with_observe.children]
    
    # react_with_act should have four children
    assert len(react_with_act.children) == 4
    assert "act" in [child.action for child in react_with_act.children]


def test_contextual_state():
    """Test ContextualState base functionality."""

    class TestState:
        def __init__(self, task=""):
            self.task = task
            self.evidence = None
            self.history = []
        
        def model_copy(self, *, update=None, deep=False):
            """Simple copy implementation for testing."""
            new_state = TestState(task=self.task)
            new_state.evidence = self.evidence
            new_state.history = self.history.copy()
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

    # Tool operations - using immutable design
    evidence = evidence.child("tool.search", info={"tool": "intelligence", "cost": 0.1})

    # File operations
    evidence = evidence.child("file.write", info={"author": "ai_analyst"})

    # Human interaction
    evidence = evidence.child("need_human_confirmation", info={"user": "reviewer"})

    # Sub-agent creation
    evidence = evidence.child("spawn_subagent", info={"parent": "main_agent"})

    # Verify results
    all_actions = list(evidence.find_all())
    assert len(all_actions) == 5  # root + search + file.write + human + delegate

    # File provenance check
    report_creators = [
        e for e in all_actions if e.action == "file.write" and e.info.get("author") == "ai_analyst"
    ]
    assert len(report_creators) == 1


def test_evidence_monad_basic():
    """Test EvidenceMonad basic functionality."""
    # Create traced state
    traced_state = ReActState()

    # Create monad
    monad = Result(traced_state, value="test_value", control=Control.Continue())

    assert monad.control.kind == "continue"
    assert monad.value == "test_value"
    assert monad.state.evidence.action == "start"


def test_traced_agent_state():
    """Test AgentState functionality."""
    traced_state = ReActState()

    # Add evidence through state
    new_state = traced_state.with_evidence(
        "tool.call", input_data={"tool": "search", "query": "test"}, info={"type": "lookup"}
    )

    assert new_state is not traced_state  # New state instance
    assert new_state.evidence.action == "start"  # Root evidence action remains "start"
    assert len(new_state.evidence.children) == 1  # Should have one child evidence
    child_evidence = new_state.evidence.children[0]
    assert child_evidence.action == "tool.call"  # Child evidence action is "tool.call"
    # Check the info structure in child evidence
    assert "type" in child_evidence.info
    assert child_evidence.info["type"] == "lookup"


def test_workflow_integration():
    """Test evidence in a realistic monadic workflow."""
    import asyncio
    from cogent.kernel import Agent

    # Initialize traced state
    initial_state = ReActState()

    async def process_market_data(state, _, env):
        _ = env
        new_state = state.with_evidence(
            "tool.market_api",
            input_data={"query": "market_data"},
            info={"api": "market_data_v1", "cost": 0.05},
        )
        return Result(new_state, value="processed_data", control=Control.Continue())

    async def apply_ml_model(state, data, env):
        _ = env
        new_state = state.with_evidence(
            "ml_model.apply",
            input_data=data,
            output_data={"trend": "bullish", "confidence": 0.85},
            info={"model": "trend_predictor_v2"},
        )
        return Result(new_state, value="analysis_results", control=Control.Continue())

    async def write_report(state, data, env):
        _ = env
        new_state = state.with_evidence(
            "file.write",
            input_data=data,
            output_data={"file": "market_report.pdf", "size": "1.2MB"},
            info={"author": "ai_analyst", "template": "market_analysis"},
        )
        return Result(new_state, value="report_generated", control=Control.Continue())

    async def run_workflow():
        # Execute workflow
        workflow = (
            Agent.start(initial_state, "initial_data")
            .then(process_market_data)
            .then(apply_ml_model)
            .then(write_report)
        )
        env = make_fake_env()
        return await workflow.run(env)

    result = asyncio.run(run_workflow())

    # Verify final evidence
    final_evidence = result.state.evidence
    assert final_evidence is not None

    # Verify the root evidence action remains "workflow_start"
    assert final_evidence.action == "start"
    
    # Verify we have 3 child evidences (market_api, ml_model.apply, file.write)
    assert len(final_evidence.children) == 3
    
    # Verify the final step evidence is in the children
    file_write_evidences = [e for e in final_evidence.children if e.action == "file.write"]
    assert len(file_write_evidences) == 1
    
    # Check the info structure in the file.write evidence
    file_write_evidence = file_write_evidences[0]
    assert "author" in file_write_evidence.info
    assert file_write_evidence.info["author"] == "ai_analyst"
    assert "template" in file_write_evidence.info
    assert file_write_evidence.info["template"] == "market_analysis"
