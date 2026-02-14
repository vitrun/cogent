"""
Minimal example demonstrating the refactored agent runtime.

This example shows:
1. A step with internal retry_clean loop (step-level retry)
2. Concurrent execution returning list[Result]
3. Developer-defined resolution step
4. Trace enabled vs disabled

Design principles:
- Retry semantics are STRICTLY step-level
- No pipeline-level retry policy
- Runtime does NOT introduce global retry loops
- Each step is fully responsible for handling its own retry logic
- Runtime only interprets Control and records trace
- Concurrent composition is semantically neutral
"""

import asyncio
from typing import Any

from cogent import Agent, Control, Result, Env, TraceContext
from cogent.combinators import MultiState, MultiEnv, AgentRegistry, concurrent


# Define a simple state type for examples
SimpleState = str


# =============================================================================
# Fake model for testing (simulates LLM)
# =============================================================================
class FakeModel:
    async def complete(self, prompt: str) -> str:
        return f"Response to: {prompt[:50]}..."

    async def stream_complete(self, prompt: str, ctx: Any) -> str:
        await ctx.emit("stream-token-1")
        await ctx.emit("stream-token-2")
        return await self.complete(prompt)


def make_env(trace: TraceContext | None = None) -> Env:
    """Create a fake environment for testing."""
    return Env(model=FakeModel(), trace=trace)


# =============================================================================
# Example 1: Step with internal retry_clean loop
# =============================================================================
async def example_step_with_internal_retry() -> Result[SimpleState, str]:
    """
    Demonstrates step-level retry logic.
    The step itself handles the retry loop internally.
    Runtime does NOT retry - it only interprets Control.
    """

    async def step_with_retry(
        state: SimpleState,
        value: str,
        env: Env,
    ) -> Result[SimpleState, str]:
        """Step that retries up to 3 times internally."""
        initial_state = state
        max_retries = 3
        attempt = 0

        while attempt < max_retries:
            attempt += 1
            print(f"  Attempt {attempt}/{max_retries}")

            # Simulate work that might fail
            if attempt < 3:
                # First 2 attempts fail - retry internally
                print(f"    Failed, will retry...")
                # Continue the loop to retry
                continue
            else:
                # Third attempt succeeds
                return Result(
                    state=f"{initial_state}-success",
                    value=f"{value}-processed",
                    control=Control.Continue(),
                )

        # Should not reach here
        return Result(state=state, control=Control.Error("max retries exceeded"))

    # Create and run agent
    agent = Agent.start("initial", "input").then(step_with_retry)
    result = await agent.run(make_env(trace=None))

    print("\n--- Example 1: Step with internal retry ---")
    print(f"  State: {result.state}")
    print(f"  Value: {result.value}")
    print(f"  Control: {result.control.kind}")
    return result


# =============================================================================
# Example 2: Concurrent execution with raw branch Results
# =============================================================================
async def example_concurrent_branches() -> Result[MultiState, list[Result]]:
    """
    Demonstrates concurrent execution returning raw branch Results.
    Developer must explicitly write a step to interpret results.
    Runtime does NOT merge control - it returns raw Results.
    """

    # Create agents for parallel execution
    def make_branch_agent(branch_id: int):
        async def branch_step(s: MultiState, v: str, env) -> Result[MultiState, str]:
            new_state = MultiState(
                current=f"branch-{branch_id}",
                shared=s.shared + (f"branch-{branch_id}-done",),
                locals=s.locals,
            )
            return Result(
                state=new_state,
                value=f"branch-{branch_id}-result",
                control=Control.Continue(),
            )
        return Agent.start(MultiState(current=f"branch-{branch_id}")).then(branch_step)

    branch_1 = make_branch_agent(1)
    branch_2 = make_branch_agent(2)

    # Merge function for states
    def merge_states(states: list[MultiState]) -> MultiState:
        shared = ()
        for s in states:
            shared = shared + s.shared
        return MultiState(
            current="merged",
            shared=shared,
            locals={},
        )

    # Run concurrently
    concurrent_agent = concurrent([branch_1, branch_2], merge_states)
    result = await concurrent_agent.run(MultiEnv(model=FakeModel()))

    print("\n--- Example 2: Concurrent branches ---")
    print(f"  State: {result.state}")
    print(f"  Branch results: {len(result.value or [])} branches")

    # Developer-defined resolution step
    if result.value:
        print("  Resolving branch results:")
        for i, branch_result in enumerate(result.value):
            print(f"    Branch {i}: {branch_result.value} (control: {branch_result.control.kind})")

    return result


# =============================================================================
# Example 3: Trace enabled vs disabled
# =============================================================================
async def example_trace_enabled_vs_disabled() -> None:
    """
    Demonstrates trace enabled vs disabled performance characteristics.
    Trace disabled = single None check overhead.
    """

    async def simple_step(s: SimpleState, v: str, env: Env) -> Result[SimpleState, str]:
        return Result(state=f"{s}-done", value=v, control=Control.Continue())

    # With tracing disabled (None)
    agent = Agent.start("state", "value").then(simple_step)
    result_no_trace = await agent.run(make_env(trace=None))
    print("\n--- Example 3a: Trace disabled ---")
    print(f"  Result: {result_no_trace.value}")

    # With tracing enabled
    trace_ctx = TraceContext(enabled=True)
    agent = Agent.start("state", "value").then(simple_step)
    result_with_trace = await agent.run(make_env(trace=trace_ctx))
    print("\n--- Example 3b: Trace enabled ---")
    print(f"  Result: {result_with_trace.value}")
    print(f"  Events recorded: {len(trace_ctx)}")
    for event in trace_ctx.get_events():
        print(f"    - {event.action}: {event.info}")


# =============================================================================
# Example 4: Developer-defined control merge (NOT runtime responsibility)
# =============================================================================
async def example_developer_defined_merge() -> Result[MultiState, list[Result]]:
    """
    Demonstrates that runtime does NOT merge controls.
    Developer must explicitly write a step to interpret branch Results.
    """

    # Create agents with different control outcomes
    async def success_branch(s: MultiState, v: str, env) -> Result[MultiState, str]:
        return Result(state=s, value="success", control=Control.Continue())

    async def error_branch(s: MultiState, v: str, env) -> Result[MultiState, str]:
        return Result(state=s, value=None, control=Control.Error("branch failed"))

    async def retry_branch(s: MultiState, v: str, env) -> Result[MultiState, str]:
        return Result(state=s, value=None, control=Control.RetryClean())

    success_agent = Agent.start(MultiState(current="")).then(success_branch)
    error_agent = Agent.start(MultiState(current="")).then(error_branch)
    retry_agent = Agent.start(MultiState(current="")).then(retry_branch)

    def merge_states(states: list[MultiState]) -> MultiState:
        return MultiState(current="merged", shared=(), locals={})

    # Run concurrently - returns raw Results
    concurrent_agent = concurrent(
        [success_agent, error_agent, retry_agent],
        merge_states,
    )
    result = await concurrent_agent.run(MultiEnv(model=FakeModel()))

    print("\n--- Example 4: Developer-defined merge ---")
    print(f"  Raw branch results: {len(result.value or [])}")

    # Developer-defined resolution (NOT runtime)
    def resolve_results(results: list[Result[MultiState, Any]]) -> str:
        """Developer defines how to merge branch results."""
        errors = [r for r in results if r.control.kind == "error"]
        if errors:
            return f"Failed: {errors[0].control.reason}"
        successes = [r for r in results if r.control.kind == "continue"]
        return f"All succeeded: {[r.value for r in successes]}"

    resolution = resolve_results(result.value or [])
    print(f"  Resolution: {resolution}")

    return result  # type: ignore[return-value]


# =============================================================================
# Main
# =============================================================================
async def main():
    print("=" * 60)
    print("Cogent Agent Runtime - Minimal Example")
    print("=" * 60)

    await example_step_with_internal_retry()
    await example_concurrent_branches()
    await example_trace_enabled_vs_disabled()
    await example_developer_defined_merge()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
