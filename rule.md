## 0. Goal

This project implements a functional / monadic agent runtime (cogent).
All implementations must revolve around the core abstraction of "composable state transformation", rather than around class inheritance, callback stitching, or implicit side effects.

The system's minimal abstraction unit is:

Step : State -> Effect[State]

Any design that does not satisfy this form should be considered a potential violation of architectural principles.

## 1. Core Design Principles (Non-negotiable)
### 1.1 Single Data Channel

There can only be one explicit data flow in the system:

State  -->  Step  -->  State

Prohibited:

- Implicit global variables
- Mutable shared contexts
- Side-channel data transfer
- Concealing state through closures between Steps

All state must be explicitly passed through State.

### 1.2 Single Control Abstraction

Control flow can only be expressed through monadic composition:

- .then
- .map
- .flat_map
- .recover
- .guard
- .branch

Prohibited:

- Writing flow control inside Steps (e.g., while True main loop)
- Making a component the "dispatch center"
- Maintaining additional control stacks outside the runtime

Control flow must become "composable values".

### 1.3 Steps Must Be Pure Functional Boundaries

Each Step:

- Input: Immutable State
- Output: New State (encapsulated in Effect)
- Must not modify input State
- Must not hold internal cached state

Side effects (LLM calls, tool calls, IO) must:

- Be encapsulated within Effect
- Be explicitly declared
- Be replaceable with mocks

### 1.4 Explicit Error Propagation (Left Bias)

Errors must propagate through monadic short-circuiting.

Prohibited:

- Swallowing exceptions after try/except
- Using flags to indicate failure
- Writing error fields in State for external judgment

Once failed:

- Subsequent steps must not execute
- Failure must propagate up the pipeline

### 1.5 No "Implicit Main Loop"

Prohibited:

```python
while True:
    think()
    act()
    observe()
```

Must be expressed as:

```python
step_think
.then(step_act)
.then(step_observe)
.branch(...)
```

Loops must be constructed through recursive pipelines, not imperative loops.

## 2. State Modeling Principles
### 2.1 State Is the Complete World

State must include:

- User input
- Historical messages
- Current thinking
- Tool call information
- Observation results
- Termination marker

Prohibited:

- Hiding partial state in runtime
- Hiding history in agent internal properties

State is the system's single source of truth.

### 2.2 State Must Be Immutable

Recommended:

- dataclass(frozen=True)
- Or explicit copy-on-write

Prohibited: In-place modification of fields.

### 2.3 State Transformations Must Be Explicit

Not allowed:

```python
state.messages.append(...)
```

Must be:

```python
new_state = state.with_messages(state.messages + [msg])
```

## 3. ReAct Behavior Replication Specification

ReAct behavior must be decomposed into the following independent Steps:

- build_prompt_step
- llm_inference_step
- parse_output_step
- decide_branch_step
- tool_execution_step
- observation_append_step
- termination_check_step

Each step:

- Does only one thing
- Does not cross responsibility boundaries
- Does not mix LLM and tool

## 4. Effect Design Principles

Effect must:

- Clearly distinguish between pure and async
- Support mocks
- Support tracing
- Support cancellation

Recommended interface:

- Effect.run()
- Effect.map()
- Effect.flat_map()
- Effect.recover()

Prohibited:

- Directly awaiting external resources within Steps
- Directly calling LLM client outside the runtime

## 5. Failure-First Principle (Engineering Constraint)

Before implementing any new capability, you must first answer:

- At which stage will the system fail earliest?

Common failure points:

- Unparsable LLM output
- Tool schema mismatch
- Context loss due to state explosion
- Recursive pipeline runaways

Each new feature must:

- Identify the most likely failure point
- Provide a protection strategy
- Provide testable use cases

## 6. Prohibited Patterns

The following patterns are strictly prohibited:

❌ Agent class holding mutable state
```python
class Agent:
    self.memory = []
```

❌ Runtime becoming the brain
```python
runtime.run(agent)
```

❌ Central dispatcher pattern
```python
if state.phase == "think":
```

❌ Communication between Steps through shared objects

## 7. Testing Specification

Each Step must:

- Be testable in isolation
- Not depend on external networks
- Allow injection of mock LLM

Must cover:

- Normal path
- LLM output exception path
- Tool failure path
- Termination path

## 8. Iteration Principles

Priority order:

1. Correctness
2. Composability
3. Testability
4. Performance

Do not sacrifice structural integrity for performance.

## 9. Final Judgment Criteria

After implementation, it must satisfy:

- Any Step can be plugged in
- Any Step can be replaced
- Any failure automatically short-circuits
- Any pipeline can be nested
- No hidden control flow
- No hidden state

If any of the above cannot be satisfied, it is considered a violation of the cogent architecture.