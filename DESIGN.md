# 0. Goal

Cogent is a compositional agent kernel with product-level ergonomics. It is a minimal algebra for state transformation inspired by [Monadic Context Engineering](https://arxiv.org/pdf/2512.22431).

The system must balance:
- Mathematical integrity (kernel)
- Architectural boundary discipline
- Practical usability

No feature may sacrifice architectural clarity for convenience.

# 1. Architectural Layers (Strict Boundaries)

cogent is divided into layers:

- kernel        → compositional algebra
- ports         → external world interfaces
- agents        → product-facing strategies
- combinators   → higher-order agent composition
- providers     → model adapters
- structured    → typed output extension

## 1.1 Dependency Direction (Non-negotiable)

Dependencies must flow downward only:

agents
↓
combinators
↓
kernel
↓
(no dependency)

kernel must not depend on:

- providers
- concrete tool implementations
- model-specific formats

If dependency direction is violated, the change must be rejected.


# 2. Kernel Principles (Sacred Zone)

The kernel defines how composition works.
It does NOT define what agents do.

Kernel must remain:

- Minimal
- Abstract
- Strategy-free
- Business-logic-free

Prohibited in kernel:

- ReAct logic
- Built-in retry loops
- Tool registry logic
- Memory persistence logic
- Model formatting

Kernel evolution must be rare and deliberate.


# 3. Single Data Channel

There is exactly one explicit data channel:

State → Transformation → State

All data must flow through State.

Prohibited:

- Hidden shared memory
- Global mutable variables
- Cross-step side-channel mutation
- Injecting state via closures

State is the single source of truth.


# 4. Control as Composition

Control flow must be expressed as composable values.

The system must not:

- Embed while-loops as implicit drivers
- Maintain external execution stacks
- Introduce hidden dispatch centers

Agents are values.
Composition is structure.
Execution is interpretation.


# 5. Retry Semantics (Explicit Decision)

Retry is a Step-level concern.

Retry must be represented structurally in the pipeline,
not implemented as an outer execution loop.

Correct:
```
Step → Result(Control=Retry) → Recomposition
```
Incorrect:
```
while retry:
    run_step()
```
Retry must:

- Be explicit in Result / Control
- Preserve state integrity
- Not hide failure history

Retry logic must remain composable.


# 6. Memory vs Context (Critical Distinction)

Memory and Context are NOT the same concept.

Memory:
- Persistent knowledge storage
- Long-lived
- External to a single execution
- Accessed via ports
- Replaceable implementation (e.g., in-memory, vector DB)

Context:
- Execution-local accumulation of state
- Ephemeral
- Exists only within one agent run
- Fully represented inside State

Rules:

- Memory must not mutate Context implicitly
- Context must not reach into Memory directly
- Memory access must go through explicit ports
- Context must remain deterministic and serializable

If these boundaries blur, architecture degrades.


# 7. Effect Boundaries

All side effects (LLM calls, tool calls, IO):

- Must be isolated behind ports
- Must be mockable
- Must not leak into kernel logic

Side effects must never define architecture.


# 8. Combinators Discipline

Combinators compose Agents.

They must:

- Preserve closure (Agent → Agent)
- Not become workflow engines
- Not implement DAG orchestration systems

If combinators begin to resemble a workflow framework,
the design must be reconsidered.


# 9. Product Layer Discipline

Agents layer provides product-facing usability.

It must:

- Expose clear strategy types (e.g., ReActAgent)
- Provide sensible defaults
- Minimize required configuration
- Hide kernel complexity from normal users

Prohibited:

- Forcing users to construct Env manually
- Exposing combinators in beginner workflows
- Requiring understanding of monadic internals

Kernel purity must not degrade usability.


# 10. Error Semantics

Failure must be structural.

Errors must:

- Propagate deterministically
- Stop downstream execution
- Avoid flag-based signaling
- Avoid writing error markers into State

Short-circuiting must be algebraic, not procedural.


# 11. Anti-Patterns

Do not introduce:

- Hidden execution loops
- State mutation outside transformation
- Global registries inside kernel
- Strategy logic inside kernel
- Implicit retry outside Control

Convenience that weakens boundaries must be rejected.


# 12. Stability Gradient

Stability increases downward:

kernel      → most stable
combinators
agents
providers

Breaking changes are increasingly unacceptable as you move downward.


# 13. Design Decision Checklist

Before merging any feature, ask:

1. Does this belong in kernel?
2. Does this violate dependency direction?
3. Does this blur Memory and Context?
4. Is retry implemented structurally or procedurally?
5. Does this increase cognitive load for normal users?
6. Can this be implemented at a higher layer instead?

If unsure, implement at the highest possible layer.


# 14. What We Optimize For

cogent optimizes for:

- Compositional clarity
- Architectural boundary integrity
- Product-level usability
- Debuggability
- Long-term maintainability