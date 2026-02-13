"""Combinator laws and algebra documentation."""

# Combinators satisfy the following algebraic laws:
#
# 1. Identity: agent.then(Agent.start) == agent
#    An agent bound to identity returns the same agent
#
# 2. Associativity: (agent.then(f)).then(g) == agent.then(lambda x: f(x).then(g))
#    Chaining steps is associative
#
# 3. Handoff is transitive: handoff("a").then(handoff("b")) == handoff("b")
#    Multiple handoffs can be collapsed
#
# 4. Concurrent is commutative: concurrent([a, b], merge) == concurrent([b, a], merge)
#    Order of agents doesn't matter for concurrent execution
#
# 5. Emit is idempotent: emit(x).then(emit(x)) == emit(x)
#    Duplicate emissions are absorbed
