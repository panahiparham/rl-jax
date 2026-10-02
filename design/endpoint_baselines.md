# Unanchored and reservoir baselines for endpoint replay

Design for two baselines that keep endpoint replay's buffer layout but change
what the long-term buffer stores. Both reconstruct runs from coresets'
endpoint replay experiments and are added to `pinball_endpoint` and
`atari_50m_endpoint`. Builds on `design/composable_endpoint-buffer.md`.

## Summary

- Unanchored stores every 10th 1-step transition that leaves recency, instead
  of connected 10-step transitions.
- Reservoir stores 1-step transitions picked by reservoir sampling.
- Both train with DDQN on every row, with no expectile SARSA.
- Neither needs a new loss or a new selector. `DDQNAgent._train_step` on a
  `ComposedBuffer` already gives the original update, and
  `NStepSelector(n_step=1)` already gives 1-step transitions.
- New code: a `Subsample` pass-through stage, a `ReservoirBuffer` storage
  stage, two thin agents with their own configs, and 8 experiment
  components. Existing agents and the analysis notebooks are not changed.

## Original baselines

Source: coresets `src/buffers/CompositeBuffer.py`, `SubsampleBuffer.py`,
`ReservoirBuffer.py`, and `src/algorithms/nn/DDQN.py`.

| Setting | Unanchored | Reservoir |
| --- | --- | --- |
| Pinball, medium | `sarsa-test/pinball_1000/coreset_ddqn_onestep_squaredloss.json` | `sarsa-test/pinball_1000/coreset_ddqn_reservoir_composite.json` |
| Pinball, small | `sarsa-test/pinball_500/coreset_ddqn_onestep_squaredloss.json` | `sarsa-test/pinball_500/coreset_ddqn_reservoir_composite.json` |
| Atari, medium | `atari-50M-endpoint/truefinal/endpoint_medium_noexpectilesarsanstep.json` | `atari-50M-endpoint/truefinal/reservoir_medium.json` |
| Atari, small | `atari-50M-endpoint/truefinal/endpoint_small_noexpectilesarsanstep.json` | `atari-50M-endpoint/truefinal/reservoir_small.json` |

All eight use `agent: DDQN`, `buffer_type: composite_coreset`,
`coreset_nstep: 1`, and `recency_sample_number: 28` of a batch of 32.
Every other hyperparameter matches the corresponding endpoint run.

| Setting | Recency | Long-term |
| --- | --- | --- |
| Pinball, medium | 100 | 900 |
| Pinball, small | 100 | 400 |
| Atari, medium | 10,000 | 90,000 |
| Atari, small | 10,000 | 10,000 |

Data flow, shared by both:

1. Each timestep enters a lag-1 shadow buffer the size of recency.
2. Each 1-step transition the shadow buffer overwrites goes to the long-term
   buffer.

The long-term buffers differ:

- `SubsampleBuffer` keeps a transition when `trans_id % 10 == 0`. `trans_id`
  is a global counter over every 1-step transition, so the kept steps are
  every 10th step of the run, with no alignment to episodes.
- `ReservoirBuffer` keeps the first `N` transitions offered. After that, the
  `i`-th transition (counting from 0) draws `j` uniformly from `[0, i]` and
  replaces slot `j` if `j < N`.

Loss (`DDQN._composite_loss`): the double-Q target on recency and long-term
rows alike, with the mean of `0.5 * delta**2` over all 32 rows. While the
long-term buffer is empty, all 32 rows come from recency.

## Approach

`DDQNAgent._train_step` calls `self._buffer.sample`. On a `ComposedBuffer`,
that is the merged mixture: 28 recency rows and 4 long-term rows, with
recency filling in for long-term rows until the long-term buffer can sample.
The loss is the double-Q target on every row, averaged over valid rows. This
is the original update, apart from the rl-jax loss scale, which already
applies to every DDQN run. So the baselines only build a different buffer
chain:

```
endpoint:    recency -> NStepSelector(10) ->                TransitionBuffer
unanchored:  recency -> NStepSelector(1)  -> Subsample(10) -> TransitionBuffer
reservoir:   recency -> NStepSelector(1)  ->                ReservoirBuffer
```

`NStepSelector(n_step=1)` emits one 1-step transition per step:

- A termination or a life loss gives `discount = 0`.
- The truncating step is dropped, because its successor belongs to the next
  episode.

## Buffers

### Subsample

`src/components/buffers/subsample.py`. A pass-through stage that forwards
every `every`-th valid item unchanged.

- Config: `every: int`.
- State: `count`, the number of valid items seen so far.
- `push(state, item, valid)` forwards `item` with
  `out_valid = valid & (count % every == 0)`, then adds `valid` to `count`.
  Invalid items do not advance the count.
- `size` is 0. It is not sampleable.
- It works on any item type, so it is not tied to transitions.

The counter is global, as in the original, so the kept phase drifts across
episodes. Episode alignment matters for connected chunks, not for
disconnected 1-step transitions.

### ReservoirBuffer

`src/components/buffers/reservoir.py`. Storage like `TransitionBuffer`, but it
picks the slot to write by reservoir sampling.

- Config: `capacity`, `batch_size`.
- State: `data` (a `Batch` with leading axis `capacity`), `size`, `seen`
  (valid items offered so far), `key`, `slot` (where the next item goes, or
  `capacity` to drop it), and `outgoing` (the item at `slot`).
- `push(state, item, valid)` follows these rules:
  - The item is written to `slot` when `valid & (slot < capacity)`.
  - `out` is the stored `outgoing`, and `out_valid` is true when a stored
    item was replaced.
  - `seen` advances when `valid`.
  - The next `slot` is `seen` while `seen < capacity`. After that it is a
    uniform draw from `[0, seen]`, which keeps the item only when the draw is
    below `capacity`.
  - The next `outgoing` is read after the write, with a fresh key split from
    `key`.
- `size` is `min(seen, capacity)`.
- `sample` draws uniformly with replacement from `[0, size)`, and
  `can_sample` is `size > 0`, as in `TransitionBuffer`.

Choosing the next slot after each write, and reading the item there, follows
`TransitionBuffer`'s outgoing-in-state pattern. XLA then updates the storage
in place instead of copying it each step.

### Source of reservoir randomness

`push` gets no key, and `init` gets only the observation space.

`ReservoirBuffer.init` starts from `jax.random.key(0)`. `ReservoirAgent.init`
replaces that key with `jax.random.fold_in(key, 1)`, where `key` is the
agent's `init` key, by `_replace` on the reservoir stage state. The agent
built the chain, so it knows that stage's position. Each seed gets its own
reservoir randomness. The network-init key is unchanged, and so is the
`Buffer` contract.

## Agents and configs

Each baseline has its own module and config. Existing agents, including
`EndpointAgent` and `EndpointConfig`, are not changed.

### Configs

Each config subclasses `DDQNConfig` and repeats `EndpointConfig`'s Atari
defaults: `TOTAL_TIMESTEPS`, `LR`, `ADAM_EPS`, `BUFFER_SIZE`, `BATCH_SIZE`,
`LEARNING_STARTS`, `TRAIN_FREQUENCY`, `TARGET_NETWORK_FREQUENCY`,
`EPSILON_END`, `EPSILON_DECAY_STEPS`, `NETWORK_PRESET`, and `REWARD_CLIP`.
Each then adds its own long-term fields.

```python
# src/agents/unanchored.py
@dataclass(frozen=True, kw_only=True)
class UnanchoredConfig(DDQNConfig):
    """Unanchored replay's hyperparameters; defaults are for Atari experiments."""

    # ... the Atari defaults listed above
    LONG_TERM_SIZE: int = 90_000
    LONG_TERM_BATCH_SIZE: int = 4
    SUBSAMPLE: int = 10


# src/agents/reservoir.py
@dataclass(frozen=True, kw_only=True)
class ReservoirConfig(DDQNConfig):
    """Reservoir replay's hyperparameters; defaults are for Atari experiments."""

    # ... the Atari defaults listed above
    LONG_TERM_SIZE: int = 90_000
    LONG_TERM_BATCH_SIZE: int = 4
```

### Agents

- `UnanchoredAgent(DDQNAgent)` in `src/agents/unanchored.py` overrides
  `__init__` to build
  `[recency, NStepSelector(1), Subsample(SUBSAMPLE), TransitionBuffer]`.
- `ReservoirAgent(DDQNAgent)` in `src/agents/reservoir.py` overrides
  `__init__` to build `[recency, NStepSelector(1), ReservoirBuffer]`, and
  `init` to seed the reservoir key.
- Each `__init__` builds its own recency buffer with `build_buffer`, the same
  way `EndpointAgent` does.
- Both inherit `DDQNAgent._train_step`, so they need no loss code.
- They are registered in `src/agents/__init__.py` as `"unanchored"` and
  `"reservoir"`.

## Experiments

Each new component copies its setting's endpoint component, with the agent
and long-term fields changed. The existing components and their results are
not touched, and only the new components run.

| Experiment | Component | Agent | Long-term size | Runs |
| --- | --- | --- | --- | --- |
| `pinball_endpoint` | `unanchored_pinball` | unanchored | 900 | 100 |
| `pinball_endpoint` | `unanchored_small_pinball` | unanchored | 400 | 100 |
| `pinball_endpoint` | `reservoir_pinball` | reservoir | 900 | 100 |
| `pinball_endpoint` | `reservoir_small_pinball` | reservoir | 400 | 100 |
| `atari_50m_endpoint` | `unanchored_atari` | unanchored | 90,000 | 120 |
| `atari_50m_endpoint` | `unanchored_small_atari` | unanchored | 10,000 | 120 |
| `atari_50m_endpoint` | `reservoir_atari` | reservoir | 90,000 | 120 |
| `atari_50m_endpoint` | `reservoir_small_atari` | reservoir | 10,000 | 120 |

- Pinball: 100 seeds, `PinballConfig(SETTING="easy")`, recency 100, 4
  long-term rows per batch.
- Atari: 12 games, 10 seeds, `EPRAtariConfig`, recency 10,000, packed 4 runs
  per GPU. That makes 120 workers. A worker should take about 4.5 hours, close
  to DDQN, because the long-term storage per transition matches endpoint's.

The analysis notebooks are not changed. Plots for the new components come
later.

## Testing

- **`Subsample`:**
  - forwards exactly every k-th valid item, in order;
  - does not count invalid items;
  - forwards nothing when the input is always invalid;
  - in a chain, produces the same transitions as subsampling
    `NStepSelector(1)`'s output by hand.
- **`ReservoirBuffer`:**
  - stores the first `capacity` items in order;
  - stays at `capacity` afterwards;
  - treats an invalid push as a no-op;
  - keeps each item with probability close to `capacity / seen` over many
    keys (a statistical check with a deterministic seed);
  - reports a replaced item as `out` with `out_valid`.
- **Agents:**
  - both run under jit and vmap, and their long-term buffers fill;
  - two reservoir agents with different seeds fill their reservoirs
    differently;
  - both build from their default configs in the registry test.
- **Microbenchmark:** add a reservoir chain step next to the endpoint one,
  asserting an in-place update.

## Implementation sequence

1. Add `Subsample`, then its tests.
2. Add `ReservoirBuffer`, then its tests.
3. Add `UnanchoredConfig` and `UnanchoredAgent` with a registry entry, then
   tests.
4. Add `ReservoirConfig` and `ReservoirAgent` with a registry entry, then
   tests.
5. Add the reservoir microbenchmark.
6. Add the four Pinball components, then dispatch them.
7. Add the four Atari components, then dispatch them.
