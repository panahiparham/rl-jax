# Composable replay buffers and endpoint replay

Design for composable replay buffers. It covers a unified buffer contract,
three new buffers (N-step selector, transition buffer, composed buffer), and an
endpoint replay agent built from them.

## Summary

- Every buffer implements one functional contract: `init`, `push`, `size`, and,
  if it can be sampled, `batch_size`, `sample`, `can_sample`.
- `push` takes an item plus a `valid` flag and returns the item that aged out of
  the buffer, plus its own `valid` flag. Composition passes each buffer's
  outgoing items to the next buffer.
- `ReplayBuffer` keeps today's `add`, `sample`, and `can_sample` unchanged. It
  gains `push`, which calls `add` and also returns the timestep leaving the
  sampleable range, with its stacked observation.
- `Batch` gains `boot_action`, the action taken at the bootstrap observation.
- `ComposedBuffer` chains buffers. It samples mixture batches, merged or per
  buffer, with each buffer's constructor batch size. While a buffer cannot
  sample yet, the first sampleable buffer fills its rows.
- `EndpointAgent` (a DDQN subclass) uses the layout
  `ReplayBuffer -> NStepSelector -> TransitionBuffer`. Recency rows use
  double-Q. Long-term rows use SARSA with an expectile loss, as in the original.

## Current state

`src/components/buffer.py` holds one `ReplayBuffer`. It is a frame-storing ring
buffer. Each slot stores the newest frame of `obs`, the action, reward,
termination, truncation, and discount, plus a `first` flag marking episode
starts. `sample` draws windows of `stack_size - 1 + n_step + 1` slots. It stacks
frames for `obs` and `boot_obs` and computes the n-step return with
`n_step_return`. Batch size, n-step, and gamma are fixed at construction. The
state is a `NamedTuple` carried through `jax.lax.scan` and vmapped over seeds.

Agents call `build_buffer("uniform", ...)`, `add`, `sample`, and `can_sample`.
Tests also use `stored_transitions`.

## Original endpoint replay

Source: `coresets/src/buffers/CompositeBuffer.py`, `SubsampleBuffer.py`,
`replay/ingress/LagBuffer.py`, `algorithms/nn/SARSA.py`. The reference
experiment is `atari-50M-endpoint/truefinal/endpoint_medium.json`. It uses
`buffer_type: composite_coreset` with these settings:

| Setting | Value | Meaning |
| --- | --- | --- |
| `buffer_size` | 10,000 | recency capacity, 1-step transitions |
| `coreset_size` | 90,000 | long-term capacity |
| `coreset_nstep` | 10 | length of long-term transitions |
| `subsample` | 10 | keep every 10th long-term transition |
| `batch` / `recency_sample_number` | 32 / 28 | 28 recency rows, 4 long-term rows |
| `expectile_tau` | 0.7 | expectile of the long-term SARSA loss |

`endpoint_medium_noexpectile.json` differs only by omitting `expectile_tau`,
which gives plain SARSA on long-term rows.

Data flow:

1. Each timestep enters two lag buffers. One has lag 1 and feeds the recency
   buffer. The other has lag 10 and feeds a 10k "shadow" buffer.
2. When the shadow buffer is full, each transition it overwrites goes to
   `SubsampleReplayBuffer`. That buffer keeps transitions where
   `trans_id % 10 == 0`. Inside an episode, this yields a chain of 10-step
   transitions, where each one's bootstrap state is the next one's start state.
3. The long-term buffer is a 90k deque that stores copies of the stacked `x` and
   `n_x` observations, the 10-step return, the discount product, and the next
   action `ap`.

Sampling and loss:

- While the long-term buffer is empty, all 32 rows come from recency. After
  that, 28 rows come from recency and 4 from long-term.
- Recency rows use the double-Q target and `0.5 * delta**2`.
- Long-term rows use a SARSA target, `Q_target(n_x)[ap]`, and the expectile loss
  `weight * delta**2`. Here `delta = target - q[a]`, and `weight` is `tau` when
  `delta > 0` and `1 - tau` otherwise. This loss has no 0.5 factor.
- The batch loss is the mean over all 32 rows.

Known quirks, which the new design does not reproduce:

- `trans_id` is a global counter. Episode tails consume extra ids, so the
  subsample phase drifts from episode to episode.
- The shadow lag buffer is never flushed (see the TODO in
  `CompositeBuffer.flush`). Its transitions can span a truncated episode
  boundary.
- A life loss sets the step discount to 0 without ending the lag window. The
  chunk containing the life loss runs to 10 steps with the later rewards zeroed,
  so the first states of the new life never start a chunk.

## Unified contract

```python
class Pushed(NamedTuple, Generic[S, Out]):
    state: S
    out: Out            # the item that aged out, fixed shape every step
    out_valid: jax.Array  # bool scalar, whether `out` is real


class Buffer(Protocol[S, In, Out]):
    def init(self, observation_space) -> S: ...
    def push(self, state: S, item: In, valid: jax.Array) -> Pushed[S, Out]: ...
    def size(self, state: S) -> jax.Array: ...


class SampleableBuffer(Buffer[S, In, Out], Protocol):
    @property
    def batch_size(self) -> int: ...
    def sample(self, state: S, key: jax.Array) -> Batch: ...
    def can_sample(self, state: S) -> jax.Array: ...
```

Rules:

- Every buffer is a stateless Python object holding static configuration, with a
  `NamedTuple` state that works under `jit`, `scan`, and `vmap`.
- Batch size is fixed at construction. `sample` always returns `batch_size`
  rows.
- `out` always has the same shape, so it can flow through `scan`.
- `init` receives the environment's observation space. Downstream buffers store
  stacked observations of shape `observation_space.shape`.

### The `valid` flag

Under `jit` and `scan`, every buffer's `push` runs on every environment step
with fixed-shape inputs and outputs, but upstream buffers do not produce a real
item every step. `valid` says whether the incoming item is real. With
`valid=False`, `push` leaves the state unchanged and returns `out_valid=False`.
Buffers write with `jnp.where(valid, new, old)` on the target slot, not with
`lax.cond`, which vmap turns into a select anyway.

In the endpoint layout:

| Buffer | Receives `valid` from | True when |
| --- | --- | --- |
| `ReplayBuffer` | `ComposedBuffer.add` | always |
| `NStepSelector` | recency's `out_valid` | recency is full, so a timestep aged out |
| `TransitionBuffer` | selector's `out_valid` | the selector completed a chunk this step |

The transition buffer's own `out_valid` is true when a write overwrites a
transition. It is the last buffer in the layout, so that output is discarded.

### Item types

- `TimeStep` (unchanged): one environment step, `obs, action, reward,
  termination, truncation, discount`. What goes into `ReplayBuffer` has the
  stacked obs, the same as today's `add`. What comes out of it also has the
  stacked obs.
- `Transition`: one row of `Batch`, meaning a `Batch` without the leading axis.

`Batch` gains one field:

```python
class Batch(NamedTuple):
    obs: jax.Array
    action: jax.Array
    ret: jax.Array
    discount: jax.Array
    boot_obs: jax.Array
    boot_action: jax.Array  # action taken at boot_obs
    mask: jax.Array
```

`ReplayBuffer.sample` fills `boot_action` from `data.action` at the bootstrap
slot. DQN and DDQN do not read it, so XLA removes the gather and the sample cost
stays the same.

## Buffers

### ReplayBuffer (uniform, frame-storing)

`add`, `sample`, `can_sample`, `init`, and `stored_transitions` keep their
current code and behavior. New members:

- `batch_size` returns the constructor batch size.
- `size(state)` returns `state.size`.
- `push(state, step, valid)` reads the outgoing timestep, then applies `add`.
  `add` does not call `push`. Agents that use a single buffer keep calling
  `add`.

```python
def push(self, state, step, valid):
    out = self._outgoing(state)  # stacked TimeStep at slot (head + k - 1)
    out_valid = valid & (state.size == self._capacity)
    new_state = self.add(state, *step)
    # sketch: the implementation masks only the written slot
    state = jax.tree.map(lambda n, o: jnp.where(valid, n, o), new_state, state)
    return Pushed(state, out, out_valid)
```

The outgoing timestep is the one that leaves the sampleable range when the next
write happens. It is not the slot being overwritten. With stack size `k`, a full
buffer's valid starts begin at `oldest + k - 1`, because older slots cannot form
a full frame stack. Writing slot `head` moves that boundary forward by one. So
the outgoing timestep sits at `e = (head + k - 1) % capacity`, and its stacked
obs uses slots `head .. e` with their `first` flags, through `stack_frames`.

Consequences:

- The outgoing stream is exactly the `stored_transitions` sequence, in the order
  its entries age out. Tests use this as the oracle.
- The first `k - 1` timesteps of a run never leave through `push`. While the
  buffer is filling, they can be sampled with zero padding. The write that fills
  the buffer turns on the `k - 1` lookback skip, so all of them leave the
  sampleable range on the same step, and `push` emits only one timestep per
  step. This is 3 steps per run for Atari and none for vector observations.
- `add` is untouched, so agents that use only `add` keep today's speed. `push`
  adds one gather of `k` frames per step.

`n_step_return` is unchanged. A zero step discount (life loss) removes later
rewards and the bootstrap without cutting the window, which gives the same
target as cutting at the life loss. Known limitation: if a truncation follows a
life loss within `n_step` steps, the window is cut at the truncation and gets
`mask=False`, even though its target needs no bootstrap.

### NStepSelector

This turns a stream of stacked `TimeStep`s into chained multi-step
`Transition`s. It is not sampleable.

- In: `TimeStep` (stacked obs). Out: `Transition`.
- Config: `n_step: int`, `gamma: float`.
- State: pending chunk start `obs` and `action`, running `ret`, running `weight`
  (starts at 1), `horizon`, `active`, and `closed`. `closed` means the pending
  chunk ended on a zero discount, from a termination or a life loss.

Rule: a chunk is emitted when the step after its last reward arrives, and that
arriving step supplies the bootstrap. Each arriving step `j` goes through three
checks:

1. Emit if `active & horizon >= 1 & (horizon == n_step | closed | truncation_j)`,
   with `boot_obs = obs_j`, `boot_action = action_j`, `discount = weight`, and
   `mask = True`.
2. If `truncation_j`, reset to inactive. The truncating step's reward is
   dropped, because its successor observation belongs to the next episode.
3. Otherwise, start a new chunk at `j` if one was just emitted or none is
   active. Then accumulate `ret += weight * reward_j`,
   `weight *= gamma * discount_j`, and `horizon += 1`, and set
   `closed = termination_j | (discount_j == 0)`.

Properties:

- Chunks start at steps `0, N, 2N, ...` of every episode and every life. The
  phase restarts after each termination, life loss, and truncation. Inside a
  life, each chunk's `boot_obs` is the next chunk's `obs`.
- At most one chunk is emitted per step, so `out` has a fixed shape.
- A termination or life loss produces a shorter chunk with `discount = 0`. Its
  `boot_obs` is the next step's obs: the next episode's first obs after a
  termination, and the new life's first obs after a life loss. This is the same
  convention as `ReplayBuffer`.
- Truncation ends the chunk one step early, bootstrapping from the truncating
  step's own obs. This keeps the chunk valid. A chunk that would start at the
  truncating step is never emitted.
- The first chunk of a run starts at the first timestep recency emits. With
  frame stacking, this is run step `k - 1`, so the first episode's chunk grid is
  shifted by `k - 1` steps (3 for Atari). Every later episode is aligned. Fixing
  this would need an episode step counter carried with each emitted timestep,
  which this design accepts not to add.

The N-step rule needs only an accumulator, not a ring of timesteps. The
`Buffer` contract allows a selector that keeps a window of timesteps to be added
later.

### TransitionBuffer

A ring buffer of complete `Transition`s.

- In: `Transition`. Out: the overwritten `Transition`, with
  `out_valid = valid & (size == capacity)`.
- Config: `capacity`, `batch_size`.
- State: a `Batch` of arrays with leading axis `capacity`, plus `head` and
  `size`.
- `sample` draws `batch_size` rows uniformly with replacement from `[0, size)`.
- `can_sample` is `size > 0`, matching the original. Until the buffer holds
  `batch_size` transitions, samples repeat rows.

Memory: each Atari transition holds two 84x84x4 uint8 observations (56 KB), so
90k transitions take about 5.1 GB per run. Under vmap, each packed seed adds
another 5.1 GB. This limits packing on a GPU. Deduplicating the shared endpoints
would halve it. It is a later optimisation, not part of this design.

### ComposedBuffer

```python
class ComposedBuffer:
    def __init__(self, buffers: Sequence[Buffer]): ...
```

- State: `ComposedState(buffers: tuple[...])`, one state per buffer.
- `push` feeds `(out, out_valid)` from buffer `i` into buffer `i + 1`. The
  composed buffer's own `out` is the last buffer's `out`, so a composed buffer
  can itself be one of the buffers.
- `add(state, obs, action, reward, termination, truncation, discount)` packs a
  `TimeStep` and calls `push` with `valid=True`. It has the same signature as
  `ReplayBuffer.add`, so agents use either one the same way.
- `size` is the sum of the buffers' sizes.

Batch sizes come from the buffers. Non-sampleable buffers, such as the selector,
contribute no rows.

- The fallback is the first sampleable buffer. It must not be a
  `ComposedBuffer`. Its `batch_size` is the total `B`.
- Every other sampleable buffer `i` contributes its own `batch_size` `c_i`. The
  fallback's own share is `c_fb = B - sum(c_i)`.
- `batch_size` of the composed buffer is `B`, and its `can_sample` is the
  fallback's `can_sample`.

Construction checks: there is at least one sampleable buffer, the fallback is
not composed, and `c_fb > 0`.

Sampling, where `avail_i` is buffer `i`'s `can_sample`:

- The fallback always draws `B` rows. Its first `c_fb` rows are its own. The
  rest are substitutes, aligned with the other buffers' row ranges.
- Each other sampleable buffer `i` draws its `c_i` rows.
- `sample(state, key) -> Batch` returns the merged batch in buffer order. For
  buffer `i`'s rows, it picks the buffer's own rows when `avail_i` is true and
  the fallback's substitute rows otherwise.
- `sample_components(state, key) -> tuple[Batch, ...]` returns one batch per
  sampleable buffer. The fallback's batch has `B` rows. Its substitute rows have
  `mask &= ~avail_i`. Each other buffer's batch has `c_i` rows with
  `mask &= avail_i`. A loss that sums masked rows and divides by the mask count
  therefore uses exactly `B` rows. When every buffer is available, those are
  the mixture rows. Otherwise the fallback fills in, matching the original.

Cost: the fallback always draws and trains on `B - c_fb` extra rows in
per-component mode (4 rows for endpoint). Masked rows still go through the
network. A `lax.cond` would not avoid this, because vmap over seeds turns it
into a select.

## Endpoint agent

`src/agents/endpoint.py`, registered as `"endpoint"`.

```python
@dataclass(frozen=True, kw_only=True)
class EndpointConfig(DDQNConfig):
    """Endpoint replay's hyperparameters; defaults are for Atari experiments."""

    TOTAL_TIMESTEPS: int = 12_500_000
    LR: float = traced(6.25e-5)
    ADAM_EPS: float = traced(1.5e-4)
    BUFFER_SIZE: int = 10_000
    BATCH_SIZE: int = 32
    LEARNING_STARTS: int = traced(20_000)
    TRAIN_FREQUENCY: int = traced(4)
    TARGET_NETWORK_FREQUENCY: int = traced(8_000)
    EPSILON_END: float = traced(0.01)
    # decays over 250k steps after warmup; the original finished at step 250k
    # (230k steps of decay)
    EPSILON_DECAY_STEPS: int = traced(250_000)
    NETWORK_PRESET: str = "nature_cnn"
    REWARD_CLIP: bool = True
    LONG_TERM_SIZE: int = 90_000
    LONG_TERM_N_STEP: int = 10
    LONG_TERM_BATCH_SIZE: int = 4
    EXPECTILE_TAU: float = traced(0.7)
```

Inherited fields keep their meaning. `BUFFER` selects the recency buffer type
(only `"uniform"` exists), `BUFFER_SIZE` is the recency capacity, `N_STEP` is
the recency n-step, and `BATCH_SIZE` is the total batch `B`.
`EXPECTILE_TAU = 0.5` gives plain SARSA on long-term rows, which reproduces
`endpoint_medium_noexpectile.json`.

`EndpointAgent` overrides `__init__` and `_train_step` and keeps the whole loss
inline, in the same style as `DDQNAgent`. `DQNAgent` and `DDQNAgent` are not
changed. `update`, `act`, and `init` are inherited from `DQNAgent`, because
`ComposedBuffer` has the same `add` and `can_sample` signatures as
`ReplayBuffer`.

```python
class EndpointAgent(DDQNAgent):
    def __init__(self, config: EndpointConfig):
        self._config = config
        self._buffer = ComposedBuffer([
            build_buffer(config.BUFFER, capacity=config.BUFFER_SIZE,
                         batch_size=config.BATCH_SIZE, n_step=config.N_STEP,
                         gamma=config.GAMMA),
            NStepSelector(n_step=config.LONG_TERM_N_STEP, gamma=config.GAMMA),
            TransitionBuffer(capacity=config.LONG_TERM_SIZE,
                             batch_size=config.LONG_TERM_BATCH_SIZE),
        ])
        self._optimizer = optax.adam(config.LR, eps=config.ADAM_EPS)

    def _train_step(self, state: DQNState, key: jax.Array):
        config = self._config
        recent, long_term = self._buffer.sample_components(state.buffer_state, key)

        def loss_fn(q: QNetwork) -> jax.Array:
            # recency rows: double-Q target
            q_sa = jax.vmap(q)(recent.obs)
            q_a = jnp.take_along_axis(q_sa, recent.action[:, None], axis=-1).squeeze(-1)
            boot_action = jnp.argmax(jax.vmap(q)(recent.boot_obs), axis=-1)
            boot = jnp.take_along_axis(
                jax.vmap(state.target_q)(recent.boot_obs), boot_action[:, None], axis=-1
            ).squeeze(-1)
            target = jax.lax.stop_gradient(recent.ret + recent.discount * boot)
            recent_err = jnp.where(recent.mask, jnp.square(q_a - target), 0.0)

            # long-term rows: SARSA target from the stored boot action, expectile loss
            q_sa = jax.vmap(q)(long_term.obs)
            q_a = jnp.take_along_axis(q_sa, long_term.action[:, None], axis=-1).squeeze(-1)
            boot = jnp.take_along_axis(
                jax.vmap(state.target_q)(long_term.boot_obs),
                long_term.boot_action[:, None],
                axis=-1,
            ).squeeze(-1)
            target = jax.lax.stop_gradient(long_term.ret + long_term.discount * boot)
            delta = target - q_a
            weight = jnp.where(delta > 0, config.EXPECTILE_TAU, 1.0 - config.EXPECTILE_TAU)
            # 2x matches the original, whose recency rows carry a 0.5 factor this one lacks
            long_term_err = jnp.where(
                long_term.mask, 2.0 * weight * jnp.square(delta), 0.0
            )

            count = jnp.sum(recent.mask) + jnp.sum(long_term.mask)
            return (jnp.sum(recent_err) + jnp.sum(long_term_err)) / jnp.maximum(count, 1)

        grads = eqx.filter_grad(loss_fn)(state.q)
        updates, opt_state = self._optimizer.update(
            grads, state.opt_state, eqx.filter(state.q, eqx.is_array)
        )
        return state._replace(
            q=eqx.apply_updates(state.q, updates), opt_state=opt_state
        )
```

A timestep enters the long-term buffer once it is older than `BUFFER_SIZE`
steps, plus up to `N` steps for its chunk to finish. This is the same age as in
the original, where it passed through a 10-step lag and a 10k shadow buffer.
With `LEARNING_STARTS >= BUFFER_SIZE`, as in the reference config, the
long-term buffer already has data before the first update. The fallback then
never activates.

### Mapping the reference config

The `EndpointConfig` defaults carry these values. The environment uses
`EPRAtariConfig`.

| Original | rl-jax |
| --- | --- |
| `total_steps: 12.5M` | `TOTAL_TIMESTEPS = 12_500_000` |
| `frame_skip 4`, `repeat_action_probability 0.25`, `episode_cutoff 27000`, life loss sets gamma 0 | `EPRAtariConfig` (108k frames = 27k steps) |
| `reward_clip: 1` (set by the Atari problem, `np.clip` to [-1, 1]) | `REWARD_CLIP = True` (sign). The two agree on integer rewards. |
| `warmup_steps 20000` | `LEARNING_STARTS = 20_000` |
| epsilon 1.0 -> 0.01, ends at step 250k, decay starts after warmup | `EPSILON_END = 0.01`, `EPSILON_DECAY_STEPS = 250_000` (ends at step 270k) |
| `update_freq 4` | `TRAIN_FREQUENCY = 4` |
| `target_refresh 2000` (counted in updates) | `TARGET_NETWORK_FREQUENCY = 8_000` (counted in steps) |
| adam `6.25e-5`, eps `1.5e-4` | `LR = 6.25e-5`, `ADAM_EPS = 1.5e-4` |
| `AtariNet`, hidden 512 | `NETWORK_PRESET = "nature_cnn"` |
| `buffer_size 10000`, `n_step 1`, `batch 32` | `BUFFER_SIZE = 10_000`, `N_STEP = 1`, `BATCH_SIZE = 32` |
| `coreset_size 90000`, `coreset_nstep 10`, `recency_sample_number 28` | `LONG_TERM_SIZE = 90_000`, `LONG_TERM_N_STEP = 10`, `LONG_TERM_BATCH_SIZE = 4` |
| `expectile_tau 0.7` | `EXPECTILE_TAU = 0.7` |

### Differences from the original

| | Original | New design |
| --- | --- | --- |
| Chunk phase | global `trans_id` counter, drifts after each episode tail | restarts at every episode and life |
| First episode of a run | complete, aligned at run step 0 | first `k - 1` steps lost, grid shifted by `k - 1` |
| Truncation | shadow chunks span the episode boundary | chunk ends one step early and stays valid |
| Termination | tail transitions subsampled by `trans_id` | exactly one shorter chunk |
| Life loss | chunk runs on with zeroed rewards, new life's first states skipped | chunk closes, next chunk starts at the new life |
| Loss scale | recency `0.5 * delta**2`, long-term `weight * delta**2` | recency `delta**2`, long-term `2 * weight * delta**2` |
| Epsilon | reaches 0.01 at step 250k | reaches 0.01 at step 270k |
| Life-loss step | frame skip stops at the life-loss frame (1 to 4 frames) | always 4 frames inside ALE |

The loss scale keeps the original ratio between recency and long-term rows. The
overall factor of 2 is largely absorbed by Adam, apart from the interaction with
`eps`.

The life-loss step difference is an environment difference and is out of scope
for this design. In the original, rewards from frames after a life loss go to
the next step, and a life lost on frame 0 or 1 repeats the previous frame in the
stack. In rl-jax, ALE runs all 4 frames, so those rewards are credited to the
action before the life loss. The original's episode cutoff also counts agent
steps, while rl-jax counts frames.

## Module layout

`src/components/buffer.py` becomes a package. `components/__init__.py`
re-exports the public names, so agent imports stay the same.

```
src/components/buffers/
    __init__.py      # build_buffer, public exports
    contract.py      # Buffer, SampleableBuffer, Pushed, TimeStep, Batch
    uniform.py       # ReplayBuffer, sample_windows, stack_frames, n_step_return
    selector.py      # NStepSelector
    transition.py    # TransitionBuffer
    composed.py      # ComposedBuffer
src/agents/endpoint.py
```

`build_buffer` keeps resolving single-buffer names. The endpoint agent builds
its composed layout directly.

## Testing

- `ReplayBuffer`: the existing tests and microbenchmarks pass unchanged.
  Property test: pushing a stream yields, in order, the `stored_transitions` a
  larger buffer would hold for the evicted range. `boot_action` matches the
  reference sampler.
- `NStepSelector`: a Python reference that splits episodes and lives into
  chunks. Use Hypothesis over reward, termination, truncation, and discount
  streams. Check chaining, the phase restart at episodes and life losses,
  terminal and life-loss tails, truncation cuts, and at most one emission per
  step. The return and discount agree with `n_step_return` on unbroken windows.
- `TransitionBuffer`: order of writes and evictions, the `valid=False` no-op,
  uniform coverage of `[0, size)`.
- `ComposedBuffer`: chaining, per-buffer row counts, merged versus per-component
  consistency, fallback substitution and masks when a buffer is unavailable,
  nesting, and construction checks.
- `EndpointAgent`: runs under `jit` and `vmap`. The long-term rows change the
  update compared with DDQN on the same stream. `EXPECTILE_TAU = 0.5` matches a
  plain SARSA loss on long-term rows.
- Microbenchmark: `push` on `ReplayBuffer` and a composed endpoint step, next to
  the existing add and sample benchmarks.

## Implementation sequence

Each step is its own commit, or a few commits, and each one leaves the tests
passing.

1. Move `buffer.py` into the `buffers/` package with no behavior change.
2. Add the `Buffer` and `SampleableBuffer` protocols, plus `batch_size` and
   `size` on `ReplayBuffer`.
3. Add `boot_action` to `Batch` and fill it in `ReplayBuffer`.
4. Add `ReplayBuffer.push` with the outgoing timestep, then its tests.
5. Add `NStepSelector`, then its tests.
6. Add `TransitionBuffer`, then its tests.
7. Add `ComposedBuffer` with merged sampling, then per-component sampling and
   fallback, each with tests.
8. Add `EndpointAgent` and its registry entry, then tests.
9. Add an endpoint experiment config that mirrors the reference run.
