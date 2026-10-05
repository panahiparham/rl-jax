# MinAtar environments and the original DQN reproduction

Design for adding four MinAtar games as one environment, a network, a DQN
configuration with the hyperparameters of reg-duel-q's MinAtar baseline, and an
experiment that runs that DQN on all four games.

Sources: Young and Tian, "MinAtar" (arXiv:1903.03176), and
`kenjyoung/MinAtar` (`examples/dqn.py`, `minatar/environment.py`). A second
reference, `brett-daley/reg-duel-q`, is compared in its own section.

## Summary

- New environment `minatar`, one config class, `GAME` selects the game. Built
  on gymnax's MinAtar games, with every gymnax setting exposed as a config
  field, as `AtariConfig` does for ale-py.
- Four games: Asterix, Breakout, Freeway, Space Invaders. Seaquest is left out
  because gymnax cannot run it under `jit`.
- Environment defaults match the original MinAtar `Environment`: sticky
  actions with probability 0.1, difficulty ramping on, and no time limit except
  Freeway's own 2500 steps. The action set defaults to the minimal set, as in
  gymnax and reg-duel-q; `USE_MINIMAL_ACTION_SET` selects all 6 instead.
- gymnax has no sticky actions. The adapter adds them.
- New network preset `minatar_cnn`: conv 16x3x3, dense 128, linear head.
- `MinAtarDQNConfig`, next to `DQNConfig`, sets the hyperparameters of
  reg-duel-q's DQN: Adam, squared error, update every 4 steps. `DQNAgent` does
  not change except for the network preset.
- New experiment `minatar_dqn`: 4 games, 30 seeds, 10M steps, no tuning.

## Original setup

Paper, section 3 (DQN):

- One conv layer with 16 3x3 filters, stride 1, then a dense layer of 128
  units. Both are one quarter of Mnih et al. 2015.
- Replay size, target update frequency, epsilon annealing time and replay
  fill time are one tenth of Mnih et al. Everything else matches Mnih et al.
- Trained on every frame, no frame skipping, 5M frames, 30 seeds per game.
- Step size swept over powers of 2. Figure 3 uses, per game, the largest step
  size whose confidence interval overlaps the best final return in Figure 2.
- Figure 2 is the average return over the final 100 episodes. Figure 3 is the
  100-episode moving average of return, with standard error over 30 runs.
- Sticky actions with probability 0.1 (section 2). Difficulty ramping is on by
  default in `Environment`. Each game is one life.

`examples/dqn.py`, against this design:

| Setting | Original value | This design |
| --- | --- | --- |
| Frames | 5,000,000 | 10,000,000 |
| Batch size | 32 | same |
| Replay size | 100,000 | same |
| Replay start | 5,000 (uniform random actions before this) | 1,000 |
| Training frequency | every frame | every 4 steps |
| Target update | every 1,000 gradient updates | every 1,000 steps |
| Discount | 0.99 | same |
| Epsilon | 1.0, linear to 0.1 over 100,000 frames after replay start | 1.0, linear to 0.01 over 250,000 steps after replay start |
| Optimizer | RMSprop, centered, lr 0.00025, alpha 0.95, eps 0.01 | Adam, lr 0.00025, eps 3.125e-4 |
| Loss | Huber (`smooth_l1_loss`, beta 1) | squared error (existing) |
| Reward clipping | none | same |
| Actions | `env.num_actions()`, the full set of 6 | minimal set |
| Network init | PyTorch default | same |
| Environment | `sticky_action_prob=0.1`, `difficulty_ramping=True` | same |

Terminal next states bootstrap with zero. `Environment.last_action` starts at 0,
is set after the sticky substitution, and is never reset between episodes.

## Current state

- `ENVIRONMENTS` in `src/environments/__init__.py` maps a name to an
  `EnvSpec(config_cls, build, vmappable)`. Classic control uses gymnax through
  `GymnaxEnv.make` and `AutoresetImmediate`. Atari has a flat config
  (`AtariConfig`) with presets as subclasses.
- `GymnaxEnv.step` reports `truncated = state.time >= max_steps` and
  `terminated = done & ~truncated`.
- `NETWORK_PRESET` in `DQNAgent._build_q` picks the network. Presets are
  `mlp`, `mlp_ln`, `nature_cnn`, `nature_cnn_ln`.
- `DQNConfig` fields marked `traced` can vary inside one vmap. `linear_epsilon`
  holds epsilon at `EPSILON_START` until `LEARNING_STARTS`, then decays over
  `EPSILON_DECAY_STEPS`. This already matches the original schedule.
- The replay buffer stores frames in `observation_space.dtype`. A space
  without `frame_channels` is stored as one frame per step, with no stacking.
- Experiments are `experiments/<name>/{config.py,run.py,analysis.ipynb}` with
  one `Experiment` of `Component`s.

### What gymnax provides

Installed gymnax (checked in `.venv`): `Asterix`, `Breakout`, `Freeway`,
`SpaceInvaders` work through `gymnax.make`. Each takes
`use_minimal_action_set` in its constructor and has an `EnvParams` dataclass.

| Game | Channels | Params besides `max_steps_in_episode` |
| --- | --- | --- |
| Breakout | 4 | none |
| Asterix | 4 | `ramping`, `ramp_interval`, `init_spawn_speed`, `init_move_interval`, `shot_cool_down` |
| Freeway | 7 | `player_speed` |
| Space Invaders | 6 | `shot_cool_down`, `enemy_move_interval`, `enemy_shot_interval` |

Space Invaders has no `ramping` param: `reset_env` sets `ramping=True` in the
state, so its difficulty always ramps. Breakout and Freeway do not ramp, as in
the original.

Observations are built as a `bool` grid and cast to `float32`, so every cell
is 0 or 1. `observation_space` is `Box(0, 1, (10, 10, C))` with `float32`.

Seaquest is excluded. `gymnax.make("Seaquest-MinAtar")` raises `ValueError`
because it is not registered, and the module cannot be traced: `get_obs`
slices with a traced oxygen value, loops with `range(state.diver_count)` over
traced counts, and indexes the empty lists `fish`, `sub` and `diver`.
`reset_env` under `jit` fails with `IndexError` on the oxygen slice. Supporting
it needs a rewrite of the game, which is out of scope here.

Gaps relative to the original:

1. **No sticky actions.** There is no mention of them in the gymnax MinAtar
   code. The original applies them in `Environment.act`, not in a game.
2. **Time limit.** Defaults are `max_steps_in_episode = 1000` (Freeway 2500).
   The original has no limit except Freeway's. A 1000 step cap changes returns
   for good agents in Breakout, Space Invaders and Asterix.
3. **Action set.** gymnax defaults to the minimal set. The original DQN uses
   all 6 actions. This design follows gymnax and reg-duel-q.
4. **Observation.** `float32` of shape `(10, 10, C)`, which is 4 bytes per
   cell in the replay buffer.

## Design

### 1. Environment

New `src/environments/minatar.py`, registered as `"minatar"`, vmappable.

```python
@dataclass(frozen=True)
class MinAtarConfig:
    GAME: str = "breakout"  # asterix, breakout, freeway, space_invaders
    STICKY_ACTION_PROB: float = 0.1
    USE_MINIMAL_ACTION_SET: bool = True
    EPISODE_CUTOFF: int | None = None  # None: the game's own limit, if any
    # gymnax EnvParams. None keeps the game's default; setting a field the
    # selected game does not have raises at build time.
    RAMPING: bool | None = None
    RAMP_INTERVAL: int | None = None
    INIT_SPAWN_SPEED: int | None = None
    INIT_MOVE_INTERVAL: int | None = None
    SHOT_COOL_DOWN: int | None = None
    ENEMY_MOVE_INTERVAL: int | None = None
    ENEMY_SHOT_INTERVAL: int | None = None
    PLAYER_SPEED: int | None = None
```

Field to game mapping follows the table above. Raising on a field the game
lacks keeps a sweep over `ENV_HYPERS.SHOT_COOL_DOWN` from silently doing
nothing on Breakout. Each `EnvParams` field other than
`max_steps_in_episode` maps to its `UPPER_CASE` config name, so the mapping is
one comprehension, not a table. An unknown `GAME`
raises with the list of supported games.

`build` creates the game, builds `EnvParams` from the non-`None` fields, and
returns `AutoresetImmediate(MinAtarEnv(...), params)`.

`MinAtarEnv` is the adapter `AutoresetImmediate` wraps. It has the same
`reset`/`step`/space interface as `GymnaxEnv` and adds:

- **State.** `MinAtarState(game: <gymnax state>, last_action: int32)`.
- **Sticky actions.** `a = where(uniform(key) < p, last_action, action)`,
  then `last_action = a`, as in the original. The step key is split into the
  sticky draw and the game's key. `last_action` is 0 after a reset. The
  original never resets it; the difference is at most one action per episode.
- **Time limit.** gymnax ends an episode when `state.time >=
  max_steps_in_episode`, counting steps from 0 at reset. The game's
  `max_steps_in_episode` is set to `min(EPISODE_CUTOFF, game limit)`, where the
  game limit is 2500 for Freeway and `2**31 - 1` (`int32` max) for the rest.
  `state.time` is `int32`, and a 10M-step run cannot reach that limit. Freeway's
  2500 steps are part of the game and the original reports them as terminal,
  so they are `terminated`. An `EPISODE_CUTOFF` below the game limit is a time
  limit, so reaching it is `truncated`. A cutoff at or above Freeway's 2500
  leaves Freeway's end `terminated`. `terminated = done & ~truncated`, with
  `truncated = time >= EPISODE_CUTOFF` only when the cutoff is below the game
  limit. Whether it is, is decided in `build`.
- **Observation.** Converted to `bool`, with `observation_space` reporting
  `dtype=bool` and shape `(10, 10, C)`. At 100k transitions this is 40 to
  70 MB per seed instead of 160 to 280 MB, which matters with 30 seeds in one
  vmap. The network casts to float, as `NatureCNN` already casts `uint8`.
- **Spaces.** `Discrete(num_actions)`. The minimal set has 5 actions in
  Asterix, 3 in Breakout, 3 in Freeway and 4 in Space Invaders. The full set
  has 6.

`GymnaxEnv` is not reused: its `make` fixes the game's default params and its
truncation rule does not know about game-owned limits. The adapter is written
for MinAtar only, with no shared base.

### 2. Network

`MinAtarCNN` in `src/components/networks.py`, preset `"minatar_cnn"`:

```
x (H, W, C) bool -> float32 -> (C, H, W)
conv 16 x 3x3, stride 1, valid -> relu      # (16, 8, 8)
flatten (1024) -> dense 128 -> relu
dense num_actions
```

Built with the existing `_conv` and `_linear` helpers. Their
`U(+/- 1/sqrt(fan_in))` init is PyTorch's default init for `Conv2d` and
`Linear`, which is what `dqn.py` relies on. The flattened size comes from a
helper, as `_nature_flat_dim` does, not a constant.

Parameter count for Breakout (4 channels, 3 actions): 592 + 131,200 + 387 =
132,179.

`DQNAgent._build_q` gets one more branch. `DDQNAgent` inherits it.
`Agent0Agent` has its own copy of the preset chain and is not changed.

### 3. DQN configuration

`DQNAgent` is unchanged apart from the preset branch. `MinAtarDQNConfig(DQNConfig)`
sits next to `DQNConfig` in `agents/dqn.py` and only overrides defaults, as
`EndpointConfig` does. It lists every setting of the table, including those
equal to `DQNConfig`'s defaults, so the reproduction does not change if those
defaults do:

```python
@dataclass(frozen=True, kw_only=True)
class MinAtarDQNConfig(DQNConfig):
    """DQN hyperparameters of reg-duel-q's MinAtar baseline."""

    LR: float = traced(0.00025)
    ADAM_EPS: float = traced(3.125e-4)
    BUFFER_SIZE: int = 100_000
    BATCH_SIZE: int = 32
    TOTAL_TIMESTEPS: int = 10_000_000
    LEARNING_STARTS: int = traced(1_000)
    TRAIN_FREQUENCY: int = traced(4)
    TARGET_NETWORK_FREQUENCY: int = traced(1_000)
    GAMMA: float = traced(0.99)
    EPSILON_START: float = traced(1.0)
    EPSILON_END: float = traced(0.01)
    EPSILON_DECAY_STEPS: int = traced(250_000)
    NETWORK_PRESET: str = "minatar_cnn"
    REWARD_CLIP: bool = False
```

`LR` and `ADAM_EPS` are reg-duel-q's Adam settings, with a fixed step size
and no sweep. The loss is the existing squared error. It is used with
`AGENT="dqn"`. No new registry entry is needed,
because the agent class is unchanged. `ExperimentConfig.AGENT_HYPERS` accepts
it as a `DQNConfig`, and `split_traced` resets its traced fields to its own
defaults, so its runs shard like any `DQNConfig`.

### 4. Experiment

`experiments/minatar_dqn/{config.py,run.py,analysis.ipynb}`, modeled on
`atari_10m`:

```python
Component(
    name="dqn_minatar",
    config=ExperimentConfig(
        AGENT="dqn",
        ENV="minatar",
        AGENT_HYPERS=MinAtarDQNConfig(),
        ENV_HYPERS=MinAtarConfig(),
    ),
    sweep={"ENV_HYPERS.GAME": [
        "asterix", "breakout", "freeway", "space_invaders",
    ]},
    seeds=list(range(30)),
)
```

- 10M steps is 10M frames in the papers' terms, because there is no frame
  skip.
- 30 seeds, as in both references. `GAME` is a static field, so each game is its own
  shard group of up to 30 seeds in one vmap. `shard_size` and Slurm resources
  are set after a timing run on one game. Metrics are `[30, 10M]` reward and
  done per game, about 1.5 GB of `float32` reward and `bool` done.
- No step-size sweep. All games use the one step size above.
- Analysis: per game, return against steps with the existing `return_curve`,
  as in the other experiments, with mean and interval over 30 seeds from
  `analysis.stats.mean_ci`.
- Reference values, two JSON files in the format of
  `atari_10m/dopamine_dqn_reference.json`, each with a `_source` block. The
  notebook overlays both on the learning curves.
  - `reg_duel_q_dqn_reference.json`: per game, `steps` (0 to 10M, 11 points),
    and the `mean_return` and `stderr` of the DQN evaluation score over 30
    seeds, computed from `rlc_results.zip`. The hyperparameters match this
    design, but the score is evaluation at epsilon 0.01 and ours is training
    return.
  - `minatar_paper_dqn_reference.json`: per game, `steps` (0 to 5M, every 0.1M)
    and `mean_return` of the paper's DQN training return. Neither MinAtar repo
    nor the paper releases numbers, only the plot (`img/learning_curves.gif`,
    the paper's Figure 3), so the curve is digitized from its pixels with an
    error of about half a percent of the axis height. It uses the original
    setup (RMSprop, Huber loss, all 6 actions, best step size per game).

  The comparison with either is of shape and range, not a numeric match.

### 5. Further experiments

Both copy `minatar_dqn`: the four games, 30 seeds each, one vmap of all seeds
per game, 10M steps and the `MinAtarDQNConfig` hyperparameters.

- `minatar_deterministic`: the same component, run with
  `--xla_gpu_deterministic_ops=true`, so same-seed runs reproduce exactly. Its
  notebook overlays both references.
- `minatar_endpoint`: seven components per game, all learning with the
  `minatar_dqn` hyperparameters (taken from `MinAtarDQNConfig`, not repeated)
  and differing only in memory. It also runs on deterministic GPU kernels.

  | Component | Agent | Memory |
  | --- | --- | --- |
  | `ddqn_minatar` | DDQN | 100k buffer |
  | `ddqn_medium_minatar` | DDQN | 10k buffer |
  | `ddqn_small_minatar` | DDQN | 2k buffer |
  | `endpoint_minatar` | endpoint | 1k recency + 9k long-term |
  | `endpoint_small_minatar` | endpoint | 1k recency + 1k long-term |
  | `unanchored_minatar` | unanchored | 1k recency + 9k long-term |
  | `unanchored_small_minatar` | unanchored | 1k recency + 1k long-term |

  Endpoint and unanchored use the settings of `pinball_endpoint` and
  `atari_50m_endpoint`: 4 of every 32 rows come from the long-term memory,
  endpoint stores chained 10-step transitions trained with an expectile of 0.7,
  and unanchored stores every 10th 1-step transition. The notebook follows
  `atari_50m_endpoint`: one grid of learning curves per memory budget with a
  panel per game, laid out as in `minatar_deterministic`, and one figure of two
  bar plots of min-max normalized average return, one per budget.

The replay agents had not run on bool observations. `stack_frames` masked
frames with an int zero, which turned them into int32 and broke the scan carry
of the composed buffer, so it now keeps the frame's dtype.

## Comparison with reg-duel-q

`brett-daley/reg-duel-q` (Daley, Nagarajan, White, Machado, RLC 2025) runs DQN,
dueling DQN and regularized dueling Q-learning on five MinAtar games with 30
seeds each. Its baseline is `pfrl.agents.DQN` in `scripts/train_dqn_minatar.py`.
Its environments are `gymnasium.make("MinAtar/<Game>-v1")` from the `minatar`
package (`minatar/gym.py`), which passes no extra arguments to `Environment`.
So sticky actions (0.1) and difficulty ramping stay at the original defaults,
and there is no time limit besides Freeway's. `-v1` selects the minimal action
set; `-v0` selects all 6 actions.

| Setting | Young and Tian (`dqn.py`) | reg-duel-q DQN | This design |
| --- | --- | --- | --- |
| Steps | 5M | 10M | 10M |
| Batch size | 32 | 32 | 32 |
| Replay size | 100,000 | 100,000 | 100,000 |
| Replay start | 5,000 | 1,000 | 1,000 |
| Update frequency | every step | every 4 steps | every 4 steps |
| Target update | 1,000 gradient updates | 250 gradient updates (1,000 steps) | 1,000 steps |
| Epsilon | 1.0 to 0.1 over 100k, after replay start | 1.0 to 0.01 over 250k, from step 0 | 1.0 to 0.01 over 250k, after replay start |
| Optimizer | centered RMSprop, lr 0.00025, alpha 0.95, eps 0.01 | Adam, lr 0.00025, eps 3.125e-4 | Adam, lr 0.00025, eps 3.125e-4 |
| Loss | Huber | `mse / 2`, mean | `mse`, mean |
| Reward clipping | none | none (`clip_delta=False`) | none |
| Action set | all 6 | minimal (`-v1`) | minimal (configurable) |
| Network | conv 16, dense 128 | same, PyTorch init | same |
| Sticky actions, ramping | 0.1, on | 0.1, on | 0.1, on |
| Reported score | training returns, moving average over 100 episodes | 1,000 evaluation episodes, epsilon 0.01, every 1M steps | training returns |
| Algorithm | DQN | DQN | DQN |

Notes from reading the code:

- The Adam settings are the "alternative values" the script cites from
  arXiv:2011.14826. They map onto existing `DQNConfig` fields: `LR` and
  `ADAM_EPS`. No optimizer or loss change is needed to run them. This design
  uses them.
- Their loss is half the mean squared error, so gradients are half of ours.
  With Adam this changes the effective step only through `eps`.
- `LinearDecayEpsilonGreedy` decays from step 0. `linear_epsilon` decays from
  `LEARNING_STARTS`, so this design decays 1,000 steps later.
- Evaluation uses a separate environment with its own seed, so it never
  shares a stream with training.
- The repo ships `rlc_results.zip`: `scores.txt` per run, 11 evaluation points
  (steps 0 to 10M) for DQN, dueling DQN and regularized dueling Q-learning,
  30 seeds per game (22 for regularized dueling on Freeway).

Final evaluation score of DQN at 10M steps, mean and standard error over 30
seeds, computed from `rlc_results.zip`:

| Game | Mean | Standard error |
| --- | --- | --- |
| Asterix | 36.95 | 0.96 |
| Breakout | 44.15 | 1.08 |
| Freeway | 60.50 | 0.14 |
| Space Invaders | 147.79 | 3.30 |
| Seaquest (not supported here) | 72.69 | 2.33 |

These are far above the paper's 5M-step training returns (for example Breakout
about 10 in Figure 3). The setups differ in steps, epsilon at the end of
training (0.01 evaluation against 0.1 training), update frequency, optimizer
and action set, so the two do not bound each other.

## Differences from the references

From reg-duel-q's DQN:

- Training returns, not evaluation episodes at epsilon 0.01.
- Epsilon decays from step 1,000, not step 0.
- The loss is `mse`, not `mse / 2`.
- Target updates count environment steps, every 1,000. reg-duel-q counts 250
  gradient updates, which is the same interval at update frequency 4.
- The games are gymnax's JAX ports, so random streams differ. No step-for-step
  match is possible.
- `last_action` resets each episode (see Environment).
- Freeway episodes last 2500 steps. The original decrements a 2500 timer and
  ends when it drops below 0, after 2501 steps.
- No Seaquest.

From Young and Tian's `dqn.py`: the differences are the "This design" column
of the table in Original setup, plus the points above. Batches here also
sample with replacement, where `dqn.py` uses `random.sample`.

## Verification

- Environment: observation shape, dtype and action count per game;
  `STICKY_ACTION_PROB=0` equals the raw gymnax dynamics; `=1` means every
  step executes action 0; Freeway ends as `terminated` at 2500, and an
  `EPISODE_CUTOFF` of 10 gives `truncated` at step 10; each param field changes
  behavior (for example `PLAYER_SPEED`); a field the game lacks raises; an
  unknown game raises; all four games run under `jit` and `vmap`.
- Network: output shape, parameter count above, init bounds, input layout.
- DQN: `MinAtarDQNConfig` fields equal the table above; the agent builds and
  steps with `minatar_cnn` on a MinAtar environment.
- Experiment: a short smoke run of every game through `process_shard`.

## Commit plan

Each commit is independently valid and about 20 changed lines or fewer, with
the exceptions noted. Additive features get a follow-on test commit.

1. `docs(design)`: this document.
2. `feat(environments)`: `MinAtarEnv` with `bool` observations.
3. `feat(environments)`: step `MinAtarEnv` on game termination.
4. `feat(environments)`: `MinAtarConfig` and `build`.
5. `feat(environments)`: apply each game's own time limit.
6. `feat(environments)`: register the `minatar` environment.
7. `test(environments)`: spaces and action sets.
8. `test(environments)`: termination and `jit`.
9. `feat(environments)`: sticky actions.
10. `test(environments)`: sticky actions.
11. `feat(environments)`: `EPISODE_CUTOFF`.
12. `test(environments)`: `EPISODE_CUTOFF`.
13. `feat(environments)`: gymnax params in `MinAtarConfig`.
14. `feat(environments)`: reject params the selected game lacks.
15. `test(environments)`: param fields.
16. `feat(components)`: `MinAtarCNN` network.
17. `test(components)`: `MinAtarCNN`.
18. `feat(agents)`: the `minatar_cnn` network preset.
19. `test(agents)`: DQN with the `minatar_cnn` preset.
20. `feat(agents)`: `MinAtarDQNConfig`.
21. `test(agents)`: `MinAtarDQNConfig` values.
22. `test(experiments)`: smoke run of DQN on every MinAtar game.
23. `feat(experiments)`: `minatar_dqn` config.
24. `feat(experiments)`: `minatar_dqn` entry point.
25. `feat(experiments)`: reg-duel-q DQN reference results (data file).
26. `feat(experiments)`: MinAtar paper DQN reference results (data file).
27. `docs(experiments)`: `minatar_dqn` analysis notebook.
28. `feat(experiments)`: `minatar_deterministic` config.
29. `feat(experiments)`: `minatar_deterministic` entry point.
30. `feat(experiments)`: reference results for `minatar_deterministic`.
31. `docs(experiments)`: `minatar_deterministic` analysis notebook.
32. `feat(experiments)`: `minatar_endpoint` config with the DDQN components.
33. `feat(experiments)`: endpoint components.
34. `feat(experiments)`: unanchored components.
35. `fix(components)`: keep the dtype of stacked frames.
36. `test(agents)`: train the replay agents on MinAtar.
37. `feat(experiments)`: `minatar_endpoint` entry point.
38. `docs(experiments)`: `minatar_endpoint` analysis notebook.

Running the 10M-step experiment is a separate cluster job.

## Decisions

- Hyperparameters are reg-duel-q's DQN: 10M steps, batch 32, replay 100,000,
  warmup 1,000 steps, update every 4 steps, target update every 1,000 steps,
  epsilon 1.0 to 0.01 over 250k steps after warmup, Adam with lr 0.00025 and
  eps 3.125e-4, squared error, minimal action set.
- Score is training return, as in the other environments. No evaluation loop.
- Fixed step size, no sweep.
- `EPISODE_CUTOFF` defaults to `None`, matching both references. gymnax's 1000
  step default would cap good Breakout and Space Invaders runs.
- The config is `MinAtarDQNConfig` in `agents/dqn.py`, used with the existing
  `dqn` agent, plus the `minatar_cnn` preset in `components/networks.py`.
- `USE_MINIMAL_ACTION_SET` is a config field defaulting to `True`.
- No parity test against the original Python MinAtar for now. The gymnax games
  are used as shipped, so a port that drifts from the original goes
  undetected. This also removes the `minatar` dev dependency and the need for
  a `slow` pytest marker.
- The reference results come from `rlc_results.zip` as a JSON file.
- Seaquest is deferred. It needs an in-repo port, or a fixed gymnax.

## Open questions

None.
