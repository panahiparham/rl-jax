from pathlib import Path

from experiment.design import Component, Experiment, SlurmResources

from agents.agent0 import Agent0Config
from agents.agent1 import Agent1Config
from environments.atari import AtariConfig
from main import ExperimentConfig

_GAMES = [
    "asterix",
    "venture",
    "atlantis",
    "tetris",
    "frogger",
    "gopher",
    "assault",
    "demon_attack",
    "video_pinball",
    "star_gunner",
    "alien",
    "amidar",
    "asteroids",
    "bank_heist",
    "berzerk",
    "bowling",
    "boxing",
    "chopper_command",
    "crazy_climber",
    "enduro",
    "fishing_derby",
    "freeway",
    "frostbite",
    "gravitar",
    "hero",
    "ice_hockey",
    "jamesbond",
    "kangaroo",
    "krull",
    "kung_fu_master",
    "montezuma_revenge",
    "pitfall",
    "private_eye",
    "riverraid",
    "road_runner",
    "robotank",
    "skiing",
    "solaris",
    "tennis",
    "time_pilot",
    "tutankham",
    "up_n_down",
    "wizard_of_wor",
    "yars_revenge",
    "zaxxon",
]

# Start from the DQN hypers; Agent0 and Agent1 adapt them below.
_DQN_HYPERS = {
    "TOTAL_TIMESTEPS": 50_000_000,      # 200M frames at FRAMESKIP=4
    "LR": 6.25e-05,
    "ADAM_EPS": 1.5e-4,
    "BUFFER_SIZE": 1_000_000,
    "BATCH_SIZE": 32,
    "LEARNING_STARTS": 20_000,
    "TRAIN_FREQUENCY": 4,
    "TARGET_NETWORK_FREQUENCY": 8_000,
    "GAMMA": 0.99,
    "EPSILON_START": 1.0,
    "EPSILON_END": 0.01,
    "EPSILON_DECAY_STEPS": 250_000,
    "NETWORK_PRESET": "nature_cnn",
}

# Agent0 and Agent1 have no target network and use the LN variant of the
# Nature CNN.
_AGENT_HYPERS = {
    **{k: v for k, v in _DQN_HYPERS.items() if k != "TARGET_NETWORK_FREQUENCY"},
    "NETWORK_PRESET": "nature_cnn_ln",
}

_AGENTS = {"agent0": Agent0Config, "agent1": Agent1Config}

EXPERIMENT = Experiment(
    name="benchmarking_atari_200m",
    results_dir=Path(__file__).resolve().parent / "results",
    components=[
        Component(
            name=f"{agent}_atari",
            config=ExperimentConfig(
                AGENT=agent,
                ENV="atari",
                AGENT_HYPERS=config_cls(**_AGENT_HYPERS, REWARD_CLIP=True),
                ENV_HYPERS=AtariConfig(),
            ),
            sweep={"ENV_HYPERS.GAME": _GAMES},
            seeds=[0],
            shard_size=1,
            parallel_shards=5,
        )
        for agent, config_cls in _AGENTS.items()
    ],
    # Five packed runs (parallel_shards=5) share one GPU through CUDA MPS,
    # with a CPU each.
    # atari_200m_deterministic's packed workers took about 20h.
    slurm=SlurmResources(
        time="23:59:00", gpus=1, mps=True, cpus_per_task=5, account="aip-whitem"
    ),
)
