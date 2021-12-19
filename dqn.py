from argparse import ArgumentParser
import json
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras import layers
from tensorflow.keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

import gym
import gym_roguelike


ENV_NAME = "RogueLike-v0"
WEIGHTS_FILENAME = "weights.h5"
RANDOM_SEED = 123


def create_model(model_cfg, input_shape, output_shape, verbose=False):
    model = Sequential()
    for i, layer in enumerate(model_cfg):
        name = layer["name"]
        kwargs = layer.get("kwargs", {})
        if i == 0:
            kwargs["input_shape"] = input_shape
        cls = eval(f"layers.{name}")
        model.add(cls(**kwargs))
    model.add(layers.Dense(output_shape, activation='relu'))
    if verbose:
        print(model.summary())
        print(model.layers[0].input_shape)
    return model


def plot_history(log_filename, plot_filename):
    with open(log_filename, 'r') as f:
        r = json.load(f)
    nb_episode_steps = r['nb_episode_steps']
    episode_reward = r['episode_reward']
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['font.size'] = 22
    fig, ax = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle('{} training history'.format(ENV_NAME))
    ax[0].plot(nb_episode_steps)
    ax[0].set_xlabel('episode')
    ax[0].set_ylabel('episode steps')
    ax[0].set_yscale('log')
    ax[1].plot(episode_reward)
    ax[1].set_xlabel('episode')
    ax[1].set_ylabel('episode reward')
    fig.savefig(plot_filename, bbox_inches='tight')
    return


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("mode", choices=["train", "test"])
    parser.add_argument("config")
    parser.add_argument("-w", "--weights")
    parser.add_argument("-t", "--continue-training", action="store_true", default=False)
    parser.add_argument("-n", "--nb-steps", type=int, default=None)
    parser.add_argument("-m", "--nb-max-episode-steps", type=int, default=None)
    parser.add_argument("-u", "--nb-steps-warmup", type=int, default=None)
    parser.add_argument("-e", "--epsilon-range", type=float, nargs=2, default=None)
    parser.add_argument("-a", "--action-repetition", type=int, default=None)
    parser.add_argument("-v", "--verbose", action="store_true", default=False)
    parser.add_argument("-o", "--outdir", default=None)
    parser.add_argument("-r", "--render", action="store_true", default=False)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.Loader)
    cfg = cfg["agent"]
    for k, v in args.__dict__.items():
        if k in cfg and v is not None:
            cfg[k] = v

    # Create output directory
    if cfg["outdir"] is None:
        cfg["outdir"] = os.getcwd()
    if not os.path.exists(cfg["outdir"]):
        os.mkdir(cfg["outdir"])

    # Initialize game environment
    env = gym.make(ENV_NAME, config=args.config)
    if args.mode == 'train':
        np.random.seed(RANDOM_SEED)
        env.seed(RANDOM_SEED)
    env.reset()
    nb_actions = env.action_space.n

    # Create model
    input_shape = (1,) + env.observation_space.shape
    model = create_model(cfg["model"], input_shape, nb_actions, verbose=args.verbose)

    # Create deep-Q agent
    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr='eps',
        value_max=max(cfg["epsilon_range"]),
        value_min=min(cfg["epsilon_range"]),
        value_test=min(cfg["epsilon_range"]),
        nb_steps=cfg["nb_steps"]
    )
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(
        model=model,
        nb_actions=nb_actions,
        memory=memory,
        nb_steps_warmup=cfg["nb_steps_warmup"],
        target_model_update=1e-2,
        policy=policy,
        gamma=0.9
    )
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    outdir = cfg["outdir"]
    weights_filename=os.path.join(outdir, "weights.h5")
    if args.verbose:
        verbose = 1
    else:
        verbose = 2

    if args.mode == 'train':
        callbacks = []
        if "checkpt_interval" in cfg.keys():
            checkpt_weights_filename = os.path.join(outdir, 'weights_{step}.h5f')
            checkpt = ModelIntervalCheckpoint(checkpt_weights_filename, interval=cfg["checkpt_interval"])
            callbacks.append(checkpt)
        log = "log_interval" in cfg.keys()
        if log:
            log_filename = os.path.join(outdir, 'log.json')
            plot_filename = os.path.join(outdir, 'plot.png')
            logger = FileLogger(log_filename, interval=cfg["log_interval"])
            callbacks.append(logger)
        dqn.fit(
            env,
            cfg["nb_steps"],
            callbacks=callbacks,
            action_repetition=cfg["action_repetition"],
            nb_max_episode_steps=cfg["nb_max_episode_steps"],
            visualize=args.render,
            verbose=verbose,
        )
        dqn.save_weights(weights_filename, overwrite=True)
        if log:
            plot_history(log_filename, plot_filename)
    elif args.mode == 'test':
        dqn.load_weights(weights_filename)
        dqn.test_policy = policy
        dqn.test(
            env,
            nb_episodes=100,
            action_repetition=cfg["action_repetition"],
            nb_max_episode_steps=100, #cfg["nb_max_episode_steps"],
            visualize=args.render,
            verbose=verbose,
        )

