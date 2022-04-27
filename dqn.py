from argparse import ArgumentParser
import json
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.processors import Processor
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory
from rl.callbacks import ModelIntervalCheckpoint, FileLogger

import gym
import gym_roguelike


ENV_NAME = "RogueLike-v0"
WEIGHTS_FILENAME = "weights.h5"
RANDOM_SEED = 123


def create_model(layer_dict, input_shape, output_shape, verbose=False):
    model = Sequential()
    for i, layer in enumerate(layer_dict):
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
        metrics = json.load(f)
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['font.size'] = 22
    fig, ax = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('{} training history'.format(ENV_NAME))
    fig.subplots_adjust(hspace=0.4)
    ax[0, 0].plot(metrics['episode_reward'])
    ax[0, 0].set_title('episode reward')
    ax[0, 0].set_yscale('log')
    ax[0, 1].plot(np.cumsum(metrics['episode_reward']))
    ax[0, 1].set_title('cumulative episode reward')
    ax[1, 0].plot(metrics['loss'])
    ax[1, 0].set_xlabel('episode')
    ax[1, 0].set_title('loss')
    ax[1, 0].set_yscale('log')
    ax[1, 1].plot(metrics['mae'])
    ax[1, 1].set_xlabel('episode')
    ax[1, 1].set_title('mean absolute error')
    fig.savefig(plot_filename, bbox_inches='tight')
    return


class MyProcessor(Processor):
    def process_state_batch(self, batch):
        return batch.reshape((batch.shape[0],) + batch.shape[2:])


def main(args):
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
    model_params = cfg["model"]
    input_shape = env.observation_space.shape
    model = create_model(model_params["layers"], input_shape, nb_actions, verbose=args.verbose)

    # Create deep-Q agent
    policy_params = cfg["policy"]
    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr='eps',
        value_max=policy_params["eps_min"],
        value_min=policy_params["eps_max"],
        value_test=policy_params["eps_test"],
        nb_steps=cfg["nb_steps"],
    )
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(
        model=model,
        processor=MyProcessor(),
        nb_actions=nb_actions,
        memory=memory,
        nb_steps_warmup=cfg["nb_steps_warmup"],
        target_model_update=cfg["target_model_update"],
        policy=policy,
        test_policy=policy,
        gamma=cfg["gamma"],
    )
    dqn.compile(Adam(lr=model_params["learning_rate"]), metrics=model_params["metrics"])
    outdir = cfg["outdir"]
    if args.weights is not None:
        weights_filename = args.weights
    else:
        weights_filename = os.path.join(outdir, "weights.h5")
    if args.verbose:
        verbose = 1
    else:
        verbose = 2

    if args.mode == 'train':
        old_weights = cfg["train_from_weights"]
        if old_weights is not None:
            print(old_weights)
            dqn.load_weights(old_weights)
        callbacks = []
        if "chkpt_interval" in cfg.keys():
            chkpt_weights_filename = os.path.join(outdir, 'chkpt_weights_{step}.h5')
            chkpt = ModelIntervalCheckpoint(chkpt_weights_filename, interval=cfg["chkpt_interval"])
            callbacks.append(chkpt)
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
        dqn.test(
            env,
            nb_episodes=100,
            action_repetition=cfg["action_repetition"],
            nb_max_episode_steps=cfg["nb_max_test_episode_steps"],
            visualize=args.render,
            verbose=verbose,
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("mode", choices=["train", "test"])
    parser.add_argument("config")
    parser.add_argument("-w", "--weights")
    parser.add_argument("-n", "--nb-steps", type=int, default=None)
    parser.add_argument("-u", "--nb-steps-warmup", type=int, default=None)
    parser.add_argument("-m", "--nb-max-episode-steps", type=int, default=None)
    parser.add_argument("-t", "--nb-max-test-episode-steps", type=int, default=None)
    parser.add_argument("-a", "--action-repetition", type=int, default=None)
    parser.add_argument("-o", "--outdir", default=None)
    parser.add_argument("-v", "--verbose", action="store_true", default=False)
    parser.add_argument("-r", "--render", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
