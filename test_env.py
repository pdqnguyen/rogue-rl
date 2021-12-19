from argparse import ArgumentParser
# from tensorforce.environments import Environment
import gym
import gym_roguelike


parser = ArgumentParser()
parser.add_argument("config")
parser.add_argument("--human", action="store_true", default=False)
parser.add_argument("-v", "--verbose", action="store_true", default=False)
args = parser.parse_args()

env = gym.make('RogueLike-v0', config=args.config)
env.render()
if args.human:
    from pyglet.window import key
    action = 0

    def key_press(k, mod):
        global action, restart, quit
        if k == key.BACKSPACE:
            restart = True
        if k == key.ESCAPE:
            quit = True
        if k == key.A:
            action = 1
        if k == key.D:
            action = 2
        if k == key.W:
            action = 3
        if k == key.S:
            action = 4

    def key_release(k, mod):
        global action
        if k == key.A and action != 0:
            action = 0
        if k == key.D and action != 0:
            action = 0
        if k == key.W and action != 0:
            action = 0
        if k == key.S and action != 0:
            action = 0

    env.window.on_key_press = key_press
    env.window.on_key_release = key_release

quit = False
while not quit:
    env.reset()
    total_reward = 0.0
    steps = 0
    restart = False
    while True:
        if not args.human:
            action = env.action_space.sample()
        s, r, done, info = env.step(action)
        total_reward += r
        steps += 1
        env.render()
        if done or restart or quit:
            break
        # print(total_reward)
env.close()
