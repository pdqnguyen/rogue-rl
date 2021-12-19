from gym.envs.registration import register

register(
    id='RogueLike-v0',
    entry_point='gym_roguelike.envs:RogueLike',
)