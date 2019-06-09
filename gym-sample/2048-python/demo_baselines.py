from baselines import deepq
from gym_2048.env.game2048_env import Game2048Env


def callback(lcl, _glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 1000 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 19900
    return is_solved


def main():
    env = Game2048Env()
    obs = env.reset()
    print("state space number:", env.observation_space.shape[0])
    print("type of action:", env.action_space)
    act = deepq.learn(
        env,
        network='mlp',
        lr=1e-3,
        total_timesteps=1000000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=10,
        callback=callback
    )
    print("Saving model to cartpole_model.pkl")
    act.save("game2048.pkl")


if __name__ == '__main__':
    main()
