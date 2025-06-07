from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from gymnasium.utils.save_video import save_video


def reward_fn(state) -> float:
    state = state.decode('UTF-8')
    if state == 'G':
        return 1
    elif state == 'H' or state == 'S' or state == 'F':
        return 0
    else:
        raise ValueError(f"Unknown state: {state}")


def get_state_reward(env_desc: np.ndarray) -> np.ndarray:
    state_space = env_desc.ravel()
    state_len = len(state_space)

    rewards = np.zeros(state_len)

    for idx, state in enumerate(state_space):
        rewards[idx] = reward_fn(state)

    return rewards


def greedy_action(obs, v, probabilities) -> int:
    current_state = probabilities[obs]

    next_states = {prob_tuple[0][1] for prob_tuple in current_state.values()}

    actions_to_state = {action: prob_tuple[0][1] for action, prob_tuple in current_state.items()}
    action_to_value = {action: v[state] for action, state in actions_to_state.items()}

    return max(action_to_value, key=action_to_value.get)


def to_s(row, col, ncol):
    return row * ncol + col


def fill_prob_matrix(P, matrix_length, num_actions, prob_dist):
    for row in range(matrix_length):
        for col in range(matrix_length):
            for action in P[row]:
                prob_dist[row][col] += (1 / num_actions) * (P[row][action][0][1] == col)


def new_frozenlake_experiment(is_slippery: bool = False, desc: Optional[str] = None, size=8, render_mode="human",
                              gamma=0.9):
    if desc is None:
        desc = generate_random_map(size=size)

    env = gym.make("FrozenLake-v1", is_slippery=is_slippery, desc=desc, render_mode=render_mode)

    num_obs = env.observation_space.n
    num_actions = env.action_space.n
    n_rows = env.nrow
    n_cols = env.ncol

    rewards = get_state_reward(env.desc)
    matrix_length = n_rows * n_cols
    prob_dist = np.zeros((matrix_length, matrix_length))

    fill_prob_matrix(env.P, matrix_length, num_actions, prob_dist)

    v = np.dot(np.linalg.inv(np.identity(matrix_length) - gamma * prob_dist), rewards)

    run_episode(env, v)

    env.close()


def main():
    for _ in range(10):
        # new_frozenlake_experiment(size=4, render_mode="rgb_array_list")
        new_frozenlake_experiment(size=8, render_mode="human")


def run_episode(env, v, steps=1_000):
    for _ in range(steps):
        obs, _ = env.reset()

        step_starting_index = 0
        episode_index = 0

        while True:
            action = greedy_action(obs, v, env.P)
            new_obs, reward, is_done, truncated, _ = env.step(action)

            env.render()

            if is_done or truncated:
                # save_video(
                #     env.render(),
                #     "./videos",
                #     fps=env.metadata["render_fps"],
                #     step_starting_index=step_starting_index,
                #     episode_index=episode_index
                # )
                break

            obs = new_obs

        if is_done or truncated:
            break


if __name__ == '__main__':
    main()
