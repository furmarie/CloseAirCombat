import numpy as np
import torch
from envs.JSBSim.envs import SingleCombatEnv, SingleControlEnv, MultipleCombatEnv
from envs.env_wrappers import SubprocVecEnv, DummyVecEnv
from envs.JSBSim.core.catalog import Catalog as c
from envs.JSBSim.model.baseline_actor import BaselineActor as Actor
import logging
logging.basicConfig(level=logging.DEBUG)


def _t2n(x):
    return x.detach().cpu().numpy()

num_agents = 1
parallel_num = 1
experiment_name = 'new_v2'
run_dir = "/home/lqh/jyh/CloseAirCombat/scripts/results/SingleCombat/1v1/Missile/vsBaseline/ppo/new_v2/wandb/latest-run/files"


envs = DummyVecEnv([lambda: SingleControlEnv("1/heading")])
obss = envs.reset()
policy = Actor()
policy.load_state_dict(torch.load(run_dir+"/actor.pt"))
policy.eval()
actions = np.array([[envs.action_space.sample() for _ in range(envs.num_agents)] for _ in range(parallel_num)])


render_episode_rewards = 0
render_obs = envs.reset()
render_rnn_states = np.zeros((1, num_agents, 1, 128), dtype=np.float32)
envs.render(mode='txt', filepath=f'{experiment_name}.txt.acmi')
while True:
    render_actions, render_rnn_states = policy(np.concatenate(render_obs), np.concatenate(render_rnn_states),)
    render_actions = np.expand_dims(_t2n(render_actions), axis=0)
    render_rnn_states = np.expand_dims(_t2n(render_rnn_states), axis=0)

    # Obser reward and next obs
    render_obs, render_rewards, render_dones, render_infos = envs.step(render_actions)
    render_episode_rewards += render_rewards
    envs.render(mode='txt', filepath=f'{experiment_name}.txt.acmi')
    if render_dones.any():
        break

render_infos = {}
render_infos['render_episode_reward'] = render_episode_rewards
print("render episode reward of agent: " + str(render_infos['render_episode_reward']))
envs.close()