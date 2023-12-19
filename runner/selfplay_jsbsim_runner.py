import torch
import logging
import time
import numpy as np
from typing import List
from .base_runner import Runner, ReplayBuffer
from .jsbsim_runner import JSBSimRunner

from envs.JSBSim.core.catalog import Catalog as c


def _t2n(x):
    return x.detach().cpu().numpy()


class SelfplayJSBSimRunner(JSBSimRunner):

    def load(self):
        self.use_selfplay = self.all_args.use_selfplay 
        assert self.use_selfplay == True, "Only selfplay can use SelfplayRunner"
        self.obs_space = self.envs.observation_space
        self.act_space = self.envs.action_space
        self.num_agents = self.envs.num_agents
        self.num_opponents = self.all_args.n_choose_opponents
        assert self.eval_episodes >= self.num_opponents, \
        f"Number of evaluation episodes:{self.eval_episodes} should be greater than number of opponents:{self.num_opponents}"
        self.init_elo = self.all_args.init_elo
        self.latest_elo = self.init_elo

        # policy & algorithm
        if self.algorithm_name == "ppo":
            from algorithms.ppo.ppo_trainer import PPOTrainer as Trainer
            from algorithms.ppo.ppo_policy import PPOPolicy as Policy
        else:
            raise NotImplementedError
        self.policy = Policy(self.all_args, self.obs_space, self.act_space, device=self.device)
        self.trainer = Trainer(self.all_args, device=self.device)

        # buffer
        self.buffer = ReplayBuffer(self.all_args, self.num_agents // 2, self.obs_space, self.act_space)

        # [Selfplay] allocate memory for opponent policy/data in training
        from algorithms.utils.selfplay import get_algorithm
        self.selfplay_algo = get_algorithm(self.all_args.selfplay_algorithm)

        assert self.num_opponents <= self.n_rollout_threads, \
            "Number of different opponents({}) must less than or equal to number of training threads({})!" \
            .format(self.num_opponents, self.n_rollout_threads)
        self.policy_pool = {}  # type: dict[str, float]
        self.opponent_policy = [
            Policy(self.all_args, self.obs_space, self.act_space, device=self.device)
            for _ in range(self.num_opponents)]
        self.opponent_env_split = np.array_split(np.arange(self.n_rollout_threads), len(self.opponent_policy))
        self.opponent_obs = np.zeros_like(self.buffer.obs[0])
        self.opponent_rnn_states = np.zeros_like(self.buffer.rnn_states_actor[0])
        self.opponent_masks = np.ones_like(self.buffer.masks[0])

        if self.use_eval:
            self.eval_opponent_policy = Policy(self.all_args, self.obs_space, self.act_space, device=self.device)

        logging.info("\n Load selfplay opponents: Algo {}, num_opponents {}.\n"
                        .format(self.all_args.selfplay_algorithm, self.num_opponents))

        if self.model_dir is not None:
            self.restore()

    def warmup(self):
        # reset env
        obs = self.envs.reset()
        # [Selfplay] divide ego/opponent of initial obs
        self.opponent_obs = obs[:, self.num_agents // 2:, ...]
        obs = obs[:, :self.num_agents // 2, ...]
        self.buffer.step = 0
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step):
        self.policy.prep_rollout()
        values, actions, action_log_probs, rnn_states_actor, rnn_states_critic \
            = self.policy.get_actions(np.concatenate(self.buffer.obs[step]),
                                      np.concatenate(self.buffer.rnn_states_actor[step]),
                                      np.concatenate(self.buffer.rnn_states_critic[step]),
                                      np.concatenate(self.buffer.masks[step]))
        # split parallel data [N*M, shape] => [N, M, shape]
        values = np.array(np.split(_t2n(values), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(actions), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_probs), self.n_rollout_threads))
        rnn_states_actor = np.array(np.split(_t2n(rnn_states_actor), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))

        # [Selfplay] get actions of opponent policy
        opponent_actions = np.zeros_like(actions)
        for policy_idx, policy in enumerate(self.opponent_policy):
            env_idx = self.opponent_env_split[policy_idx]
            opponent_action, opponent_rnn_states \
                = policy.act(np.concatenate(self.opponent_obs[env_idx]),
                                np.concatenate(self.opponent_rnn_states[env_idx]),
                                np.concatenate(self.opponent_masks[env_idx]))
            opponent_actions[env_idx] = np.array(np.split(_t2n(opponent_action), len(env_idx)))
            self.opponent_rnn_states[env_idx] = np.array(np.split(_t2n(opponent_rnn_states), len(env_idx)))
        actions = np.concatenate((actions, opponent_actions), axis=1)

        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def insert(self, data: List[np.ndarray]):
        obs, actions, rewards, dones, action_log_probs, values, rnn_states_actor, rnn_states_critic = data

        dones_env = np.all(dones.squeeze(axis=-1), axis=-1)

        rnn_states_actor[dones_env == True] = np.zeros(((dones_env == True).sum(), *rnn_states_actor.shape[1:]), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), *rnn_states_critic.shape[1:]), dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        # [Selfplay] divide ego/opponent of collecting data
        self.opponent_obs = obs[:, self.num_agents // 2:, ...]
        self.opponent_masks = masks[:, self.num_agents // 2:, ...]
        self.opponent_rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), *rnn_states_actor.shape[1:]), dtype=np.float32)
        
        obs = obs[:, :self.num_agents // 2, ...]
        actions = actions[:, :self.num_agents // 2, ...]
        rewards = rewards[:, :self.num_agents // 2, ...]
        masks = masks[:, :self.num_agents // 2, ...]

        self.buffer.insert(obs, actions, rewards, masks, action_log_probs, values, rnn_states_actor, rnn_states_critic)

    @torch.no_grad()
    def eval(self, total_num_steps):
        logging.info("\nStart evaluation...")
        self.policy.prep_rollout()
        total_episodes = 0
        episode_rewards, opponent_episode_rewards = [], []
        cumulative_rewards = np.zeros((self.n_eval_rollout_threads, *self.buffer.rewards.shape[2:]), dtype=np.float32)
        opponent_cumulative_rewards= np.zeros_like(cumulative_rewards)

        # [Selfplay] Choose opponent policy for evaluation
        eval_choose_opponents = [self.selfplay_algo.choose(self.policy_pool) for _ in range(self.num_opponents)]
        eval_each_episodes = self.eval_episodes // self.num_opponents
        logging.info(f" Choose opponents {eval_choose_opponents} for evaluation")

        eval_cur_opponent_idx = 0
        while total_episodes < self.eval_episodes:

            # [Selfplay] Load opponent policy
            if total_episodes >= eval_cur_opponent_idx * eval_each_episodes:
                policy_idx = eval_choose_opponents[eval_cur_opponent_idx]
                self.eval_opponent_policy.actor.load_state_dict(torch.load(str(self.save_dir) + f'/actor_{policy_idx}.pt'))
                self.eval_opponent_policy.prep_rollout()
                eval_cur_opponent_idx += 1
                logging.info(f" Load opponent {policy_idx} for evaluation ({total_episodes}/{self.eval_episodes})")

                # reset obs/rnn/mask
                obs = self.eval_envs.reset()
                masks = np.ones((self.n_eval_rollout_threads, *self.buffer.masks.shape[2:]), dtype=np.float32)
                rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states_actor.shape[2:]), dtype=np.float32)
                opponent_obs = obs[:, self.num_agents // 2:, ...]
                obs = obs[:, :self.num_agents // 2, ...]
                opponent_masks = np.ones_like(masks, dtype=np.float32)
                opponent_rnn_states = np.zeros_like(rnn_states, dtype=np.float32)

            # [Selfplay] get actions
            actions, rnn_states = self.policy.act(np.concatenate(obs),
                                                    np.concatenate(rnn_states),
                                                    np.concatenate(masks), deterministic=True)
            actions = np.array(np.split(_t2n(actions), self.n_eval_rollout_threads))
            rnn_states = np.array(np.split(_t2n(rnn_states), self.n_eval_rollout_threads))

            opponent_actions, opponent_rnn_states \
                = self.eval_opponent_policy.act(np.concatenate(opponent_obs),
                                                np.concatenate(opponent_rnn_states),
                                                np.concatenate(opponent_masks), deterministic=True)
            opponent_rnn_states = np.array(np.split(_t2n(opponent_rnn_states), self.n_eval_rollout_threads))
            opponent_actions = np.array(np.split(_t2n(opponent_actions), self.n_eval_rollout_threads))
            actions = np.concatenate((actions, opponent_actions), axis=1)

            # Obser reward and next obs
            obs, eval_rewards, dones, eval_infos = self.eval_envs.step(actions)
            dones_env = np.all(dones.squeeze(axis=-1), axis=-1)
            total_episodes += np.sum(dones_env)

            # [Selfplay] Reset obs, masks, rnn_states
            opponent_obs = obs[:, self.num_agents // 2:, ...]
            obs = obs[:, :self.num_agents // 2, ...]
            masks[dones_env == True] = np.zeros(((dones_env == True).sum(), *masks.shape[1:]), dtype=np.float32)
            rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), *rnn_states.shape[1:]), dtype=np.float32)

            opponent_masks[dones_env == True] = \
                np.zeros(((dones_env == True).sum(), *opponent_masks.shape[1:]), dtype=np.float32)
            opponent_rnn_states[dones_env == True] = \
                np.zeros(((dones_env == True).sum(), *opponent_rnn_states.shape[1:]), dtype=np.float32)

            # [Selfplay] Get rewards
            opponent_rewards = eval_rewards[:, self.num_agents//2:, ...]
            opponent_cumulative_rewards += opponent_rewards
            opponent_episode_rewards.append(opponent_cumulative_rewards[dones_env == True])
            opponent_cumulative_rewards[dones_env == True] = 0

            eval_rewards = eval_rewards[:, :self.num_agents // 2, ...]
            cumulative_rewards += eval_rewards
            episode_rewards.append(cumulative_rewards[dones_env == True])
            cumulative_rewards[dones_env == True] = 0

        # Compute average episode rewards
        episode_rewards = np.concatenate(episode_rewards) # shape (self.eval_episodes, self.num_agents, 1)
        episode_rewards = episode_rewards.squeeze(-1).mean(axis=-1) # shape: (self.eval_episodes,)
        eval_average_episode_rewards = np.array(np.split(episode_rewards, self.num_opponents)).mean(axis=-1) # shape (self.num_opponents,)

        opponent_episode_rewards = np.concatenate(opponent_episode_rewards)
        opponent_episode_rewards = opponent_episode_rewards.squeeze(-1).mean(axis=-1)
        opponent_average_episode_rewards = np.array(np.split(opponent_episode_rewards, self.num_opponents)).mean(axis=-1)

        # Update elo
        ego_elo = np.array([self.latest_elo for _ in range(self.n_eval_rollout_threads)])
        opponent_elo = np.array([self.policy_pool[key] for key in eval_choose_opponents])
        expected_score = 1 / (1 + 10**((opponent_elo-ego_elo)/400))

        actual_score = np.zeros_like(expected_score)
        diff = opponent_average_episode_rewards - eval_average_episode_rewards
        actual_score[diff > 100] = 1 # win
        actual_score[abs(diff) < 100] = 0.5 # tie
        actual_score[diff < -100] = 0 # lose

        elo_gain = 32 * (actual_score - expected_score)
        update_opponent_elo = opponent_elo + elo_gain
        for i, key in enumerate(eval_choose_opponents):
            self.policy_pool[key] = update_opponent_elo[i]
        ego_elo = ego_elo - elo_gain
        self.latest_elo = ego_elo.mean()

        # Logging
        eval_infos = {}
        eval_infos['eval_average_episode_rewards'] = eval_average_episode_rewards.mean()
        eval_infos['latest_elo'] = self.latest_elo
        logging.info(" eval average episode rewards: " + str(eval_infos['eval_average_episode_rewards']))
        logging.info(" latest elo score: " + str(self.latest_elo))
        self.log_info(eval_infos, total_num_steps)
        logging.info("...End evaluation")

        # [Selfplay] Reset opponent for the following training
        self.reset_opponent()
        
    def save(self, episode):
        policy_actor_state_dict = self.policy.actor.state_dict()
        torch.save(policy_actor_state_dict, str(self.save_dir) + '/actor_latest.pt')
        policy_critic_state_dict = self.policy.critic.state_dict()
        torch.save(policy_critic_state_dict, str(self.save_dir) + '/critic_latest.pt')
        # [Selfplay] save policy & performance
        torch.save(policy_actor_state_dict, str(self.save_dir) + f'/actor_{episode}.pt')
        self.policy_pool[str(episode)] = self.latest_elo

    def reset_opponent(self):
        choose_opponents = []
        for policy in self.opponent_policy:
            choose_idx = self.selfplay_algo.choose(self.policy_pool)
            choose_opponents.append(choose_idx)
            policy.actor.load_state_dict(torch.load(str(self.save_dir) + f'/actor_{choose_idx}.pt'))
            policy.prep_rollout()
        logging.info(f" Choose opponents {choose_opponents} for training")

        # clear buffer
        self.buffer.clear()
        self.opponent_obs = np.zeros_like(self.opponent_obs)
        self.opponent_rnn_states = np.zeros_like(self.opponent_rnn_states)
        self.opponent_masks = np.ones_like(self.opponent_masks)

        # reset env
        obs = self.envs.reset()
        if self.num_opponents > 0:
            self.opponent_obs = obs[:, self.num_agents // 2:, ...]
            obs = obs[:, :self.num_agents // 2, ...]
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def render(self):
        idx = self.all_args.render_index
        opponent_idx = self.all_args.render_opponent_index
        dir_list = str(self.run_dir).split('/')
        file_path = '/'.join(dir_list[:dir_list.index('results')+1])
        
        # Load models trained for Heading task here
        self.policy.actor.load_state_dict(torch.load(
            str(self.model_dir) + f'/actor_latest.pt', map_location=torch.device('cpu')))
        self.policy.prep_rollout()
        self.eval_opponent_policy.actor.load_state_dict(torch.load(
            str(self.model_dir) + f'/actor_latest.pt', map_location=torch.device('cpu')))
        self.eval_opponent_policy.prep_rollout()

        env = self.envs.envs[0]
        state_var = env.task.state_var
        action_var = env.task.action_var
        all_vars = state_var + action_var + [c.position_long_gc_deg,
                                             c.position_lat_geod_deg,
                                             ]

        # Created logic for following another aircraft in PursueAgent
        pursue_agent = PursueAgent(env, 'A0100')
        maneuver_agent = ManeuverAgent(env, 'B0100')

        res_filepath = f'{self.run_dir}/{self.experiment_name}_{time.strftime("%m_%d_%H_%M_%S")}.csv'
        print(res_filepath)
        with open(res_filepath, "w") as f:
            res = ", ".join([x.name_jsbsim for x in all_vars])
            f.write(res + '\n')

        logging.info("\nStart render ...")
        out_path = f'{file_path}/{self.experiment_name}.txt.acmi'
        print(out_path)
        render_episode_rewards = 0
        render_obs = self.envs.reset()
        self.envs.render(mode='txt', filepath=out_path)
        render_masks = np.ones(
            (1, *self.buffer.masks.shape[2:]), dtype=np.float32)
        render_rnn_states = np.zeros(
            (1, *self.buffer.rnn_states_actor.shape[2:]), dtype=np.float32)
        render_opponent_obs = render_obs[:, self.num_agents // 2:, ...]
        render_obs = render_obs[:, :self.num_agents // 2, ...]
        render_opponent_masks = np.ones_like(render_masks, dtype=np.float32)
        render_opponent_rnn_states = np.zeros_like(
            render_rnn_states, dtype=np.float32)
        while True:
            self.policy.prep_rollout()
            actor_input = [np.concatenate(render_obs), np.concatenate(render_rnn_states),
                np.concatenate(render_masks), True]

            render_actions, render_rnn_states = self.policy.act(np.concatenate(render_obs),
                                                                np.concatenate(
                                                                    render_rnn_states),
                                                                np.concatenate(
                                                                    render_masks),
                                                                deterministic=True)

            render_actions = np.expand_dims(_t2n(render_actions), axis=0)
            render_rnn_states = np.expand_dims(_t2n(render_rnn_states), axis=0)
            render_opponent_actions, render_opponent_rnn_states \
                = self.eval_opponent_policy.act(np.concatenate(render_opponent_obs),
                                                np.concatenate(
                                                    render_opponent_rnn_states),
                                                np.concatenate(
                                                    render_opponent_masks),
                                                deterministic=True)
            render_opponent_actions = np.expand_dims(
                _t2n(render_opponent_actions), axis=0)
            render_opponent_rnn_states = np.expand_dims(
                _t2n(render_opponent_rnn_states), axis=0)
            
            render_actions = np.concatenate(
                (render_actions, render_opponent_actions), axis=1)
            # Obser reward and next obs
            render_obs, render_rewards, render_dones, render_infos = self.envs.step(
                render_actions)

            # Get observations and set target values for each agent separately
            # We can pursue agent follow the other aircraft this way
            render_obs = pursue_agent.get_obs(env, env.task)
            render_opponent_obs = maneuver_agent.get_obs(env, env.task)
            render_obs = render_obs.reshape((1, 1, 12))
            render_opponent_obs = render_opponent_obs.reshape((1, 1, 12))

            render_rewards = render_rewards[:, :self.num_agents // 2, ...]
            render_episode_rewards += render_rewards
            # self.envs.render(mode='txt', filepath=out_path)
            if render_dones.all():
                break

            renders = env.agents['A0100'].get_property_values(all_vars)
            renders = [round(x, 3) for x in renders]

            with open(res_filepath, "a") as f:
                res = ", ".join([str(x) for x in renders])
                f.write(res + '\n')
                
        print(render_episode_rewards)


class BaselineAgent():
    def __init__(self, agent_id) -> None:
        self.agent_id = agent_id

    def reset(self):
        self.rnn_states = np.zeros((1, 1, 128))

    def set_delta_value(self, env, task):
        raise NotImplementedError

    def get_action(self, env, task):
        delta_value = self.set_delta_value(env, task)
        observation = self.get_observation(env, task, delta_value)
        _action, self.rnn_states = self.actor(observation, self.rnn_states)
        action = _action.detach().cpu().numpy().squeeze()
        return action

    def get_obs(self, env, task):
        # print(f"{self.agent_id} :", env.agents[self.agent_id].get_property_value(
        #     c.target_altitude_ft))
        self.set_delta_value(env, task)
        observation = env.task.get_obs(env, self.agent_id)
        return observation


class PursueAgent(BaselineAgent):
    def __init__(self, env, agent_id) -> None:
        super().__init__(agent_id)

    def set_delta_value(self, env, task):
        # NOTE: only adapt for 1v1
        ego_uid, enm_uid = list(env.agents.keys())[0], list(
            env.agents.keys())[1]

        print(ego_uid, enm_uid)
        ego_x, ego_y, ego_z = env.agents[ego_uid].get_position()
        ego_vx, ego_vy, ego_vz = env.agents[ego_uid].get_velocity()
        enm_x, enm_y, enm_z = env.agents[enm_uid].get_position()
        # delta altitude
        delta_altitude = enm_z - ego_z
        # delta heading
        ego_v = np.linalg.norm([ego_vx, ego_vy])
        delta_x, delta_y = enm_x - ego_x, enm_y - ego_y
        R = np.linalg.norm([delta_x, delta_y])
        proj_dist = delta_x * ego_vx + delta_y * ego_vy
        ego_AO = np.arccos(np.clip(proj_dist / (R * ego_v + 1e-8), -1, 1))
        side_flag = np.sign(np.cross([ego_vx, ego_vy], [delta_x, delta_y]))
        delta_heading = ego_AO * side_flag
        # delta velocity
        delta_velocity = env.agents[enm_uid].get_property_value(c.velocities_u_mps) - \
            env.agents[ego_uid].get_property_value(c.velocities_u_mps)

        env.agents[self.agent_id].set_property_value(
            c.target_altitude_ft, enm_z * 3.281)
        cur_heading = env.agents[self.agent_id].get_property_value(
            c.attitude_heading_true_rad)
        delta_heading = ((delta_heading * 180) + 360) % 360
        new_heading = cur_heading + delta_heading
        env.agents[self.agent_id].set_property_value(
            c.target_heading_deg, new_heading)
        env.agents[self.agent_id].set_property_value(c.velocities_u_mps,
                                                     env.agents[enm_uid].get_property_value(
                                                         c.velocities_u_mps)
                                                     )

        return np.array([delta_altitude, delta_heading, delta_velocity])


class ManeuverAgent(BaselineAgent):
    def __init__(self, env, agent_id) -> None:
        super().__init__(agent_id)
        self.init_heading = None

        self.turn_interval = 30

        self.target_altitude_list = [6000] * 6
        self.target_velocity_list = [243] * 6

        self.heading_turn_counts = 0

        self.max_heading_increment = 180
        self.max_altitude_increment = 7000
        self.max_velocities_u_increment = 100
        self.check_interval = 10
        self.increment_size = [0.2, 0.4, 0.6, 0.8, 1.0] + [1.0] * 100

    def reset(self):
        self.step = 0
        self.rnn_states = np.zeros((1, 1, 128))
        self.init_heading = None

    def set_delta_value(self, env, task):
        uid = self.agent_id
        cur_heading = env.agents[uid].get_property_value(
            c.attitude_heading_true_rad)
        if self.init_heading is None:
            env.agents[uid].set_property_value(c.heading_check_time, 0)
            self.init_heading = cur_heading
            env.agents[uid].set_property_value(c.target_altitude_ft, 20000)

        check_time = env.agents[uid].get_property_value(c.heading_check_time)
        if env.agents[uid].get_property_value(c.simulation_sim_time_sec) >= check_time:
            delta = self.increment_size[self.heading_turn_counts]
            delta_heading = env.np_random.uniform(
                -delta, delta) * self.max_heading_increment
            delta_altitude = env.np_random.uniform(
                -delta, delta) * self.max_altitude_increment
            delta_velocities_u = env.np_random.uniform(
                -delta, delta) * self.max_velocities_u_increment
            new_heading = env.agents[uid].get_property_value(
                c.target_heading_deg) + delta_heading
            new_heading = (new_heading + 360) % 360
            new_altitude = env.agents[uid].get_property_value(
                c.target_altitude_ft) + delta_altitude
            new_velocities_u = env.agents[uid].get_property_value(
                c.target_velocities_u_mps) + delta_velocities_u
            env.agents[uid].set_property_value(
                c.target_heading_deg, new_heading)
            env.agents[uid].set_property_value(
                c.target_altitude_ft, new_altitude)
            env.agents[uid].set_property_value(
                c.target_velocities_u_mps, new_velocities_u)
            env.agents[uid].set_property_value(
                c.heading_check_time, check_time + self.check_interval)
            self.heading_turn_counts += 1
            print(f'target_heading:{new_heading} '
                  f'target_altitude_ft:{new_altitude} target_velocities_u_mps:{new_velocities_u}')

        delta_heading = env.agents[uid].get_property_value(c.delta_heading)

        delta_altitude = env.agents[uid].get_property_value(c.target_altitude_ft) / 3.281 - \
            env.agents[uid].get_property_value(c.position_h_sl_m)
        
        delta_velocity = 243 - \
            env.agents[uid].get_property_value(c.velocities_u_mps)
        return np.array([delta_altitude, delta_heading, delta_velocity])
