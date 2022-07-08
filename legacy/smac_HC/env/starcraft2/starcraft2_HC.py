from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from smac_HC.env.multiagentenv import MultiAgentEnv
from smac_HC.env.starcraft2.maps import get_map_params

import atexit
from operator import attrgetter
from copy import deepcopy
import numpy as np
import enum
import math
# from absl import logging
import logging

from pysc2 import maps
from pysc2 import run_configs
from pysc2.lib import protocol
# from pysc2.lib import sc_process

from s2clientprotocol import common_pb2 as sc_common
from s2clientprotocol import sc2api_pb2 as sc_pb
from s2clientprotocol import raw_pb2 as r_pb
from s2clientprotocol import debug_pb2 as d_pb

import time

races = {
    "R": sc_common.Random,
    "P": sc_common.Protoss,
    "T": sc_common.Terran,
    "Z": sc_common.Zerg,
}

difficulties = {
    "1": sc_pb.VeryEasy,
    "2": sc_pb.Easy,
    "3": sc_pb.Medium,
    "4": sc_pb.MediumHard,
    "5": sc_pb.Hard,
    "6": sc_pb.Harder,
    "7": sc_pb.VeryHard,
    "8": sc_pb.CheatVision,
    "9": sc_pb.CheatMoney,
    "A": sc_pb.CheatInsane,
}

actions = {
    "move": 16,  # target: PointOrUnit
    "attack": 23,  # target: PointOrUnit
    "stop": 4,  # target: None
    "heal": 386,  # Unit
}


class Direction(enum.IntEnum):
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3


class StarCraft2Env_HC(MultiAgentEnv):
    """The StarCraft II environment for decentralised multi-agent
    micromanagement scenarios.
    """

    def __init__(
            self,
            map_name="Sandbox-4t",
            step_mul=8,
            move_amount=1,  # to reduce the step size of the movements of agents. The default value is 2.
            difficulty="7",
            game_version=None,
            seed=None,
            continuing_episode=False,
            obs_all_health=True,
            obs_own_health=True,
            obs_last_action=False,
            obs_pathing_grid=True,
            # obs_pathing_grid=True,
            obs_terrain_height=False,
            obs_instead_of_state=False,
            obs_timestep_number=False,
            state_last_action=True,
            state_timestep_number=False,
            reward_sparse=False,
            reward_only_positive=True,
            reward_death_value=10,
            reward_win=200,
            reward_defeat=0,
            reward_negative_scale=0.5,
            reward_scale=True,
            reward_scale_rate=20,
            # ---------------------------------
            # Add rewards for moving to strategic points.
            reward_StrategicPoint_val=20,
            reward_StrategicPoint_loc=[39, 41],

            # [OLD] reward when arrive certain locations, each agents will gain 5 points if it reaches a area given
            # reward_reachArea = 5,
            # reachArea_effect_range = 4,  # radius is 4 by default.
            # areaCenter = [[20, 23], [34, 34], [44, 48], [58, 58]],
            # --------------------------------
            replay_dir="",
            replay_prefix="",
            window_size_x=1920,
            window_size_y=1200,
            heuristic_ai=False,
            heuristic_rest=False,
            debug=False,
            # debug=True # activate log
            alpha=0.5,
            # add direction
            obs_direction_command=True,

            # parameters for hierarchical  control
            hierarchical=True,
            n_ally_platoons=3,
            n_ally_agent_in_platoon=4,
            n_enemy_platoons=1,
            n_enemy_unit_in_platoon=4,
            map_sps=None,
            train_on_intervals=None,
            FALCON_demo=None,
            obs_distance_target=True,
    ):
        """
        Create a StarCraftC2Env environment.

        Parameters
        ----------
        map_name : str, optional
            The name of the SC2 map to play (default is "8m"). The full list
            can be found by running bin/map_list.
        step_mul : int, optional
            How many game steps per agent step (default is 8). None
            indicates to use the default map step_mul.
        move_amount : float, optional
            How far away units are ordered to move per step (default is 2).
        difficulty : str, optional
            The difficulty of built-in computer AI bot (default is "7").
        game_version : str, optional
            StarCraft II game version (default is None). None indicates the
            latest version.
        seed : int, optional
            Random seed used during game initialisation. This allows to
        continuing_episode : bool, optional
            Whether to consider episodes continuing or finished after time
            limit is reached (default is False).
        obs_all_health : bool, optional
            Agents receive the health of all units (in the sight range) as part
            of observations (default is True).
        obs_own_health : bool, optional
            Agents receive their own health as a part of observations (default
            is False). This flag is ignored when obs_all_health == True.
        obs_last_action : bool, optional
            Agents receive the last actions of all units (in the sight range)
            as part of observations (default is False).
        obs_pathing_grid : bool, optional
            Whether observations include pathing values surrounding the agent
            (default is False).
        obs_terrain_height : bool, optional
            Whether observations include terrain height values surrounding the
            agent (default is False).
        obs_instead_of_state : bool, optional
            Use combination of all agents' observations as the global state
            (default is False).
        obs_timestep_number : bool, optional
            Whether observations include the current timestep of the episode
            (default is False).
        state_last_action : bool, optional
            Include the last actions of all agents as part of the global state
            (default is True).
        state_timestep_number : bool, optional
            Whether the state include the current timestep of the episode
            (default is False).
        reward_sparse : bool, optional
            Receive 1/-1 reward for winning/loosing an episode (default is
            False). Whe rest of reward parameters are ignored if True.
        reward_only_positive : bool, optional
            Reward is always positive (default is True).
        reward_death_value : float, optional
            The amount of reward received for killing an enemy unit (default
            is 10). This is also the negative penalty for having an allied unit
            killed if reward_only_positive == False.
        reward_win : float, optional
            The reward for winning in an episode (default is 200).
        reward_defeat : float, optional
            The reward for loosing in an episode (default is 0). This value
            should be nonpositive.
        reward_negative_scale : float, optional
            Scaling factor for negative rewards (default is 0.5). This
            parameter is ignored when reward_only_positive == True.
        reward_scale : bool, optional
            Whether or not to scale the reward (default is True).
        reward_scale_rate : float, optional
            Reward scale rate (default is 20). When reward_scale == True, the
            reward received by the agents is divided by (max_reward /
            reward_scale_rate), where max_reward is the maximum possible
            reward per episode without considering the shield regeneration
            of Protoss units.
        replay_dir : str, optional
            The directory to save replays (default is None). If None, the
            replay will be saved in Replays directory where StarCraft II is
            installed.
        replay_prefix : str, optional
            The prefix of the replay to be saved (default is None). If None,
            the name of the map will be used.
        window_size_x : int, optional
            The length of StarCraft II window size (default is 1920).
        window_size_y: int, optional
            The height of StarCraft II window size (default is 1200).
        heuristic_ai: bool, optional
            Whether or not to use a non-learning heuristic AI (default False).
        heuristic_rest: bool, optional
            At any moment, restrict the actions of the heuristic AI to be
            chosen from actions available to RL agents (default is False).
            Ignored if heuristic_ai == False.
        debug: bool, optional
            Log messages about observations, state, actions and rewards for
            debugging purposes (default is False).
            :param reward_reachPoint:
            :param reward_reachArea:
        """
        # Hierarchical Design

        self.hierarchical = hierarchical
        self.map_sps = map_sps
        self.train_on_intervals = train_on_intervals
        self.platoons_move_record = None
        self.target_SP = None
        self.target_SP_loc = None
        self.target_SP_id = None
        self.platoons_move_record = None
        self.init_platoon_SPs = None
        if map_name in ['4t_vs_4t_SP01', '4t_vs_4t_SP12', '4t_vs_0t_8SPs_randomized']:
            self.n_sp = 1  # The number of strategic points that the agents need to reach.
        if map_name in ['4t_vs_0t_8SPs', '4t_vs_0t_8SPs_RandomEnemy', '4t_vs_0t_8SPs_RandomEnemy_075', '12t_vs_4t_8SPs_1QMIX']:
            self.n_sp = 8  # The number of strategic points that the agents need to reach.
        if map_name in ['4t_vs_4t_3paths_random_move']:
            self.n_sp = 13
            
        self.tank_maps = ['Sandbox-8t', '8t', '8t-s11a7', 'Sandbox-4t-waypoint', "4t",
                          "1p_no_enemy_flat", "1p_no_enemy", '12t_1enemy_flat',
                          '4t_vs_1t_11SPs', '4t_vs_4t_11SPs',
                          '4t_vs_4t_SP01', '4t_vs_4t_SP12', '4t_vs_4t_SP23',
                          '4t_vs_4t_SP34', '4t_vs_4t_SP45', '4t_vs_4t_SP56',
                          '4t_vs_4t_SP67', '4t_vs_4t_SP78', '4t_vs_4t_SP89',
                          '4t_vs_4t_SP910', '4t_vs_4t_SP1011',
                          '12t_4enemy', '12t_1t_demo', '12t_1t_demo_simplified',
                          '4t_vs_4t_7SPs', '4t_vs_4t_8SPs', '4t_vs_4t_8SPs_weakened',
                          '4t_vs_0t_8SPs', '4t_vs_0t_8SPs_randomized',
                          '4t_vs_0t_8SPs_RandomEnemy', '4t_vs_0t_8SPs_RandomEnemy_075', '12t_vs_4t_8SPs_1QMIX', '4t_vs_4t_3paths_random_move']
        self.reward_SP = 20
        self.reward_arrive = 50
        self.FALCON_demo = FALCON_demo

        # add the distance as part of the observation
        self.obs_distance_target = obs_distance_target
        self.n_obs_distance_target = 1

        # Map arguments
        # The map_name could be the input from prompt terminal. smac will take this input and return the setting.
        self.map_name = map_name
        map_params = get_map_params(self.map_name)

        self.n_agents = map_params["n_agents"]
        # TODO add the numbr of agents in a team
        # self.n_agent_in_team

        self.n_enemies = map_params["n_enemies"]
        self.episode_limit = map_params["limit"]
        self._move_amount = move_amount
        self._step_mul = step_mul
        self.difficulty = difficulty

        # Observations and state
        self.obs_own_health = obs_own_health
        self.obs_all_health = obs_all_health
        self.obs_instead_of_state = obs_instead_of_state
        self.obs_last_action = obs_last_action
        self.obs_pathing_grid = obs_pathing_grid
        self.obs_terrain_height = obs_terrain_height
        # add the observation of the command direction
        self.obs_direction_command = obs_direction_command

        self.obs_timestep_number = obs_timestep_number
        self.state_last_action = state_last_action
        self.state_timestep_number = state_timestep_number
        if self.obs_all_health:
            self.obs_own_health = True
        self.n_obs_pathing = 8
        self.n_obs_height = 9
        # for one strategic point scenario, the shape of direction is 4, one hot encoding of 4 directions
        self.n_obs_direction_command = 4

        # Rewards args
        self.reward_sparse = reward_sparse
        self.reward_only_positive = reward_only_positive
        self.reward_negative_scale = reward_negative_scale
        self.reward_death_value = reward_death_value
        self.reward_win = reward_win
        self.reward_defeat = reward_defeat
        self.reward_scale = reward_scale
        self.reward_scale_rate = reward_scale_rate
        # ---------------------------------
        # Add rewards for moving to strategic points.
        # self.reward_StrategicPoint_val = 20
        # self.reward_StrategicPoint_loc = [39, 41]
        # [OLD] reward when arrive certain locations, each agents will gain 5 points if it reaches a area given
        # reward_reachArea = 5,
        # reachArea_effect_range = 4,  # radius is 4 by default.
        # areaCenter = [[20, 23], [34, 34], [44, 48], [58, 58]],
        # --------------------------------
        # self.visitRecord = np.zeros((self.n_agents, 4))
        # --------------------------------

        # Other
        self.game_version = game_version
        self.continuing_episode = continuing_episode
        self._seed = seed
        self.heuristic_ai = heuristic_ai
        self.heuristic_rest = heuristic_rest
        self.debug = debug
        self.window_size = (window_size_x, window_size_y)
        self.replay_dir = replay_dir
        self.replay_prefix = replay_prefix
        self.alpha = alpha

        # Actions
        self.n_actions_no_attack = 6
        self.n_actions_move = 4
        self.n_actions = self.n_actions_no_attack + self.n_enemies

        # Map info
        '''
        TODO:  Find out how the races are decided.  
        '''
        self._agent_race = map_params["a_race"]
        self._bot_race = map_params["b_race"]
        self.shield_bits_ally = 1 if self._agent_race == "P" else 0
        self.shield_bits_enemy = 1 if self._bot_race == "P" else 0
        self.unit_type_bits = map_params["unit_type_bits"]
        self.map_type = map_params["map_type"]

        ''' 
        origin max_reward
            self.max_reward = (
               self.n_enemies * self.reward_death_value + self.reward_win
            )
        
        Before 2021-08-14, this formula is used:
            self.max_reward = (
                self.n_enemies * self.reward_death_value + self.reward_win + self.n_agents * self.reward_StrategicPoint_val
            )
            
        The max_reward is modified on 2021-08-14 with the formula below. 
        More updates could be found in Obsidian notes.
            self.max_reward = (self.n_enemies * self.reward_death_value)
        '''
        self.max_reward = (self.n_enemies * self.reward_death_value)

        self.initial_pos = []  # add initial position

        # 2021-09-21
        # set multiple platoons for hierarchical control
        self.hierarchical = hierarchical
        self.n_ally_platoons = n_ally_platoons  # 3 agent platoons
        self.n_ally_agent_in_platoon = n_ally_agent_in_platoon  # Each platoon has 4 agents
        # Enemy has only 1 platoon
        self.n_enemy_platoons = n_enemy_platoons
        self.n_enemy_unit_in_platoon = n_enemy_unit_in_platoon
        if self.hierarchical:
            # self.agent_reach_point = [[0] * self.n_ally_agent_in_platoon] * self.n_ally_platoons
            self.agents_reach_point = None
        else:
            self.agent_reach_point = [0 for _ in range(self.n_agents)]
        # -------------------------------------------
        self.agents = {}
        self.enemies = {}
        self._episode_count = 0
        self._episode_steps = 0
        self._total_steps = 0
        self._obs = None
        self.battles_won = 0
        self.battles_game = 0
        self.timeouts = 0
        self.force_restarts = 0
        self.last_stats = None
        self.death_tracker_ally = np.zeros(self.n_agents)
        self.death_tracker_enemy = np.zeros(self.n_enemies)
        self.previous_ally_units = None
        self.previous_enemy_units = None
        self.last_action = np.zeros((self.n_agents, self.n_actions))
        self._min_unit_type = 0
        self.marine_id = self.marauder_id = self.medivac_id = 0
        self.hydralisk_id = self.zergling_id = self.baneling_id = 0
        self.stalker_id = self.colossus_id = self.zealot_id = 0
        self.max_distance_x = 0
        self.max_distance_y = 0
        self.map_x = 0
        self.map_y = 0
        self.terrain_height = None
        self.pathing_grid = None
        self._run_config = None
        self._sc2_proc = None
        self._controller = None

        # Try to avoid leaking SC2 processes on shutdown
        atexit.register(lambda: self.close())

        # define logging format
        start_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        if self.debug:
            logging.basicConfig(filename="StarCraft II Training" + self.map_name + start_time + ".log", filemode="w",
                                format="%(asctime)s %(name)s:%(levelname)s:%(message)s", datefmt="%d-%m-%Y %H:%M:%S",
                                level=logging.DEBUG)

    def _launch(self):
        """Launch the StarCraft II game."""
        # TODO the meaning of this line is unclear. Update: No need to look into details of this line.
        # Guess: Set configuration?
        self._run_config = run_configs.get(version=self.game_version)
        _map = maps.get(self.map_name)  # use the input map name from the folder that contains maps.

        # Setting up the interface
        # Default: interface_options = sc_pb.InterfaceOptions(raw=True, score=False)  # create a black windows. Now the game hasn't start.
        interface_options = sc_pb.InterfaceOptions(raw=True,
                                                   score=True)  # create a black windows. Now the game hasn't start.
        self._sc2_proc = self._run_config.start(window_size=self.window_size, want_rgb=False)  # start the environment
        self._controller = self._sc2_proc.controller

        # Request to create the game
        # create is a Class
        create = sc_pb.RequestCreateGame(
            local_map=sc_pb.LocalMap(
                map_path=_map.path,
                map_data=self._run_config.map_data(_map.path)),
            realtime=False,
            random_seed=self._seed)
        create.player_setup.add(type=sc_pb.Participant)
        create.player_setup.add(type=sc_pb.Computer, race=races[self._bot_race],
                                difficulty=difficulties["7"])  # self.difficulty

        # create the game
        self._controller.create_game(create)

        join = sc_pb.RequestJoinGame(race=races[self._agent_race],
                                     options=interface_options)
        self._controller.join_game(join)  # join the game. now 2 teams are in the map, but frozen.

        game_info = self._controller.game_info()
        map_info = game_info.start_raw
        map_play_area_min = map_info.playable_area.p0
        map_play_area_max = map_info.playable_area.p1
        self.max_distance_x = map_play_area_max.x - map_play_area_min.x
        self.max_distance_y = map_play_area_max.y - map_play_area_min.y
        self.map_x = map_info.map_size.x
        self.map_y = map_info.map_size.y

        if map_info.pathing_grid.bits_per_pixel == 1:
            vals = np.array(list(map_info.pathing_grid.data)).reshape(
                self.map_x, int(self.map_y / 8))
            self.pathing_grid = np.transpose(np.array([
                [(b >> i) & 1 for b in row for i in range(7, -1, -1)]
                for row in vals], dtype=np.bool))
        else:
            self.pathing_grid = np.invert(np.flip(np.transpose(np.array(
                list(map_info.pathing_grid.data), dtype=np.bool).reshape(
                self.map_x, self.map_y)), axis=1))

        self.terrain_height = np.flip(
            np.transpose(np.array(list(map_info.terrain_height.data))
                         .reshape(self.map_x, self.map_y)), 1) / 255

    def reset(self):
        """
        Reset the environment. Required after each full episode.
        Returns initial observations and states.
        """
        self._episode_steps = 0
        if self._episode_count == 0:
            # Launch StarCraft II
            self._launch()
        else:
            self._restart()

        # Information kept for counting the reward
        # self.death_tracker_ally = np.zeros(self.n_agents)
        # TODO: Check if this death_tracker_ally works
        if self.hierarchical:
            self.death_tracker_ally = np.zeros((self.n_ally_platoons, self.n_ally_agent_in_platoon))
            # TODO: 10-19 should we organize the enemies into platoons?
            # self.death_tracker_enemy = np.zeros((self.n_enemy_platoons, self.n_enemy_unit_in_platoon))

        self.death_tracker_enemy = np.zeros(self.n_enemies)
        self.previous_ally_units = None
        self.previous_enemy_units = None
        self.win_counted = False
        self.defeat_counted = False

        # ------- refresh the visitRecord--------------
        # self.visitRecord = np.zeros((self.n_agents, 4))
        # ---------------------------------------------

        # TODO: last_action is optional to be included in the state.
        # self.last_action = np.zeros((self.n_agents, self.n_actions))

        # In hierarchical model, the last action record the action of 3 platoon, 12 agents in total.
        if self.hierarchical:
            self.last_action = np.zeros((self.n_ally_platoons, self.n_ally_agent_in_platoon, self.n_actions))
        if self.heuristic_ai:
            self.heuristic_targets = [None] * self.n_agents
        try:
            self._obs = self._controller.observe()

            # TODO init platoons and company
            if self.hierarchical:
                self.init_platoons()
            else:
                self.init_units()  # initialize units (both agents and enemies)
        except (protocol.ProtocolError, protocol.ConnectionError):
            self.full_restart()

        if self.debug:
            logging.debug("Started Episode {}"
                          .format(self._episode_count).center(60, "*"))

        # return self.get_obs(), self.get_state()
        if self.hierarchical:
            return self.get_obs_company()
        else:
            return self.get_obs(), self.get_state()

    def _restart(self):
        """Restart the environment by killing all units on the map.
        There is a trigger in the SC2Map file, which restarts the
        episode when there are no units left.
        """
        try:
            self._kill_all_units()
            self._controller.step(2)
        except (protocol.ProtocolError, protocol.ConnectionError):
            self.full_restart()

    def full_restart(self):
        """Full restart. Closes the SC2 process and launches a new one. """
        self._sc2_proc.close()
        self._launch()
        self.force_restarts += 1

    def step(self, actions):
        """A single environment step. Returns reward, terminated, info."""
        # new code
        company_actions_int = []
        # transform the platoon actions into integer
        for platoon_actions in actions:
            platoon_actions_int = [int(a) for a in platoon_actions]
            company_actions_int.append(platoon_actions_int)

        # store the last action in the environment
        self.last_action = np.eye(self.n_actions)[np.array(platoon_actions_int)]

        # Collect individual actions
        company_sc_actions = []
        if self.debug:
            logging.debug("Actions".center(60, "-"))

        for platoon_id in range(len(company_actions_int)):
            platoon_sc_actions = []
            for a_id, action in enumerate(company_actions_int[platoon_id]):
                if not self.heuristic_ai:
                    # platoon_sc_action = self.get_agent_action(a_id, action)
                    platoon_sc_action = self.get_platoon_agent_action(a_id, platoon_id, action)
                else:
                    platoon_sc_action, action_num = self.get_agent_action_heuristic(
                        a_id, action)
                    actions[a_id] = action_num
                if platoon_sc_action:
                    platoon_sc_actions.append(platoon_sc_action)
            company_sc_actions.append(platoon_sc_actions)

        # flatten the platoon_sc_actions
        flat_company_sc_actions = [action for platoon_sc_actions in company_sc_actions for action in platoon_sc_actions]

        # Send action request
        req_actions = sc_pb.RequestAction(actions=flat_company_sc_actions)
        try:
            self._controller.actions(req_actions)
            # Make step in SC2, i.e. apply actions
            self._controller.step(self._step_mul)
            # Observe here so that we know if the episode is over.
            self._obs = self._controller.observe()
        except (protocol.ProtocolError, protocol.ConnectionError):
            self.full_restart()
            return 0, True, {}

        # # -------------------------------------------------------------------------------------------
        # # old code
        # actions_int = [int(a) for a in actions]
        # # TODO separate this last_action for 3 platoons.
        # self.last_action = np.eye(self.n_actions)[np.array(actions_int)]
        #
        # # Collect individual actions
        # sc_actions = []
        # if self.debug:
        #     logging.debug("Actions".center(60, "-"))
        #
        # for a_id, action in enumerate(actions_int):
        #     if not self.heuristic_ai:
        #         sc_action = self.get_agent_action(a_id, action)
        #     else:
        #         sc_action, action_num = self.get_agent_action_heuristic(
        #             a_id, action)
        #         actions[a_id] = action_num
        #     if sc_action:
        #         sc_actions.append(sc_action)
        #
        # # Send action request
        # req_actions = sc_pb.RequestAction(actions=sc_actions)
        # try:
        #     self._controller.actions(req_actions)
        #     # Make step in SC2, i.e. apply actions
        #     self._controller.step(self._step_mul)
        #     # Observe here so that we know if the episode is over.
        #     self._obs = self._controller.observe()
        # except (protocol.ProtocolError, protocol.ConnectionError):
        #     self.full_restart()
        #     return 0, True, {}

        # count the steps
        self._total_steps += 1
        self._episode_steps += 1

        # Update units. -1 = lost , 1 = win, 0 = equal
        game_end_code = self.update_units()  # -1 means lost , 1 means win, 0 means equal

        terminated = False

        # calculate the reward
        # reward = self.reward_battle()  # the reward will ge given after the agents take a set of actions
        # reward_platoons = self.reward_battle_platoons()
        reward = 0  # fake reward

        info = {"battle_won": False}  # 为什么不运行这一行？

        # count units that are still alive
        dead_allies, dead_enemies = 0, 0
        for al_id, al_unit in self.agents.items():
            if al_unit.health == 0:
                dead_allies += 1
        for e_id, e_unit in self.enemies.items():
            if e_unit.health == 0:
                dead_enemies += 1

        info['dead_allies'] = dead_allies
        info['dead_enemies'] = dead_enemies

        if game_end_code is not None:
            # Battle is over
            terminated = True
            self.battles_game += 1
            if game_end_code == 1 and not self.win_counted:
                self.battles_won += 1
                self.win_counted = True
                info["battle_won"] = True
                if not self.reward_sparse:
                    reward += self.reward_win
                else:
                    reward = 1
            elif game_end_code == -1 and not self.defeat_counted:
                self.defeat_counted = True
                if not self.reward_sparse:
                    reward += self.reward_defeat
                else:
                    reward = -1

        elif self._episode_steps >= self.episode_limit:
            # Episode limit reached
            terminated = True
            if self.continuing_episode:
                info["episode_limit"] = True
            self.battles_game += 1
            self.timeouts += 1

        if self.debug:
            logging.debug("Reward = {}".format(reward).center(60, '-'))

        if terminated:
            self._episode_count += 1

        # add the win/loss information in the log file.
        # if game_end_code is not None:
        #    if game_end_code == 1:
        #        result = 'win'
        #    elif game_end_code == -1:
        #        result = 'loss'
        #    if self.debug:
        #        logging.debug("Result = {}".format(result).center(60, '-'))

        if self.reward_scale:
            reward /= self.max_reward / self.reward_scale_rate

        return reward, terminated, info

    def step_company(self, actions):
        """
        A single environment step. Returns reward, terminated, info.
        """
        company_actions_int = []

        for platoon_actions in actions:  # transform the platoon actions into integer
            platoon_actions_int = [int(a) for a in platoon_actions]
            company_actions_int.append(platoon_actions_int)

        # replace last action with the current actions
        self.last_action = np.array([
            np.eye(self.n_actions)[np.array(platoon_actions_int)] for platoon_actions_int in company_actions_int
        ])

        # Collect individual actions
        company_sc_actions = []
        if self.debug:
            logging.debug("Actions".center(60, "-"))
        for platoon_id in range(len(company_actions_int)):
            platoon_sc_actions = []
            for a_id, action in enumerate(company_actions_int[platoon_id]):
                if not self.heuristic_ai:
                    # platoon_sc_action = self.get_agent_action(a_id, action)
                    platoon_sc_action = self.get_platoon_agent_action(a_id, platoon_id, action)
                else:
                    platoon_sc_action, action_num = self.get_agent_action_heuristic(
                        a_id, action)
                    actions[a_id] = action_num
                if platoon_sc_action:
                    platoon_sc_actions.append(platoon_sc_action)
            company_sc_actions.append(platoon_sc_actions)

        # flatten the platoon_sc_actions
        flat_company_sc_actions = [action for platoon_sc_actions in company_sc_actions for action in platoon_sc_actions]

        # Send action request
        req_actions = sc_pb.RequestAction(actions=flat_company_sc_actions)
        try:
            self._controller.actions(req_actions)
            # Make step in SC2, i.e. apply actions
            self._controller.step(self._step_mul)
            # Observe here so that we know if the episode is over.
            self._obs = self._controller.observe()
        except (protocol.ProtocolError, protocol.ConnectionError):
            self.full_restart()
            # return 0, True, {}
            return [0] * self.n_ally_platoons, True, {}

        # # -------------------------------------------------------------------------------------------
        # # old code
        # actions_int = [int(a) for a in actions]

        # self.last_action = np.eye(self.n_actions)[np.array(actions_int)]
        #
        # # Collect individual actions
        # sc_actions = []
        # if self.debug:
        #     logging.debug("Actions".center(60, "-"))
        #
        # for a_id, action in enumerate(actions_int):
        #     if not self.heuristic_ai:
        #         sc_action = self.get_agent_action(a_id, action)
        #     else:
        #         sc_action, action_num = self.get_agent_action_heuristic(
        #             a_id, action)
        #         actions[a_id] = action_num
        #     if sc_action:
        #         sc_actions.append(sc_action)
        #
        # # Send action request
        # req_actions = sc_pb.RequestAction(actions=sc_actions)
        # try:
        #     self._controller.actions(req_actions)
        #     # Make step in SC2, i.e. apply actions
        #     self._controller.step(self._step_mul)
        #     # Observe here so that we know if the episode is over.
        #     self._obs = self._controller.observe()
        # except (protocol.ProtocolError, protocol.ConnectionError):
        #     self.full_restart()
        #     return 0, True, {}

        # count the steps
        self._total_steps += 1
        self._episode_steps += 1

        # Update units. -1 = lost , 1 = win, 0 = equal
        # game_end_code = self.update_units()  # -1 means lost , 1 means win, 0 means equal
        game_end_code, arrive_reward = self.update_platoons()

        terminated = False

        # reward = self.reward_battle()  # the reward will ge given after the agents take a set of actions

        # TODO: Update the rewards for each platoons
        reward_platoons = self.reward_battle_company()  # a list of rewards

        '''
        Update the reward of arrive the strategic point.
        Every unit in the platoon will get a reward of 50 when arrives the strategic point.
        '''
        for pid in range(self.n_ally_platoons):
            # reward_platoons[pid] += arrive_reward[pid] * 50
            reward_platoons[pid] += (1 - self.alpha) * arrive_reward[pid] * 50  # update the gained reward with alpha

        info = {"battle_won": False}  # 为什么不运行这一行？

        # count units that are still alive
        dead_allies, dead_enemies = 0, 0

        if self.hierarchical:  # update agents whether alive
            for platoon in self.ally_platoons:
                for al_id, al_unit in enumerate(platoon):
                    if al_unit.health == 0:
                        dead_allies += 1
        else:
            for al_id, al_unit in self.agents.items():
                if al_unit.health == 0:
                    dead_allies += 1

        for e_id, e_unit in self.enemies.items():  # update ememies
            if e_unit.health == 0:
                dead_enemies += 1

        info['dead_allies'] = dead_allies
        info['dead_enemies'] = dead_enemies

        if game_end_code is not None:
            terminated = True
            self.battles_game += 1
            if game_end_code == 1 and not self.win_counted:
                self.battles_won += 1
                self.win_counted = True
                info["battle_won"] = True
                if not self.reward_sparse:
                    # reward += self.reward_win
                    reward_platoons = [r + self.reward_win for r in reward_platoons]
                    # reward_platoons += self.reward_win
                else:
                    # reward = 1
                    reward_platoons = 1
            elif game_end_code == -1 and not self.defeat_counted:
                self.defeat_counted = True
                if not self.reward_sparse:
                    # reward += self.reward_defeat
                    # reward_platoons += self.reward_defeat
                    reward_platoons = [r + self.reward_defeat for r in reward_platoons]
                else:
                    # reward = -1
                    reward_platoons = -1
        elif self._episode_steps >= self.episode_limit:
            # Episode limit reached
            terminated = True
            if self.continuing_episode:
                info["episode_limit"] = True
            self.battles_game += 1
            self.timeouts += 1

        if self.debug:
            # logging.debug("Reward = {}".format(reward).center(60, '-'))
            logging.debug("Reward = {}".format(reward_platoons).center(60, '-'))

        if terminated:
            self._episode_count += 1

        # add the win/loss information in the log file.
        # if game_end_code is not None:
        #    if game_end_code == 1:
        #        result = 'win'
        #    elif game_end_code == -1:
        #        result = 'loss'
        #    if self.debug:
        #        logging.debug("Result = {}".format(result).center(60, '-'))

        if self.reward_scale:
            # reward /= self.max_reward / self.reward_scale_rate
            reward_platoons = [r / (self.max_reward / self.reward_scale_rate) for r in reward_platoons]
            # reward_platoons /= self.max_reward / self.reward_scale_rate

        return reward_platoons, terminated, info

    def get_agent_action(self, a_id, action):
        """Construct the action for agent a_id."""
        avail_actions = self.get_avail_agent_actions(a_id)
        assert avail_actions[action] == 1, \
            "Agent {} cannot perform action {}".format(a_id, action)

        unit = self.get_unit_by_id(a_id)
        tag = unit.tag
        x = unit.pos.x
        y = unit.pos.y

        if action == 0:
            # no-op (valid only when dead)
            assert unit.health == 0, "No-op only available for dead agents."
            if self.debug:
                logging.debug("Agent {}: Dead".format(a_id))
            return None
        elif action == 1:
            # stop
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["stop"],
                unit_tags=[tag],
                queue_command=False)
            if self.debug:
                logging.debug("Agent {}: Stop".format(a_id))

        elif action == 2:
            # move north
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["move"],
                target_world_space_pos=sc_common.Point2D(
                    x=x, y=y + self._move_amount),
                unit_tags=[tag],
                queue_command=False)
            if self.debug:
                logging.debug("Agent {}: Move North".format(a_id))

        elif action == 3:
            # move south
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["move"],
                target_world_space_pos=sc_common.Point2D(
                    x=x, y=y - self._move_amount),
                unit_tags=[tag],
                queue_command=False)
            if self.debug:
                logging.debug("Agent {}: Move South".format(a_id))

        elif action == 4:
            # move east
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["move"],
                target_world_space_pos=sc_common.Point2D(
                    x=x + self._move_amount, y=y),
                unit_tags=[tag],
                queue_command=False)
            if self.debug:
                logging.debug("Agent {}: Move East".format(a_id))

        elif action == 5:
            # move west
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["move"],
                target_world_space_pos=sc_common.Point2D(
                    x=x - self._move_amount, y=y),
                unit_tags=[tag],
                queue_command=False)
            if self.debug:
                logging.debug("Agent {}: Move West".format(a_id))
        else:
            # attack/heal units that are in range
            target_id = action - self.n_actions_no_attack
            if self.map_type == "MMM" and unit.unit_type == self.medivac_id:
                target_unit = self.agents[target_id]
                action_name = "heal"
            else:
                target_unit = self.enemies[target_id]
                action_name = "attack"

            action_id = actions[action_name]
            target_tag = target_unit.tag

            cmd = r_pb.ActionRawUnitCommand(
                ability_id=action_id,
                target_unit_tag=target_tag,
                unit_tags=[tag],
                queue_command=False)

            if self.debug:
                logging.debug("Agent {} {}s unit # {}".format(
                    a_id, action_name, target_id))

        sc_action = sc_pb.Action(action_raw=r_pb.ActionRaw(unit_command=cmd))
        return sc_action

    def get_platoon_agent_action(self, a_id, p_id, action):
        """Construct the action for agent a_id."""
        avail_actions = self.get_avail_platoon_agent_actions(a_id, p_id)
        assert avail_actions[action] == 1, \
            "Agent {} cannot perform action {}".format(a_id, action)

        unit = self.get_platoon_unit_by_id(a_id, p_id)
        tag = unit.tag
        x = unit.pos.x
        y = unit.pos.y

        if action == 0:
            # no-op (valid only when dead)
            assert unit.health == 0, "No-op only available for dead agents."
            if self.debug:
                logging.debug("Agent {}: Dead".format(a_id))
            return None
        elif action == 1:
            # stop
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["stop"],
                unit_tags=[tag],
                queue_command=False)
            if self.debug:
                logging.debug("Agent {}: Stop".format(a_id))

        elif action == 2:
            # move north
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["move"],
                target_world_space_pos=sc_common.Point2D(
                    x=x, y=y + self._move_amount),
                unit_tags=[tag],
                queue_command=False)
            if self.debug:
                logging.debug("Agent {}: Move North".format(a_id))

        elif action == 3:
            # move south
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["move"],
                target_world_space_pos=sc_common.Point2D(
                    x=x, y=y - self._move_amount),
                unit_tags=[tag],
                queue_command=False)
            if self.debug:
                logging.debug("Agent {}: Move South".format(a_id))

        elif action == 4:
            # move east
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["move"],
                target_world_space_pos=sc_common.Point2D(
                    x=x + self._move_amount, y=y),
                unit_tags=[tag],
                queue_command=False)
            if self.debug:
                logging.debug("Agent {}: Move East".format(a_id))

        elif action == 5:
            # move west
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["move"],
                target_world_space_pos=sc_common.Point2D(
                    x=x - self._move_amount, y=y),
                unit_tags=[tag],
                queue_command=False)
            if self.debug:
                logging.debug("Agent {}: Move West".format(a_id))
        else:
            # attack/heal units that are in range
            target_id = action - self.n_actions_no_attack
            if self.map_type == "MMM" and unit.unit_type == self.medivac_id:
                target_unit = self.agents[target_id]
                action_name = "heal"
            else:
                # target unit need to be set to the enemy in enemy platoon
                target_unit = self.enemies[target_id]
                action_name = "attack"

            action_id = actions[action_name]
            target_tag = target_unit.tag

            cmd = r_pb.ActionRawUnitCommand(
                ability_id=action_id,
                target_unit_tag=target_tag,
                unit_tags=[tag],
                queue_command=False)

            if self.debug:
                logging.debug("Agent {} {}s unit # {}".format(
                    a_id, action_name, target_id))

        sc_action = sc_pb.Action(action_raw=r_pb.ActionRaw(unit_command=cmd))
        return sc_action

    def get_agent_action_heuristic(self, a_id, action):
        unit = self.get_unit_by_id(a_id)
        tag = unit.tag

        target = self.heuristic_targets[a_id]
        if unit.unit_type == self.medivac_id:
            if (target is None or self.agents[target].health == 0 or
                    self.agents[target].health == self.agents[target].health_max):
                min_dist = math.hypot(self.max_distance_x, self.max_distance_y)
                min_id = -1
                for al_id, al_unit in self.agents.items():
                    if al_unit.unit_type == self.medivac_id:
                        continue
                    if (al_unit.health != 0 and
                            al_unit.health != al_unit.health_max):
                        dist = self.distance(unit.pos.x, unit.pos.y,
                                             al_unit.pos.x, al_unit.pos.y)
                        if dist < min_dist:
                            min_dist = dist
                            min_id = al_id
                self.heuristic_targets[a_id] = min_id
                if min_id == -1:
                    self.heuristic_targets[a_id] = None
                    return None, 0
            action_id = actions['heal']
            target_tag = self.agents[self.heuristic_targets[a_id]].tag
        else:
            if target is None or self.enemies[target].health == 0:
                min_dist = math.hypot(self.max_distance_x, self.max_distance_y)
                min_id = -1
                for e_id, e_unit in self.enemies.items():
                    if (unit.unit_type == self.marauder_id and
                            e_unit.unit_type == self.medivac_id):
                        continue
                    if e_unit.health > 0:
                        dist = self.distance(unit.pos.x, unit.pos.y,
                                             e_unit.pos.x, e_unit.pos.y)
                        if dist < min_dist:
                            min_dist = dist
                            min_id = e_id
                self.heuristic_targets[a_id] = min_id
                if min_id == -1:
                    self.heuristic_targets[a_id] = None
                    return None, 0
            action_id = actions['attack']
            target_tag = self.enemies[self.heuristic_targets[a_id]].tag

        action_num = self.heuristic_targets[a_id] + self.n_actions_no_attack

        # Check if the action is available
        if (self.heuristic_rest and
                self.get_avail_agent_actions(a_id)[action_num] == 0):

            # Move towards the target rather than attacking/healing
            if unit.unit_type == self.medivac_id:
                target_unit = self.agents[self.heuristic_targets[a_id]]
            else:
                target_unit = self.enemies[self.heuristic_targets[a_id]]

            delta_x = target_unit.pos.x - unit.pos.x
            delta_y = target_unit.pos.y - unit.pos.y

            if abs(delta_x) > abs(delta_y):  # east or west
                if delta_x > 0:  # east
                    target_pos = sc_common.Point2D(
                        x=unit.pos.x + self._move_amount, y=unit.pos.y)
                    action_num = 4
                else:  # west
                    target_pos = sc_common.Point2D(
                        x=unit.pos.x - self._move_amount, y=unit.pos.y)
                    action_num = 5
            else:  # north or south
                if delta_y > 0:  # north
                    target_pos = sc_common.Point2D(
                        x=unit.pos.x, y=unit.pos.y + self._move_amount)
                    action_num = 2
                else:  # south
                    target_pos = sc_common.Point2D(
                        x=unit.pos.x, y=unit.pos.y - self._move_amount)
                    action_num = 3

            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions['move'],
                target_world_space_pos=target_pos,
                unit_tags=[tag],
                queue_command=False)
        else:
            # Attack/heal the target
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=action_id,
                target_unit_tag=target_tag,
                unit_tags=[tag],
                queue_command=False)

        sc_action = sc_pb.Action(action_raw=r_pb.ActionRaw(unit_command=cmd))
        return sc_action, action_num

    def reward_battle(self):
        """Reward function when self.reward_spare==False.
        Returns accumulative hit/shield point damage dealt to the enemy
        + reward_death_value per enemy unit killed, and, in case
        self.reward_only_positive == False, - (damage dealt to ally units
        + reward_death_value per ally unit killed) * self.reward_negative_scale
        """
        if self.reward_sparse:
            return 0

        reward = 0
        delta_deaths = 0
        delta_ally = 0
        delta_enemy = 0
        # reward_reachAreaSum = 0
        reward_ReachStrategicPoints = 0
        neg_scale = self.reward_negative_scale

        # update deaths
        for al_id, al_unit in self.agents.items():
            if not self.death_tracker_ally[al_id]:
                # did not die so far
                prev_health = (
                        self.previous_ally_units[al_id].health
                        + self.previous_ally_units[al_id].shield
                )
                if al_unit.health == 0:
                    # just died
                    self.death_tracker_ally[al_id] = 1
                    if not self.reward_only_positive:
                        # if ally units died, the delta_death will be a negative reward, a punishment.
                        delta_deaths -= self.reward_death_value * neg_scale
                    # delta_ally is the damage dealt to the ally units.
                    delta_ally += prev_health * neg_scale
                else:
                    # still alive
                    delta_ally += neg_scale * (
                            prev_health - al_unit.health - al_unit.shield
                    )

        for e_id, e_unit in self.enemies.items():
            if not self.death_tracker_enemy[e_id]:
                prev_health = (
                        self.previous_enemy_units[e_id].health
                        + self.previous_enemy_units[e_id].shield
                )
                if e_unit.health == 0:
                    self.death_tracker_enemy[e_id] = 1
                    delta_deaths += self.reward_death_value
                    delta_enemy += prev_health
                else:
                    delta_enemy += prev_health - e_unit.health - e_unit.shield

        # r = self.reachArea_effect_range
        # n_area = 4

        # update rewards for locations
        # for al_id, al_unit in self.agents.items():
        #    for i in range(4):
        #        if abs(al_unit.pos.x - self.areaCenter[i][0]) < r and abs(al_unit.pos.y - self.areaCenter[i][1]) < r:
        #            if self.visitRecord[al_id][i] == 0:
        #                self.visitRecord[al_id][i] = 1
        #                reward_reachAreaSum += self.reward_reachArea
        #        else:
        #            continue

        # Add rewards for getting closer to the strategic points.
        # Version 1.0, use the difference between last distance and current position, divided by the last distance.
        # for al_id, al_unit in self.agents.items():
        #    last_distance = math.sqrt((self.previous_ally_units[al_id].pos.x - self.reward_StrategicPoint_loc[0])**2 + (self.previous_ally_units[al_id].pos.y - self.reward_StrategicPoint_loc[1])**2)
        #    current_distance = math.sqrt((al_unit.pos.x - self.reward_StrategicPoint_loc[0])**2 + (al_unit.pos.y - self.reward_StrategicPoint_loc[1])**2)
        #    reward_ReachStrategicPoints += (last_distance-current_distance)/last_distance * self.reward_StrategicPoint_val

        # Version 1.1, use the difference between last distance and current position, divided by the total distance
        for al_id, al_unit in self.agents.items():
            initial_distance = math.sqrt((self.initial_pos[al_id][0] - self.reward_StrategicPoint_loc[0]) ** 2 + (
                    self.initial_pos[al_id][1] - self.reward_StrategicPoint_loc[1]) ** 2)
            last_distance = math.sqrt(
                (self.previous_ally_units[al_id].pos.x - self.reward_StrategicPoint_loc[0]) ** 2 + (
                        self.previous_ally_units[al_id].pos.y - self.reward_StrategicPoint_loc[1]) ** 2)
            current_distance = math.sqrt((al_unit.pos.x - self.reward_StrategicPoint_loc[0]) ** 2 + (
                    al_unit.pos.y - self.reward_StrategicPoint_loc[1]) ** 2)
            reward_ReachStrategicPoints += (
                                                   last_distance - current_distance) / initial_distance * self.reward_StrategicPoint_val
        #    for i in range(4):
        #        if abs(al_unit.pos.x - self.areaCenter[i][0]) < r and abs(al_unit.pos.y - self.areaCenter[i][1]) < r:
        #            if self.visitRecord[al_id][i] == 0:
        #                self.visitRecord[al_id][i] = 1
        #                reward_reachAreaSum += self.reward_reachArea
        #        else:
        #            continue

        if self.reward_only_positive:
            # [OLD] reward = abs(delta_enemy + delta_deaths + reward_reachAreaSum)  # shield regeneration
            # [origin] reward = abs(delta_enemy + delta_deaths)  # shield regeneration
            # [Before 2021-08-14] reward = self.alpha * abs(delta_enemy + delta_deaths) + (1 - self.alpha) * reward_ReachStrategicPoints  # shield regeneration
            # [2021-08-14 Update]
            reward = self.alpha * abs(delta_enemy + delta_deaths) + (
                    1 - self.alpha) * reward_ReachStrategicPoints
        else:
            # [origin] reward = delta_enemy + delta_deaths - delta_ally
            # [OLD] reward = delta_enemy + delta_deaths - delta_ally + reward_reachAreaSum
            # [Before 2021-08-14] reward = self.alpha * (delta_enemy + delta_deaths - delta_ally) + (1-self.alpha) * reward_ReachStrategicPoints
            # [2021-08-14 Update]
            reward = self.alpha * (delta_enemy + delta_deaths - delta_ally) + (
                    1 - self.alpha) * reward_ReachStrategicPoints
        return reward

    # def reward_battle_platoons(self):
    #     return None

    def reward_battle_platoon(self, pid):
        """Reward function when self.reward_spare==False.
        Returns accumulative hit/shield point damage dealt to the enemy
        + reward_death_value per enemy unit killed, and, in case
        self.reward_only_positive == False, - (damage dealt to ally units
        + reward_death_value per ally unit killed) * self.reward_negative_scale
        """
        if self.reward_sparse:
            return 0

        platoon = self.ally_platoons[pid]
        reward_platoon = 0
        # TODO: 如何判断是这个platoon里面的单位击杀了敌方，而不是别的platoon的单位？
        delta_deaths = 0
        delta_ally = 0
        delta_enemy = 0

        reward_approach_sp = 0
        neg_scale = self.reward_negative_scale

        for al_id, al_unit in enumerate(platoon):  # update deaths of ally
            if not self.death_tracker_ally[pid][al_id]:
                prev_health = (
                        self.previous_ally_platoons[pid][al_id].health + self.previous_ally_platoons[pid][al_id].shield
                )
                if al_unit.health == 0:  # just died
                    self.death_tracker_ally[pid][al_id] = 1
                    if not self.reward_only_positive:  # if ally units died, the delta_death will be a negative reward
                        delta_deaths -= self.reward_death_value * neg_scale
                    delta_ally += prev_health * neg_scale  # delta_ally is the damage dealt to the ally units.
                # did not die so far
                # prev_health = (
                #         self.previous_ally_units[al_id].health
                #         + self.previous_ally_units[al_id].shield
                # )
                # if al_unit.health == 0:
                #     # just died
                #     self.death_tracker_ally[pid][al_id] = 1
                #     if not self.reward_only_positive:
                #         # if ally units died, the delta_death will be a negative reward, a punishment.
                #         delta_deaths -= self.reward_death_value * neg_scale
                #     # delta_ally is the damage dealt to the ally units.
                #     delta_ally += prev_health * neg_scale
                else:
                    delta_ally += neg_scale * (
                            prev_health - al_unit.health - al_unit.shield
                    )
        for e_id, e_unit in self.enemies.items():  # update enemy health
            if not self.death_tracker_enemy[e_id]:
                prev_health = (
                        self.previous_enemy_platoons[e_id].health
                        + self.previous_enemy_platoons[e_id].shield
                )
                if e_unit.health == 0:
                    self.death_tracker_enemy[e_id] = 1
                    delta_deaths += self.reward_death_value
                    delta_enemy += prev_health
                else:
                    delta_enemy += prev_health - e_unit.health - e_unit.shield
        # r = self.reachArea_effect_range
        # n_area = 4

        # update rewards for locations
        # for al_id, al_unit in self.agents.items():
        #    for i in range(4):
        #        if abs(al_unit.pos.x - self.areaCenter[i][0]) < r and abs(al_unit.pos.y - self.areaCenter[i][1]) < r:
        #            if self.visitRecord[al_id][i] == 0:
        #                self.visitRecord[al_id][i] = 1
        #                reward_reachAreaSum += self.reward_reachArea
        #        else:
        #            continue

        # Add rewards for getting closer to the strategic points.
        # Version 1.0, use the difference between last distance and current position, divided by the last distance.
        # for al_id, al_unit in self.agents.items():
        #    last_distance = math.sqrt((self.previous_ally_units[al_id].pos.x - self.reward_StrategicPoint_loc[0])**2 + (self.previous_ally_units[al_id].pos.y - self.reward_StrategicPoint_loc[1])**2)
        #    current_distance = math.sqrt((al_unit.pos.x - self.reward_StrategicPoint_loc[0])**2 + (al_unit.pos.y - self.reward_StrategicPoint_loc[1])**2)
        #    reward_ReachStrategicPoints += (last_distance-current_distance)/last_distance * self.reward_StrategicPoint_val

        # Version 1.1, use the difference between last distance and current position, divided by the total distance
        # TODO 10-19 update the rewards for approaching the SPs
        for al_id, al_unit in enumerate(platoon):  # reward of approaching to SP
            initial_distance = math.sqrt(
                (self.initial_pos[pid][al_id][0] - self.target_SP_loc[pid][0]) ** 2 +
                (self.initial_pos[pid][al_id][1] - self.target_SP_loc[pid][1]) ** 2)
            last_distance = math.sqrt(
                (self.previous_ally_platoons[pid][al_id].pos.x - self.target_SP_loc[pid][0]) ** 2 +
                (self.previous_ally_platoons[pid][al_id].pos.y - self.target_SP_loc[pid][1]) ** 2)
            current_distance = math.sqrt(
                (al_unit.pos.x - self.target_SP_loc[pid][0]) ** 2 +
                (al_unit.pos.y - self.target_SP_loc[pid][1]) ** 2)
            distance_diff = last_distance - current_distance
            reward_approach_sp += (distance_diff / initial_distance) * self.reward_SP

        # todo: [11-12] need to include the reward of reach the SP in the function below.
        if self.reward_only_positive:
            reward_platoon = self.alpha * abs(delta_enemy + delta_deaths) + \
                             (1 - self.alpha) * reward_approach_sp
        else:
            reward_platoon = self.alpha * (delta_enemy + delta_deaths - delta_ally) + \
                             (1 - self.alpha) * reward_approach_sp

        return reward_platoon

    def reward_battle_company(self):
        """Reward function when self.reward_spare==False.
        Returns accumulative hit/shield point damage dealt to the enemy
        + reward_death_value per enemy unit killed, and, in case
        self.reward_only_positive == False, - (damage dealt to ally units
        + reward_death_value per ally unit killed) * self.reward_negative_scale
        """
        if self.reward_sparse:
            return 0

        reward_company = [0] * self.n_ally_platoons
        # TODO: 10-19, get the reward for each platoon
        for pid in range(self.n_ally_platoons):
            reward_platoon = self.reward_battle_platoon(pid)
            reward_company[pid] = reward_platoon
        return reward_company

    #         reward = 0
    #         delta_deaths = 0
    #         delta_ally = 0
    #         delta_enemy = 0
    #         # TODO: Check the code of reward_battle_company
    # # <<<<<<< HEAD
    # #         reward_ReachStrategicPoints = 0
    # #
    # #         neg_scale = self.reward_negative_scale
    # #
    # #         # update deaths
    # #         for platoon in self.ally_platoons:
    # #             for al_id, al_unit in platoon.items():
    # #                 if not self.death_tracker_ally[al_id]:
    # #                     prev_health = (
    # #                             self.previous_ally_units[al_id].health
    # #                             + self.previous_ally_units[al_id].shield
    # #                     )
    # #                     if al_unit.health == 0:
    # #                         # just died
    # #                         self.death_tracker_ally[al_id] = 1
    # #                         if not self.reward_only_positive:
    # #                             # if ally units died, the delta_death will be a negative reward, a punishment.
    # #                             delta_deaths -= self.reward_death_value * neg_scale
    # #                         # delta_ally is the damage dealt to the ally units.
    # #                         delta_ally += prev_health * neg_scale
    # #                     else:
    # #                         # still alive
    # #                         delta_ally += neg_scale * (
    # #                                 prev_health - al_unit.health - al_unit.shield
    # #                         )
    # #
    # #         # codes for non-hierarchical design.
    # #         # for al_id, al_unit in self.agents.items():
    # #         #     if not self.death_tracker_ally[al_id]:
    # #         #         # did not die so far
    # #         #         prev_health = (
    # #         #                 self.previous_ally_units[al_id].health
    # #         #                 + self.previous_ally_units[al_id].shield
    # #         #         )
    # #         #         if al_unit.health == 0:
    # #         #             # just died
    # #         #             self.death_tracker_ally[al_id] = 1
    # #         #             if not self.reward_only_positive:
    # #         #                 # if ally units died, the delta_death will be a negative reward, a punishment.
    # #         #                 delta_deaths -= self.reward_death_value * neg_scale
    # #         #             # delta_ally is the damage dealt to the ally units.
    # #         #             delta_ally += prev_health * neg_scale
    # #         #         else:
    # #         #             # still alive
    # #         #             delta_ally += neg_scale * (
    # #         #                     prev_health - al_unit.health - al_unit.shield
    # #         #             )
    # # =======
    #         # reward_reachAreaSum = 0
    #         reward_ReachStrategicPoints = 0
    #         neg_scale = self.reward_negative_scale
    #
    #         # update deaths
    #         for al_id, al_unit in self.agents.items():
    #             if not self.death_tracker_ally[al_id]:
    #                 # did not die so far
    #                 prev_health = (
    #                         self.previous_ally_units[al_id].health
    #                         + self.previous_ally_units[al_id].shield
    #                 )
    #                 if al_unit.health == 0:
    #                     # just died
    #                     self.death_tracker_ally[al_id] = 1
    #                     if not self.reward_only_positive:
    #                         # if ally units died, the delta_death will be a negative reward, a punishment.
    #                         delta_deaths -= self.reward_death_value * neg_scale
    #                     # delta_ally is the damage dealt to the ally units.
    #                     delta_ally += prev_health * neg_scale
    #                 else:
    #                     # still alive
    #                     delta_ally += neg_scale * (
    #                             prev_health - al_unit.health - al_unit.shield
    #                     )
    #
    #         for e_id, e_unit in self.enemies.items():
    #             if not self.death_tracker_enemy[e_id]:
    #                 prev_health = (
    #                         self.previous_enemy_units[e_id].health
    #                         + self.previous_enemy_units[e_id].shield
    #                 )
    #                 if e_unit.health == 0:
    #                     self.death_tracker_enemy[e_id] = 1
    #                     delta_deaths += self.reward_death_value
    #                     delta_enemy += prev_health
    #                 else:
    #                     delta_enemy += prev_health - e_unit.health - e_unit.shield
    #
    #         # r = self.reachArea_effect_range
    #         # n_area = 4
    #
    #         # update rewards for locations
    #         # for al_id, al_unit in self.agents.items():
    #         #    for i in range(4):
    #         #        if abs(al_unit.pos.x - self.areaCenter[i][0]) < r and abs(al_unit.pos.y - self.areaCenter[i][1]) < r:
    #         #            if self.visitRecord[al_id][i] == 0:
    #         #                self.visitRecord[al_id][i] = 1
    #         #                reward_reachAreaSum += self.reward_reachArea
    #         #        else:
    #         #            continue
    #
    #         # Add rewards for getting closer to the strategic points.
    #         # Version 1.0, use the difference between last distance and current position, divided by the last distance.
    #         # for al_id, al_unit in self.agents.items():
    #         #    last_distance = math.sqrt((self.previous_ally_units[al_id].pos.x - self.reward_StrategicPoint_loc[0])**2 + (self.previous_ally_units[al_id].pos.y - self.reward_StrategicPoint_loc[1])**2)
    #         #    current_distance = math.sqrt((al_unit.pos.x - self.reward_StrategicPoint_loc[0])**2 + (al_unit.pos.y - self.reward_StrategicPoint_loc[1])**2)
    #         #    reward_ReachStrategicPoints += (last_distance-current_distance)/last_distance * self.reward_StrategicPoint_val
    #
    #         # Version 1.1, use the difference between last distance and current position, divided by the total distance
    #         for al_id, al_unit in self.agents.items():
    #             initial_distance = math.sqrt((self.initial_pos[al_id][0] - self.reward_StrategicPoint_loc[0]) ** 2 + (
    #                     self.initial_pos[al_id][1] - self.reward_StrategicPoint_loc[1]) ** 2)
    #             last_distance = math.sqrt(
    #                 (self.previous_ally_units[al_id].pos.x - self.reward_StrategicPoint_loc[0]) ** 2 + (
    #                         self.previous_ally_units[al_id].pos.y - self.reward_StrategicPoint_loc[1]) ** 2)
    #             current_distance = math.sqrt((al_unit.pos.x - self.reward_StrategicPoint_loc[0]) ** 2 + (
    #                     al_unit.pos.y - self.reward_StrategicPoint_loc[1]) ** 2)
    #             reward_ReachStrategicPoints += (
    #                                                    last_distance - current_distance) / initial_distance * self.reward_StrategicPoint_val
    #         #    for i in range(4):
    #         #        if abs(al_unit.pos.x - self.areaCenter[i][0]) < r and abs(al_unit.pos.y - self.areaCenter[i][1]) < r:
    #         #            if self.visitRecord[al_id][i] == 0:
    #         #                self.visitRecord[al_id][i] = 1
    #         #                reward_reachAreaSum += self.reward_reachArea
    #         #        else:
    #         #            continue
    #
    #         if self.reward_only_positive:
    #             # [OLD] reward = abs(delta_enemy + delta_deaths + reward_reachAreaSum)  # shield regeneration
    #             # [origin] reward = abs(delta_enemy + delta_deaths)  # shield regeneration
    #             # [Before 2021-08-14] reward = self.alpha * abs(delta_enemy + delta_deaths) + (1 - self.alpha) * reward_ReachStrategicPoints  # shield regeneration
    #             # [2021-08-14 Update]
    #             reward = self.alpha * abs(delta_enemy + delta_deaths) + (
    #                     1 - self.alpha) * reward_ReachStrategicPoints
    #         else:
    #             # [origin] reward = delta_enemy + delta_deaths - delta_ally
    #             # [OLD] reward = delta_enemy + delta_deaths - delta_ally + reward_reachAreaSum
    #             # [Before 2021-08-14] reward = self.alpha * (delta_enemy + delta_deaths - delta_ally) + (1-self.alpha) * reward_ReachStrategicPoints
    #             # [2021-08-14 Update]
    #             reward = self.alpha * (delta_enemy + delta_deaths - delta_ally) + (
    #                     1 - self.alpha) * reward_ReachStrategicPoints
    #         return reward

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take."""
        return self.n_actions

    @staticmethod
    def distance(x1, y1, x2, y2):
        """Distance between two points."""
        return math.hypot(x2 - x1, y2 - y1)

    def platoon_unit_shoot_range(self, agent_id, platoon_id):
        """Returns the shooting range for an agent."""
        if self.map_name in self.tank_maps:
            return 7
        if self.map_name == "3m":
            return 6

    def unit_shoot_range(self, agent_id):
        """Returns the shooting range for an agent."""
        if self.map_name in self.tank_maps:
            return 7
        if self.map_name == "3m":
            return 6

    def unit_sight_range(self, agent_id):
        """Returns the sight range for an agent."""
        if self.map_name in self.tank_maps:
            return 11
        if self.map_name == "3m":
            return 9

    def unit_max_cooldown(self, unit):
        """Returns the maximal cooldown for a unit."""
        switcher = {
            self.marine_id: 15,
            self.marauder_id: 25,
            self.medivac_id: 200,  # max energy
            self.stalker_id: 35,
            self.zealot_id: 22,
            self.colossus_id: 24,
            self.hydralisk_id: 10,
            self.zergling_id: 11,
            self.baneling_id: 1
        }
        return switcher.get(unit.unit_type, 15)

    def save_replay(self):
        """Save a replay."""
        # start_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        prefix = self.replay_prefix or self.map_name
        # replay_dir = self.replay_dir + '/' + self.args.alg + '/' + self.args.map + '/' + start_time or ""
        replay_dir = self.replay_dir or ""
        replay_path = self._run_config.save_replay(
            self._controller.save_replay(), replay_dir=replay_dir, prefix=prefix)
        logging.info("Replay saved at: %s" % replay_path)

    def unit_max_shield(self, unit):
        """Returns maximal shield for a given unit."""
        if unit.unit_type == 74 or unit.unit_type == self.stalker_id:
            return 80  # Protoss's Stalker
        if unit.unit_type == 73 or unit.unit_type == self.zealot_id:
            return 50  # Protoss's Zaelot
        if unit.unit_type == 4 or unit.unit_type == self.colossus_id:
            return 150  # Protoss's Colossus

    def can_move(self, unit, direction):
        """Whether a unit can move in a given direction."""
        m = self._move_amount / 2

        if direction == Direction.NORTH:
            x, y = int(unit.pos.x), int(unit.pos.y + m)
        elif direction == Direction.SOUTH:
            x, y = int(unit.pos.x), int(unit.pos.y - m)
        elif direction == Direction.EAST:
            x, y = int(unit.pos.x + m), int(unit.pos.y)
        else:
            x, y = int(unit.pos.x - m), int(unit.pos.y)

        if self.check_bounds(x, y) and self.pathing_grid[x, y]:
            return True

        return False

    def get_surrounding_points(self, unit, include_self=False):
        """Returns the surrounding points of the unit in 8 directions."""
        x = int(unit.pos.x)
        y = int(unit.pos.y)

        ma = self._move_amount

        points = [
            (x, y + 2 * ma),
            (x, y - 2 * ma),
            (x + 2 * ma, y),
            (x - 2 * ma, y),
            (x + ma, y + ma),
            (x - ma, y - ma),
            (x + ma, y - ma),
            (x - ma, y + ma),
        ]

        if include_self:
            points.append((x, y))

        return points

    def check_bounds(self, x, y):
        """Whether a point is within the map bounds."""
        return (0 <= x < self.map_x and 0 <= y < self.map_y)

    def get_surrounding_pathing(self, unit):
        """Returns pathing values of the grid surrounding the given unit."""
        points = self.get_surrounding_points(unit, include_self=False)
        vals = [
            self.pathing_grid[x, y] if self.check_bounds(x, y) else 1
            for x, y in points
        ]
        return vals

    def get_surrounding_height(self, unit):
        """Returns height values of the grid surrounding the given unit."""
        points = self.get_surrounding_points(unit,
                                             include_self=True)  # if include_self is true, return 9 points, diamond shape
        vals = [
            self.terrain_height[x, y] if self.check_bounds(x, y) else 1
            for x, y in points
        ]
        return vals

    # Update on 2021-08-05: get the direction of command
    # Update on 2022-02-18: add the deviations for each agent. Create individual directions.
    def get_direction_command(self, unit, aid, pid):
        vals = [0, 0, 0, 0]
        alpha = 3
        deviation = [0, 0]  # The deviation is adjusted for each agent.

        if aid == 0:
            deviation = [0, alpha]
        if aid == 1:
            deviation = [0, -alpha]
        if aid == 2:
            deviation = [alpha, 0]
        if aid == 3:
            deviation = [-alpha, 0]

        if unit.pos.x < (self.target_SP_loc[pid][0] + deviation[0]):
            vals[0] = 1  # should go east
        if unit.pos.y > (self.target_SP_loc[pid][1] + deviation[1]):
            vals[1] = 1  # should go south
        if unit.pos.x > (self.target_SP_loc[pid][0] + deviation[0]):
            vals[2] = 1  # should go west
        if unit.pos.y < (self.target_SP_loc[pid][1] + deviation[1]):
            vals[3] = 1  # should go north
        return vals

    # 2021-12-07: Add the distance to the target point into agents' obs.
    # 2022-02-18: Calculate distance for individual agents. Normalize them.
    def get_distance_target(self, unit, aid, pid):
        alpha = 3
        deviation = [0, 0]  # The deviation is adjusted for each agent.
        if aid == 0:
            deviation = [0, alpha]
        if aid == 1:
            deviation = [0, -alpha]
        if aid == 2:
            deviation = [alpha, 0]
        if aid == 3:
            deviation = [-alpha, 0]

        # self.distance use math.hypot to calculate the distance
        distance = self.distance(unit.pos.x, unit.pos.y, (self.target_SP_loc[pid][0] + deviation[0]),
                                 (self.target_SP_loc[pid][1] + deviation[1]))
        return distance

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id. The observation is composed of:

           - agent movement features (where it can move to, height information and pathing grid)
           - enemy features (available_to_attack, health, relative_x, relative_y, shield, unit_type)
           - ally features (visible, distance, relative_x, relative_y, shield, unit_type)
           - agent unit features (health, shield, unit_type)

           All of this information is flattened and concatenated into a list,
           in the aforementioned order. To know the sizes of each of the
           features inside the final list of features, take a look at the
           functions ``get_obs_move_feats_size()``,
           ``get_obs_enemy_feats_size()``, ``get_obs_ally_feats_size()`` and
           ``get_obs_own_feats_size()``.

           The size of the observation vector may vary, depending on the
           environment configuration and type of units present in the map.
           For instance, non-Protoss units will not have shields, movement
           features may or may not include terrain height and pathing grid,
           unit_type is not included if there is only one type of unit in the
           map etc.).

           NOTE: Agents should have access only to their local observations
           during decentralised execution.
        """
        unit = self.get_unit_by_id(agent_id)

        move_feats_dim = self.get_obs_move_feats_size()
        enemy_feats_dim = self.get_obs_enemy_feats_size()
        ally_feats_dim = self.get_obs_ally_feats_size()
        own_feats_dim = self.get_obs_own_feats_size()

        move_feats = np.zeros(move_feats_dim, dtype=np.float32)
        enemy_feats = np.zeros(enemy_feats_dim, dtype=np.float32)
        ally_feats = np.zeros(ally_feats_dim, dtype=np.float32)
        own_feats = np.zeros(own_feats_dim, dtype=np.float32)

        if unit.health > 0:  # otherwise dead, return all zeros
            x = unit.pos.x
            y = unit.pos.y
            sight_range = self.unit_sight_range(agent_id)  # default vaue is 9

            # Movement features
            avail_actions = self.get_avail_agent_actions(agent_id)
            for m in range(self.n_actions_move):
                move_feats[m] = avail_actions[m + 2]  # skip the first 2 elements, no-op and stop in avili_actions[]

            ind = self.n_actions_move  # default is 4

            if self.obs_pathing_grid:
                move_feats[
                ind: ind + self.n_obs_pathing  # default is 8
                ] = self.get_surrounding_pathing(unit)
                ind += self.n_obs_pathing

            if self.obs_terrain_height:
                move_feats[
                ind: ind + self.n_obs_height  # default is 9
                ] = self.get_surrounding_height(unit)
                ind += self.n_obs_height

            if self.obs_direction_command:
                move_feats[ind:] = self.get_direction_command(unit)

            # Enemy features
            for e_id, e_unit in self.enemies.items():
                e_x = e_unit.pos.x
                e_y = e_unit.pos.y
                dist = self.distance(x, y, e_x, e_y)

                if (
                        dist < sight_range and e_unit.health > 0
                ):  # visible and alive
                    # Sight range > shoot range
                    enemy_feats[e_id, 0] = avail_actions[
                        self.n_actions_no_attack + e_id
                        ]  # available
                    enemy_feats[e_id, 1] = dist / sight_range  # distance
                    enemy_feats[e_id, 2] = (
                                                   e_x - x
                                           ) / sight_range  # relative X
                    enemy_feats[e_id, 3] = (
                                                   e_y - y
                                           ) / sight_range  # relative Y

                    ind = 4
                    if self.obs_all_health:
                        enemy_feats[e_id, ind] = (
                                e_unit.health / e_unit.health_max
                        )  # health
                        ind += 1
                        if self.shield_bits_enemy > 0:
                            max_shield = self.unit_max_shield(e_unit)
                            enemy_feats[e_id, ind] = (
                                    e_unit.shield / max_shield
                            )  # shield
                            ind += 1

                    if self.unit_type_bits > 0:
                        type_id = self.get_unit_type_id(e_unit, False)
                        enemy_feats[e_id, ind + type_id] = 1  # unit type

            # Ally features
            al_ids = [
                al_id for al_id in range(self.n_agents) if al_id != agent_id
            ]
            for i, al_id in enumerate(al_ids):

                al_unit = self.get_unit_by_id(al_id)
                al_x = al_unit.pos.x
                al_y = al_unit.pos.y
                dist = self.distance(x, y, al_x, al_y)

                if (
                        dist < sight_range and al_unit.health > 0
                ):  # visible and alive
                    ally_feats[i, 0] = 1  # visible
                    ally_feats[i, 1] = dist / sight_range  # distance
                    ally_feats[i, 2] = (al_x - x) / sight_range  # relative X
                    ally_feats[i, 3] = (al_y - y) / sight_range  # relative Y

                    ind = 4
                    if self.obs_all_health:
                        ally_feats[i, ind] = (
                                al_unit.health / al_unit.health_max
                        )  # health
                        ind += 1
                        if self.shield_bits_ally > 0:
                            max_shield = self.unit_max_shield(al_unit)
                            ally_feats[i, ind] = (
                                    al_unit.shield / max_shield
                            )  # shield
                            ind += 1

                    if self.unit_type_bits > 0:
                        type_id = self.get_unit_type_id(al_unit, True)
                        ally_feats[i, ind + type_id] = 1
                        ind += self.unit_type_bits

                    if self.obs_last_action:
                        ally_feats[i, ind:] = self.last_action[al_id]

            # Own features
            ind = 0
            if self.obs_own_health:
                own_feats[ind] = unit.health / unit.health_max
                ind += 1
                if self.shield_bits_ally > 0:
                    max_shield = self.unit_max_shield(unit)
                    own_feats[ind] = unit.shield / max_shield
                    ind += 1

            if self.unit_type_bits > 0:
                type_id = self.get_unit_type_id(unit, True)
                own_feats[ind + type_id] = 1

        agent_obs = np.concatenate(
            (
                move_feats.flatten(),
                enemy_feats.flatten(),
                ally_feats.flatten(),
                own_feats.flatten(),
            )
        )

        if self.obs_timestep_number:
            agent_obs = np.append(agent_obs,
                                  self._episode_steps / self.episode_limit)

        if self.debug:
            logging.debug("Obs Agent: {}".format(agent_id).center(60, "-"))
            logging.debug("Avail. actions {}".format(
                self.get_avail_agent_actions(agent_id)))
            logging.debug("Move feats {}".format(move_feats))
            logging.debug("Enemy feats {}".format(enemy_feats))
            logging.debug("Ally feats {}".format(ally_feats))
            logging.debug("Own feats {}".format(own_feats))

        return agent_obs

    def get_obs(self):
        """Returns all agent observations in a list.
        NOTE: Agents should have access only to their local observations
        during decentralised execution.
        """
        agents_obs = [self.get_obs_agent(i) for i in range(self.n_agents)]
        return agents_obs

    # TODO: 2021-10-10: Observation of the Level 3 commander
    def get_obs_L3(self):
        # The L3 observation contains 3 element, and in total, length is 48
        PlatoonsLocations = None  # it should be a tracker of the past positions of the platoons.
        PlatoonsHealth = None
        Enemies = None
        obs_L3 = PlatoonsLocations + PlatoonsHealth + Enemies
        return obs_L3

    def get_obs_company(self):
        """Returns the company observations.
        """
        company_obs = [self.get_obs_platoon(i) for i in range(self.n_ally_platoons)]
        return company_obs

    def get_obs_platoon(self, platoon_id):
        """
        Get the overall observation of a platoon. Each platoon have 4 agents.
        """
        platoon_obs = [self.get_obs_unit_agent(i, platoon_id) for i in range(self.n_ally_agent_in_platoon)]
        return platoon_obs

    def get_obs_unit_agent(self, agent_id, platoon_id):
        # Get the observation of each unit agent from a given platoon.

        """Returns observation for agent_id. The observation is composed of:

           - agent movement features (where it can move to, height information and pathing grid)
           - enemy features (available_to_attack, health, relative_x, relative_y, shield, unit_type)
           - ally features (visible, distance, relative_x, relative_y, shield, unit_type)
           - agent unit features (health, shield, unit_type)

           All of this information is flattened and concatenated into a list,
           in the aforementioned order. To know the sizes of each of the
           features inside the final list of features, take a look at the
           functions ``get_obs_move_feats_size()``,
           ``get_obs_enemy_feats_size()``, ``get_obs_ally_feats_size()`` and
           ``get_obs_own_feats_size()``.

           The size of the observation vector may vary, depending on the
           environment configuration and type of units present in the map.
           For instance, non-Protoss units will not have shields, movement
           features may or may not include terrain height and pathing grid,
           unit_type is not included if there is only one type of unit in the
           map etc.).

           NOTE: Agents should have access only to their local observations
           during decentralised execution.
        """

        # platoon = self.get_platoon_by_id(platoon_id)
        unit = self.get_platoon_unit_by_id(agent_id, platoon_id)

        # create the placeholder of observation
        move_feats_dim = self.get_obs_move_feats_size()  # update with the direction to the target point.
        enemy_feats_dim = self.get_obs_enemy_feats_size()
        ally_feats_dim = self.get_obs_ally_feats_size()
        own_feats_dim = self.get_obs_own_feats_size()

        move_feats = np.zeros(move_feats_dim, dtype=np.float32)
        enemy_feats = np.zeros(enemy_feats_dim, dtype=np.float32)
        ally_feats = np.zeros(ally_feats_dim, dtype=np.float32)
        own_feats = np.zeros(own_feats_dim, dtype=np.float32)

        if unit.health > 0:  # otherwise dead, return all zeros
            x = unit.pos.x
            y = unit.pos.y
            sight_range = self.unit_sight_range(agent_id)  # default value is 9

            # Movement features
            avail_actions = self.get_avail_platoon_agent_actions(agent_id, platoon_id)
            for m in range(self.n_actions_move):
                move_feats[m] = avail_actions[m + 2]  # skip the first 2 elements, no-op and stop in avili_actions[]

            ind = self.n_actions_move  # default is 4

            if self.obs_pathing_grid:
                move_feats[
                ind: ind + self.n_obs_pathing  # default is 8
                ] = self.get_surrounding_pathing(unit)
                ind += self.n_obs_pathing

            if self.obs_terrain_height:
                move_feats[
                ind: ind + self.n_obs_height  # default is 9
                ] = self.get_surrounding_height(unit)
                ind += self.n_obs_height

            if self.obs_direction_command:
                move_feats[
                ind: ind + self.n_obs_direction_command
                ] = self.get_direction_command(unit, agent_id, platoon_id)
                # move_feats[ind:] = self.get_direction_command(unit, platoon_id)
                ind += self.n_obs_direction_command

            # 2021-12-07: Add the direction to the target point in the observation
            # 2022-02-18: Normalize the distance with sight range.
            if self.obs_distance_target:
                move_feats[ind:] = self.get_distance_target(unit, agent_id, platoon_id) / sight_range

            # Enemy features
            for e_id, e_unit in self.enemies.items():
                e_x = e_unit.pos.x
                e_y = e_unit.pos.y
                dist = self.distance(x, y, e_x, e_y)

                if (
                        dist < sight_range and e_unit.health > 0
                ):  # visible and alive
                    # Sight range > shoot range
                    enemy_feats[e_id, 0] = avail_actions[
                        self.n_actions_no_attack + e_id
                        ]  # available
                    enemy_feats[e_id, 1] = dist / sight_range  # distance
                    enemy_feats[e_id, 2] = (e_x - x) / sight_range  # relative X
                    enemy_feats[e_id, 3] = (e_y - y) / sight_range  # relative Y

                    ind = 4
                    if self.obs_all_health:
                        enemy_feats[e_id, ind] = (
                                e_unit.health / e_unit.health_max
                        )  # health
                        ind += 1
                        if self.shield_bits_enemy > 0:
                            max_shield = self.unit_max_shield(e_unit)
                            enemy_feats[e_id, ind] = (
                                    e_unit.shield / max_shield
                            )  # shield
                            ind += 1

                    if self.unit_type_bits > 0:
                        type_id = self.get_unit_type_id(e_unit, False)
                        enemy_feats[e_id, ind + type_id] = 1  # unit type

            # Ally features
            al_ids = [
                al_id for al_id in range(self.n_ally_agent_in_platoon) if al_id != agent_id
            ]
            for i, al_id in enumerate(al_ids):

                al_unit = self.get_platoon_unit_by_id(al_id, platoon_id)
                al_x = al_unit.pos.x
                al_y = al_unit.pos.y
                dist = self.distance(x, y, al_x, al_y)

                if (dist < sight_range and al_unit.health > 0):  # visible and alive
                    ally_feats[i, 0] = 1  # visible
                    ally_feats[i, 1] = dist / sight_range  # distance
                    ally_feats[i, 2] = (al_x - x) / sight_range  # relative X
                    ally_feats[i, 3] = (al_y - y) / sight_range  # relative Y

                    ind = 4
                    if self.obs_all_health:
                        ally_feats[i, ind] = (
                                al_unit.health / al_unit.health_max
                        )  # health
                        ind += 1
                        if self.shield_bits_ally > 0:
                            max_shield = self.unit_max_shield(al_unit)
                            ally_feats[i, ind] = (
                                    al_unit.shield / max_shield
                            )  # shield
                            ind += 1

                    if self.unit_type_bits > 0:
                        type_id = self.get_unit_type_id(al_unit, True)
                        ally_feats[i, ind + type_id] = 1
                        ind += self.unit_type_bits

                    if self.obs_last_action:
                        ally_feats[i, ind:] = self.last_action[al_id]

            # Own features
            ind = 0
            if self.obs_own_health:
                own_feats[ind] = unit.health / unit.health_max
                ind += 1
                if self.shield_bits_ally > 0:
                    max_shield = self.unit_max_shield(unit)
                    own_feats[ind] = unit.shield / max_shield
                    ind += 1

            if self.unit_type_bits > 0:
                type_id = self.get_unit_type_id(unit, True)
                own_feats[ind + type_id] = 1

        agent_obs = np.concatenate(
            (move_feats.flatten(),
             enemy_feats.flatten(),
             ally_feats.flatten(),
             own_feats.flatten(),
             )
        )

        if self.obs_timestep_number:
            agent_obs = np.append(agent_obs,
                                  self._episode_steps / self.episode_limit)

        if self.debug:
            logging.debug("Platoon: {}".format(platoon_id).center(60, "-"))
            logging.debug("Obs Agent: {}".format(agent_id).center(60, "-"))
            logging.debug("Avail. actions {}".format(
                self.get_avail_agent_actions(agent_id)))
            logging.debug("Move feats {}".format(move_feats))
            logging.debug("Enemy feats {}".format(enemy_feats))
            logging.debug("Ally feats {}".format(ally_feats))
            logging.debug("Own feats {}".format(own_feats))

        return agent_obs

    def get_state(self):
        """Returns the global state.
        NOTE: This functon should not be used during decentralised execution.
        """
        if self.obs_instead_of_state:
            obs_concat = np.concatenate(self.get_obs(), axis=0).astype(
                np.float32
            )
            return obs_concat
        nf_al = 4 + self.shield_bits_ally + self.unit_type_bits  # default length is 4, [health, cooldown, rel_x, rel_y]
        nf_en = 3 + self.shield_bits_enemy + self.unit_type_bits  # default length is 3, [health, rel_x, rel_y]

        ally_state = np.zeros((self.n_agents, nf_al))  # shape (8,4)
        enemy_state = np.zeros((self.n_enemies, nf_en))  # shape (8,3)

        center_x = self.map_x / 2
        center_y = self.map_y / 2

        for al_id, al_unit in self.agents.items():
            if al_unit.health > 0:
                x = al_unit.pos.x
                y = al_unit.pos.y
                max_cd = self.unit_max_cooldown(al_unit)

                ally_state[al_id, 0] = (
                        al_unit.health / al_unit.health_max
                )  # health
                if (
                        self.map_type == "MMM"
                        and al_unit.unit_type == self.medivac_id
                ):
                    ally_state[al_id, 1] = al_unit.energy / max_cd  # energy
                else:
                    ally_state[al_id, 1] = (
                            al_unit.weapon_cooldown / max_cd
                    )  # cooldown
                ally_state[al_id, 2] = (
                                               x - center_x
                                       ) / self.max_distance_x  # relative X
                ally_state[al_id, 3] = (
                                               y - center_y
                                       ) / self.max_distance_y  # relative Y

                ind = 4
                if self.shield_bits_ally > 0:
                    max_shield = self.unit_max_shield(al_unit)
                    ally_state[al_id, ind] = (
                            al_unit.shield / max_shield
                    )  # shield
                    ind += 1

                if self.unit_type_bits > 0:
                    type_id = self.get_unit_type_id(al_unit, True)
                    ally_state[al_id, ind + type_id] = 1

        for e_id, e_unit in self.enemies.items():
            if e_unit.health > 0:
                x = e_unit.pos.x
                y = e_unit.pos.y

                enemy_state[e_id, 0] = (
                        e_unit.health / e_unit.health_max
                )  # health
                enemy_state[e_id, 1] = (
                                               x - center_x
                                       ) / self.max_distance_x  # relative X
                enemy_state[e_id, 2] = (
                                               y - center_y
                                       ) / self.max_distance_y  # relative Y

                ind = 3
                if self.shield_bits_enemy > 0:
                    max_shield = self.unit_max_shield(e_unit)
                    enemy_state[e_id, ind] = (
                            e_unit.shield / max_shield
                    )  # shield
                    ind += 1

                if self.unit_type_bits > 0:
                    type_id = self.get_unit_type_id(e_unit, False)
                    enemy_state[e_id, ind + type_id] = 1

        state = np.append(ally_state.flatten(), enemy_state.flatten())
        if self.state_last_action:
            state = np.append(state, self.last_action.flatten())
        if self.state_timestep_number:
            state = np.append(state,
                              self._episode_steps / self.episode_limit)

        state = state.astype(dtype=np.float32)

        if self.debug:
            logging.debug("STATE".center(60, "-"))
            logging.debug("Ally state {}".format(ally_state))
            logging.debug("Enemy state {}".format(enemy_state))
            if self.state_last_action:
                logging.debug("Last actions {}".format(self.last_action))

        return state

    def get_state_company(self):
        """Returns the global state of the whole company.
        NOTE: This function should not be used during decentralised execution.
        """
        if self.obs_instead_of_state:
            obs_company = np.concatenate(self.get_obs_company(), axis=0).astype(np.float32)
            # obs_concat = np.concatenate(self.get_obs(), axis=0).astype(np.float32)
            return obs_company

        nf_al = 4 + self.shield_bits_ally + self.unit_type_bits  # default length is 4, [health, cooldown, rel_x, rel_y]
        nf_en = 3 + self.shield_bits_enemy + self.unit_type_bits  # default length is 3, [health, rel_x, rel_y]

        ally_state = np.zeros((self.n_agents, nf_al))  # shape (8,4)
        enemy_state = np.zeros((self.n_enemies, nf_en))  # shape (8,3)

        center_x = self.map_x / 2  # center of the map
        center_y = self.map_y / 2

        for pid in self.ally_platoons:
            for al_id, al_unit in enumerate(pid):
                # for al_id, al_unit in self.agents.items():
                if al_unit.health > 0:
                    x = al_unit.pos.x
                    y = al_unit.pos.y
                    max_cd = self.unit_max_cooldown(al_unit)

                    ally_state[al_id, 0] = (
                            al_unit.health / al_unit.health_max
                    )  # health
                    if (
                            self.map_type == "MMM"
                            and al_unit.unit_type == self.medivac_id
                    ):
                        ally_state[al_id, 1] = al_unit.energy / max_cd  # energy
                    else:
                        ally_state[al_id, 1] = (
                                al_unit.weapon_cooldown / max_cd
                        )  # cooldown
                    ally_state[al_id, 2] = (
                                                   x - center_x
                                           ) / self.max_distance_x  # relative X
                    ally_state[al_id, 3] = (
                                                   y - center_y
                                           ) / self.max_distance_y  # relative Y

                    ind = 4
                    if self.shield_bits_ally > 0:
                        max_shield = self.unit_max_shield(al_unit)
                        ally_state[al_id, ind] = (
                                al_unit.shield / max_shield
                        )  # shield
                        ind += 1

                    if self.unit_type_bits > 0:
                        type_id = self.get_unit_type_id(al_unit, True)
                        ally_state[al_id, ind + type_id] = 1

        for e_id, e_unit in self.enemies.items():
            if e_unit.health > 0:
                x = e_unit.pos.x
                y = e_unit.pos.y

                enemy_state[e_id, 0] = (
                        e_unit.health / e_unit.health_max
                )  # health
                enemy_state[e_id, 1] = (
                                               x - center_x
                                       ) / self.max_distance_x  # relative X
                enemy_state[e_id, 2] = (
                                               y - center_y
                                       ) / self.max_distance_y  # relative Y

                ind = 3
                if self.shield_bits_enemy > 0:
                    max_shield = self.unit_max_shield(e_unit)
                    enemy_state[e_id, ind] = (
                            e_unit.shield / max_shield
                    )  # shield
                    ind += 1

                if self.unit_type_bits > 0:
                    type_id = self.get_unit_type_id(e_unit, False)
                    enemy_state[e_id, ind + type_id] = 1

        state = np.append(ally_state.flatten(), enemy_state.flatten())
        if self.state_last_action:  # By default, yes.
            state = np.append(state, self.last_action.flatten())  # TODO: Check how does the last action added
        if self.state_timestep_number:  # default is false
            state = np.append(state,
                              self._episode_steps / self.episode_limit)

        state = state.astype(dtype=np.float32)

        if self.debug:
            logging.debug("STATE".center(60, "-"))
            logging.debug("Ally state {}".format(ally_state))
            logging.debug("Enemy state {}".format(enemy_state))
            if self.state_last_action:
                logging.debug("Last actions {}".format(self.last_action))

        return state

    def get_obs_enemy_feats_size(self):
        """ Returns the dimensions of the matrix containing enemy features.
        Size is n_enemies x n_features.
        """
        nf_en = 4 + self.unit_type_bits

        if self.obs_all_health:
            nf_en += 1 + self.shield_bits_enemy

        return self.n_enemies, nf_en

    def get_obs_ally_feats_size(self):
        """Returns the dimensions of the matrix containing ally features.
        Size is n_allies x n_features.
        """
        nf_al = 4 + self.unit_type_bits

        if self.obs_all_health:
            nf_al += 1 + self.shield_bits_ally

        if self.obs_last_action:
            nf_al += self.n_actions
        if self.hierarchical:
            return self.n_ally_agent_in_platoon - 1, nf_al
        else:
            return self.n_agents - 1, nf_al

    def get_obs_own_feats_size(self):
        """Returns the size of the vector containing the agents' own features.
        """
        own_feats = self.unit_type_bits
        if self.obs_own_health:
            own_feats += 1 + self.shield_bits_ally
        if self.obs_timestep_number:
            own_feats += 1

        return own_feats

    def get_obs_move_feats_size(self):
        """Returns the size of the vector containing the agents's movement-related features."""
        move_feats = self.n_actions_move
        if self.obs_pathing_grid:
            move_feats += self.n_obs_pathing
        if self.obs_terrain_height:
            move_feats += self.n_obs_height
        # 2021-08-05: add the observation of the direction of the strategic point.
        if self.obs_direction_command:
            move_feats += self.n_obs_direction_command
        # 2021-12-07: add the distance to the target point as part of the observation
        if self.obs_distance_target:
            move_feats += self.n_obs_distance_target

        return move_feats

    def get_obs_size(self):
        """Returns the size of the observation."""
        own_feats = self.get_obs_own_feats_size()
        move_feats = self.get_obs_move_feats_size()

        n_enemies, n_enemy_feats = self.get_obs_enemy_feats_size()  # Returns the dimensions of the matrix containing enemy features.
        n_allies, n_ally_feats = self.get_obs_ally_feats_size()  # Returns the dimensions of the matrix containing ally features. Size is n_allies x n_features.

        enemy_feats = n_enemies * n_enemy_feats
        ally_feats = n_allies * n_ally_feats

        return move_feats + enemy_feats + ally_feats + own_feats

    def get_company_obs_size(self):
        """Returns the size of the observation."""
        own_feats = self.get_obs_own_feats_size()
        move_feats = self.get_obs_move_feats_size()

        n_enemies, n_enemy_feats = self.get_obs_enemy_feats_size()  # Returns the dimensions of the matrix containing enemy features.
        n_allies, n_ally_feats = self.get_obs_ally_feats_size()  # Returns the dimensions of the matrix containing ally features. Size is n_allies x n_features.

        enemy_feats = n_enemies * n_enemy_feats
        ally_feats = n_allies * n_ally_feats

        return move_feats + enemy_feats + ally_feats + own_feats

    def get_platoon_obs_size(self):
        """Returns the size of the observation."""
        own_feats = self.get_obs_own_feats_size()
        move_feats = self.get_obs_move_feats_size()

        n_enemies, n_enemy_feats = self.get_obs_enemy_feats_size()  # Returns the dimensions of the matrix containing enemy features.
        n_allies, n_ally_feats = self.get_obs_ally_feats_size()  # Returns the dimensions of the matrix containing ally features. Size is n_allies x n_features.

        enemy_feats = n_enemies * n_enemy_feats
        ally_feats = n_allies * n_ally_feats

        return move_feats + enemy_feats + ally_feats + own_feats

    def get_state_size(self):
        """Returns the size of the global state."""
        if self.obs_instead_of_state:
            return self.get_obs_size() * self.n_agents

        nf_al = 4 + self.shield_bits_ally + self.unit_type_bits
        nf_en = 3 + self.shield_bits_enemy + self.unit_type_bits

        enemy_state = self.n_enemies * nf_en
        ally_state = self.n_agents * nf_al

        size = enemy_state + ally_state

        if self.state_last_action:
            size += self.n_agents * self.n_actions
        if self.state_timestep_number:
            size += 1

        return size

    # added in hierarchical control
    def get_company_state_size(self):
        """Returns the size of the company global state."""
        if self.obs_instead_of_state:
            return self.get_obs_size() * self.n_agents

        nf_al = 4 + self.shield_bits_ally + self.unit_type_bits
        nf_en = 3 + self.shield_bits_enemy + self.unit_type_bits

        enemy_state = self.n_enemies * nf_en
        ally_state = self.n_agents * nf_al

        size = enemy_state + ally_state

        if self.state_last_action:
            size += self.n_agents * self.n_actions
        if self.state_timestep_number:
            size += 1
        return size

    # added in hierarchical control
    def get_platoon_state_size(self):
        """Returns the size of the company global state."""
        if self.obs_instead_of_state:
            return self.get_obs_size() * self.n_ally_agent_in_platoon

        nf_al = 4 + self.shield_bits_ally + self.unit_type_bits
        nf_en = 3 + self.shield_bits_enemy + self.unit_type_bits

        enemy_state = self.n_enemies * nf_en
        ally_state = self.n_ally_agent_in_platoon * nf_al

        size = enemy_state + ally_state

        if self.state_last_action:
            size += self.n_ally_agent_in_platoon * self.n_actions
        if self.state_timestep_number:
            size += 1
        return size

    def get_visibility_matrix(self):
        """Returns a boolean numpy array of dimensions 
        (n_agents, n_agents + n_enemies) indicating which units
        are visible to each agent.
        """
        arr = np.zeros(
            (self.n_agents, self.n_agents + self.n_enemies),
            dtype=np.bool,
        )

        for agent_id in range(self.n_agents):
            current_agent = self.get_unit_by_id(agent_id)
            if current_agent.health > 0:  # it agent not dead
                x = current_agent.pos.x
                y = current_agent.pos.y
                sight_range = self.unit_sight_range(agent_id)

                # Enemies
                for e_id, e_unit in self.enemies.items():
                    e_x = e_unit.pos.x
                    e_y = e_unit.pos.y
                    dist = self.distance(x, y, e_x, e_y)

                    if (dist < sight_range and e_unit.health > 0):
                        # visible and alive
                        arr[agent_id, self.n_agents + e_id] = 1

                # The matrix for allies is filled symmetrically
                al_ids = [
                    al_id for al_id in range(self.n_agents)
                    if al_id > agent_id
                ]
                for i, al_id in enumerate(al_ids):
                    al_unit = self.get_unit_by_id(al_id)
                    al_x = al_unit.pos.x
                    al_y = al_unit.pos.y
                    dist = self.distance(x, y, al_x, al_y)

                    if (dist < sight_range and al_unit.health > 0):
                        # visible and alive
                        arr[agent_id, al_id] = arr[al_id, agent_id] = 1

        return arr

    def get_unit_type_id(self, unit, ally):
        """Returns the ID of unit type in the given scenario."""
        if ally:  # use new SC2 unit types
            type_id = unit.unit_type - self._min_unit_type
        else:  # use default SC2 unit types
            if self.map_type == "stalkers_and_zealots":
                # id(Stalker) = 74, id(Zealot) = 73
                type_id = unit.unit_type - 73
            elif self.map_type == "colossi_stalkers_zealots":
                # id(Stalker) = 74, id(Zealot) = 73, id(Colossus) = 4
                if unit.unit_type == 4:
                    type_id = 0
                elif unit.unit_type == 74:
                    type_id = 1
                else:
                    type_id = 2
            elif self.map_type == "bane":
                if unit.unit_type == 9:
                    type_id = 0
                else:
                    type_id = 1
            elif self.map_type == "MMM":
                if unit.unit_type == 51:
                    type_id = 0
                elif unit.unit_type == 48:
                    type_id = 1
                else:
                    type_id = 2

        return type_id

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id."""
        # return a list full of 0 & 1, with n_action as length. 1 means the agent could do the action represented in this place.
        unit = self.get_unit_by_id(agent_id)
        if unit.health > 0:
            # cannot choose no-op when alive
            avail_actions = [0] * self.n_actions  # in 3m map, n_action is 6+3=9. In 8m, is 6+8=14

            # stop should be allowed
            avail_actions[1] = 1

            # see if we can move
            if self.can_move(unit, Direction.NORTH):
                avail_actions[2] = 1
            if self.can_move(unit, Direction.SOUTH):
                avail_actions[3] = 1
            if self.can_move(unit, Direction.EAST):
                avail_actions[4] = 1
            if self.can_move(unit, Direction.WEST):
                avail_actions[5] = 1

            # Can attack only alive units that are alive in the shooting range
            shoot_range = self.unit_shoot_range(agent_id)  # default value is 6

            # the enemies could be replaced by enemy_platoon
            target_items = self.enemies.items()
            if self.map_type == "MMM" and unit.unit_type == self.medivac_id:
                # Medivacs cannot heal themselves or other flying units
                target_items = [
                    (t_id, t_unit)
                    for (t_id, t_unit) in self.agents.items()
                    if t_unit.unit_type != self.medivac_id
                ]

            for t_id, t_unit in target_items:
                if t_unit.health > 0:
                    dist = self.distance(
                        unit.pos.x, unit.pos.y, t_unit.pos.x, t_unit.pos.y
                    )
                    if dist <= shoot_range:
                        avail_actions[t_id + self.n_actions_no_attack] = 1

            return avail_actions

        else:
            # only no-op allowed
            return [1] + [0] * (self.n_actions - 1)

    def get_avail_platoon_agent_actions(self, agent_id, platoon_id):
        """ Get the available actions for an agent in a given platoon. """
        # return a list full of 0 & 1, with n_action as length. 1 means the agent could do the action represented in this place.
        unit = self.get_platoon_unit_by_id(agent_id, platoon_id)

        if unit.health > 0:
            # cannot choose no-op when alive
            avail_actions = [0] * self.n_actions  # in 3m map, n_action is 6+3=9. In 8m, is 6+8=14

            # stop should be allowed
            avail_actions[1] = 1

            # see if we can move
            if self.can_move(unit, Direction.NORTH):
                avail_actions[2] = 1
            if self.can_move(unit, Direction.SOUTH):
                avail_actions[3] = 1
            if self.can_move(unit, Direction.EAST):
                avail_actions[4] = 1
            if self.can_move(unit, Direction.WEST):
                avail_actions[5] = 1

            # The agents could only attack the enemies in the shooting range.
            shoot_range = self.platoon_unit_shoot_range(agent_id, platoon_id)  # default value is 7

            # the enemies could be replaced by enemy_platoon
            target_items = self.enemies.items()
            if self.map_type == "MMM" and unit.unit_type == self.medivac_id:
                # Medivacs cannot heal themselves or other flying units
                target_items = [
                    (t_id, t_unit)
                    for (t_id, t_unit) in self.agents.items()
                    if t_unit.unit_type != self.medivac_id
                ]

            # check if each enemy could be attacked
            for t_id, t_unit in target_items:
                if t_unit.health > 0:
                    dist = self.distance(
                        unit.pos.x, unit.pos.y, t_unit.pos.x, t_unit.pos.y
                    )
                    if dist <= shoot_range:
                        avail_actions[t_id + self.n_actions_no_attack] = 1

            return avail_actions

        else:
            # only no-op allowed
            return [1] + [0] * (self.n_actions - 1)

    def get_avail_actions(self):
        """Returns the available actions of all agents in a list."""
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def close(self):
        """Close StarCraft II."""
        if self._sc2_proc:
            self._sc2_proc.close()

    def seed(self):
        """Returns the random seed used by the environment."""
        return self._seed

    def render(self):
        """Not implemented."""
        pass

    def _kill_all_units(self):
        """Kill all units on the map."""
        if self.hierarchical:
            units_alive = [
                              unit.tag for platoon in self.ally_platoons for unit in platoon if unit.health > 0
                          ] + [unit.tag for unit in self.enemies.values() if unit.health > 0]

            debug_command = [
                d_pb.DebugCommand(kill_unit=d_pb.DebugKillUnit(tag=units_alive))
            ]
        else:
            units_alive = [
                              unit.tag for unit in self.agents.values() if unit.health > 0
                          ] + [unit.tag for unit in self.enemies.values() if unit.health > 0]
            debug_command = [
                d_pb.DebugCommand(kill_unit=d_pb.DebugKillUnit(tag=units_alive))
            ]

        self._controller.debug(debug_command)

    def init_platoons(self):
        n_ally_platoons = self.n_ally_platoons
        n_ally_agent_in_platoon = self.n_ally_agent_in_platoon
        n_enemy_platoons = self.n_enemy_platoons
        n_enemy_unit_in_platoon = self.n_enemy_unit_in_platoon

        while True:
            #print('stuck init platoons')
            self.ally_platoons = [[] for _ in range(n_ally_platoons)]
            self.enemy_platoons = [[] for _ in range(n_enemy_platoons)]

            if not self.hierarchical:
                self.agents = {}  # disable the self.
            self.enemies = {}

            # grab the units from the environment
            ally_units = [
                unit
                for unit in self._obs.observation.raw_data.units
                if unit.owner == 1
            ]

            # The agents are sorted by pos.y
            ally_units_sorted = sorted(
                ally_units,
                key=attrgetter("unit_type", "pos.y"),  # sort the agents by y position
                reverse=False,
            )

            # Update the init positions of units
            self.initial_pos = [[] * self.n_ally_agent_in_platoon] * self.n_ally_platoons
            for i in range(len(ally_units_sorted)):
                # self.agents[i] = ally_units_sorted[i]
                for t in range(len(self.ally_platoons)):
                    if len(self.ally_platoons[t]) < n_ally_agent_in_platoon:
                        self.ally_platoons[t].append(ally_units_sorted[i])
                        if self.debug:
                            logging.debug(
                                "Ally unit {} is {}, x = {}, y = {}, in platoon {}.".format(
                                    i,
                                    ally_units_sorted[i].unit_type,
                                    ally_units_sorted[i].pos.x,
                                    ally_units_sorted[i].pos.y,
                                    t,
                                )
                            )
                        self.initial_pos[t].append([ally_units_sorted[i].pos.x,
                                                    ally_units_sorted[i].pos.y])
                        break

            # sort enemies
            enemy_units = [
                unit
                for unit in self._obs.observation.raw_data.units
                if unit.owner == 2
            ]

            # update max_reward  # todo 2022-03-08 this is where cause the reward to be very huge
            counter = 0
            # if self.map_name in ['4t_vs_0t_8SPs_randomized', '4t_vs_0t_8SPs',
            #                      '4t_vs_0t_8SPs_RandomEnemy',
            #                      '4t_vs_0t_8SPs_RandomEnemy_075']:  # these scenarios doesn't have enemies
            #     for i in range(self.n_enemies):
            #         self.max_reward += 160
            #         counter += 1

            # for unit in self._obs.observation.raw_data.units:
            #     if unit.owner == 2:
            #         self.enemies[len(self.enemies)] = unit
            #         # separate the function to update the max_reward
            #         if self._episode_count == 0:
            #             self.max_reward += unit.health_max + unit.shield_max
            #             counter += 1

            for unit in self._obs.observation.raw_data.units:
                if unit.owner == 2:
                    self.enemies[len(self.enemies)] = unit

            # separate the function to update the max_reward
            if len(self.enemies) != 0:  # the map has some enemies
                for unit in self._obs.observation.raw_data.units:
                    if unit.owner == 2:
                        if self._episode_count == 0:
                            self.max_reward += unit.health_max + unit.shield_max
                            counter += 1
            else:  # there is no enemy in the map
                if self._episode_count == 0:
                    if self.map_name in ['4t_vs_0t_8SPs_randomized', '4t_vs_0t_8SPs',
                                         '4t_vs_0t_8SPs_RandomEnemy',
                                         '4t_vs_0t_8SPs_RandomEnemy_075']:  # these scenarios doesn't have enemies
                        for i in range(self.n_enemies):
                            self.max_reward += 160
                            counter += 1

            if counter == self.n_enemies:
                # use alpha to change the proportion of reward for combat and reward for moving.
                # self.max_reward = self.alpha * self.max_reward \
                #                   + (1 - self.alpha) * self.n_agents * self.reward_SP \
                #                   + self.reward_win
                self.max_reward = self.alpha * self.max_reward \
                                  + (1 - self.alpha) * (self.n_agents * self.reward_SP
                                                        + self.n_sp * self.reward_arrive) \
                                  + self.reward_win
            # # Palse to put enemies into platoon
            # for i in range(len(enemy_units)):
            #     self.enemies[i] = enemy_units[i]
            #     for t in range(len(self.enemy_platoons)):
            #         if len(self.enemy_platoons[t]) < n_enemy_unit_in_platoon:
            #             self.enemy_platoons[t].append(enemy_units[i])
            #             if self.debug:
            #                 logging.debug(
            #                     "Enemy unit {} is {}, x = {}, y = {}, in platoon {}.".format(
            #                         i,
            #                         ally_units_sorted[i].unit_type,
            #                         ally_units_sorted[i].pos.x,
            #                         ally_units_sorted[i].pos.y,
            #                         t,
            #                     )
            #                 )
            #             break

            # while True:
            #     # Sometimes not all units have yet been created by SC2
            #     self.agents = {}
            #     self.enemies = {}
            #
            #     # grab the units from the environment
            #     ally_units = [
            #         unit
            #         for unit in self._obs.observation.raw_data.units
            #         if unit.owner == 1
            #     ]
            #
            #     # sort the agent units
            #     ally_units_sorted = sorted(
            #         ally_units,
            #         key=attrgetter("unit_type", "pos.x", "pos.y"),
            #         reverse=False,
            #     )
            #
            #
            #     self.initial_pos = []
            #     for i in range(len(ally_units_sorted)):
            #         self.agents[i] = ally_units_sorted[i]
            #         if self.debug:
            #             logging.debug(
            #                 "Unit {} is {}, x = {}, y = {}".format(
            #                     len(self.agents),
            #                     self.agents[i].unit_type,
            #                     self.agents[i].pos.x,
            #                     self.agents[i].pos.y,
            #                 )
            #             )
            #         #  record the initialized position for agents.
            #         self.initial_pos.append([self.agents[i].pos.x, self.agents[i].pos.y])

            # -------------------------------------------
            # use initial position as the last position
            # for i in range(len(self.agents)):
            #    self.agents[i].last_pos.last_x = self.agents[i].pos.x
            #    self.agents[i].last_pos.last_y = self.agents[i].pos.y
            # self.agents[i].pos.last_x = self.agents[i].pos.x
            # self.agents[i].pos.last_y = self.agents[i].pos.y
            # -------------------------------------------

            # -------------------------------------------
            # 2021-09-14: Add the reach-location counter
            # todo this counter of reaching needs to be changed for hierarchical learning

            # add record of arriving points
            # TODO: [10-19] init the first target location

            # TODO Update on 2022-03-16: Get the start location.
            self.init_platoon_SPs = []
            for platoon in self.ally_platoons:
                init_platoon_SP = self.get_platoon_current_sp(platoon)  # get the current SP of the selected platoon
                self.init_platoon_SPs.append(init_platoon_SP)

            # initialize the recording of reaching the SPs
            if self.hierarchical:
                self.agent_reach_point = [[0] * self.n_ally_agent_in_platoon for _ in range(self.n_ally_platoons)]
                self.platoons_move_record = [[0] * len(self.map_sps) for _ in
                                             range(self.n_ally_platoons)]  # Track where the platoons has been

                if self.FALCON_demo:  # todo: initial different SPs for each platoon
                    '''
                    In hierarchical control architecture, the FALCON_demo initialize the platoons with different targets, 
                    where P1, P2, P3 are given with SP1, SP1, SP2 as the start of the episode.
                    In non-hierarchical scenarios, the platoons are given with the same target point, SP1, at the start of 
                    the episode. 

                    NOTE: When do the experiments of the movement in the intervals, the first SP will changes. 
                    Every agents need to go to sp 2 in the 4t_vs_4t_SP12 scenario
                    '''
                    self.target_SP = ["0", "1", "2"]
                    self.target_SP_loc = [self.map_sps["0"], self.map_sps["1"], self.map_sps["2"]]
                    self.target_SP_id = [0, 1, 2]

                    # TODO: [2021-11-05] Make P2 the first platon to move
                    self.next_movement_platoon = 2  # the first platoon to move is p0
                else:
                    # self.target_SP = ["0"] * self.n_ally_platoons
                    # self.target_SP_loc = [self.map_sps["0"] * self.n_ally_platoons]
                    # self.target_SP_id = [0 * self.n_ally_platoons]

                    self.target_SP_id = self.get_target_SP_id(self.init_platoon_SPs)
                    #print('---------', self.target_SP_id, self.n_ally_platoons)
                    self.target_SP_loc = [self.map_sps[str(self.target_SP_id[i])] for i in range(self.n_ally_platoons)]
                    self.target_SP = [str(self.target_SP_id[i]) for i in range(self.n_ally_platoons)]
            else:
                self.agent_reach_point = [0 for _ in range(self.n_ally_agent_in_platoon)]
            # origin -------------------------------------------

            # counter = 0
            # for unit in self._obs.observation.raw_data.units:
            #     if unit.owner == 2:
            #         self.enemies[len(self.enemies)] = unit
            #         if self._episode_count == 0:
            #             self.max_reward += unit.health_max + unit.shield_max
            #             counter += 1

            #
            # if counter == self.n_enemies:
            #     # use alpha to change the proportion of reward for combat and reward for moving.
            #     self.max_reward = self.alpha * self.max_reward \
            #                       + (1 - self.alpha) * self.n_agents * self.reward_StrategicPoint_val \
            #                       + self.reward_win

            # TODO: Check the usage of min_unit_type
            if self._episode_count == 0:
                if self.hierarchical:
                    min_unit_type = min(
                        # unit.unit_type for unit in self.agents.values()
                        unit.unit_type for platoon in self.ally_platoons for unit in platoon
                    )
                else:
                    min_unit_type = min(
                        unit.unit_type for unit in self.agents.values()
                    )
                self._init_ally_unit_types(min_unit_type)

            # check if all units are created
            all_ally_platoons_created = (sum([len(platoon) for platoon in self.ally_platoons]) == self.n_agents)
            # all_enemy_platoons_created = (sum([len(platoon) for platoon in self.enemy_platoons]) == self.n_enemies)
            all_enemies_created = (len(self.enemies) == self.n_enemies)

            # Update on 2022-03-07: as there is no enemies in 4t_vs_0t_8SPs scenario, the
            if self.map_name in ['4t_vs_0t_8SPs', '4t_vs_0t_8SPs_randomized',
                                 '4t_vs_0t_8SPs_RandomEnemy', '4t_vs_0t_8SPs_RandomEnemy_075']:
                all_enemies_created = True
            #print('SHUBHAM', all_ally_platoons_created, all_enemies_created)
            if all_ally_platoons_created and all_enemies_created:  # all good
                return
            try:
                self._controller.step(1)
                # TODO: if the code stop at this line, check if the n_agent parameter is correct.
                self._obs = self._controller.observe()
            except (protocol.ProtocolError, protocol.ConnectionError):
                self.full_restart()
                self.reset()
        return
        # use the result of self.init_units.

    def init_units(self):
        """Initialise the units."""
        while True:
            print('stuck init units')
            # Sometimes not all units have yet been created by SC2
            self.agents = {}
            self.enemies = {}

            # get the ally units from the environment
            ally_units = [
                unit
                for unit in self._obs.observation.raw_data.units
                if unit.owner == 1
            ]

            # sort the ally units
            ally_units_sorted = sorted(
                ally_units,
                key=attrgetter("unit_type", "pos.x", "pos.y"),
                reverse=False,
            )

            # put the sorted agents in the dictionary.
            self.initial_pos = []
            for i in range(len(ally_units_sorted)):
                self.agents[i] = ally_units_sorted[i]
                if self.debug:
                    logging.debug(
                        "Unit {} is {}, x = {}, y = {}".format(
                            len(self.agents),
                            self.agents[i].unit_type,
                            self.agents[i].pos.x,
                            self.agents[i].pos.y,
                        )
                    )
                #  record the initialized position for agents.
                self.initial_pos.append([self.agents[i].pos.x, self.agents[i].pos.y])

            # -------------------------------------------
            # use initial position as the last position
            # for i in range(len(self.agents)):
            #    self.agents[i].last_pos.last_x = self.agents[i].pos.x
            #    self.agents[i].last_pos.last_y = self.agents[i].pos.y
            # self.agents[i].pos.last_x = self.agents[i].pos.x
            # self.agents[i].pos.last_y = self.agents[i].pos.y
            # -------------------------------------------

            # 2021-09-14: Add the reach-location counter
            self.agent_reach_point = [0 for _ in range(self.n_agents)]
            if self.hierarchical:
                self.agent_reach_point = [[False] * self.n_ally_agent_in_platoon] * self.n_ally_platoons

            # put the enemies in the dictionary.
            # Meanwhile update the max_reward
            counter = 0
            for unit in self._obs.observation.raw_data.units:
                if unit.owner == 2:
                    self.enemies[len(self.enemies)] = unit
                    if self._episode_count == 0:
                        '''
                        Update the max reward with the aggregated enemy health value.
                        In 4t scenario, the aggregated enemy health value is 160 * 4 = 640.
                        The shield value for siege tank is 0.
                        '''
                        self.max_reward += unit.health_max + unit.shield_max
                        counter += 1
            if counter == self.n_enemies:
                # self.max_reward = self.alpha * self.max_reward \
                #                   + (1 - self.alpha) * self.n_agents * self.reward_SP \
                #                   + self.reward_win
                '''
                Use alpha to balance the proportion of combat reward and movement reward.
                Update the max_reward with the reward_arrive (the reward for arriving strategic point)
                '''
                self.max_reward = self.alpha * self.max_reward \
                                  + (1 - self.alpha) * (self.n_agents * self.reward_SP
                                                        + self.n_sp * self.reward_arrive) \
                                  + self.reward_win

            # TODO What does this code chunk means?
            if self._episode_count == 0:
                min_unit_type = min(
                    unit.unit_type for unit in self.agents.values()
                )
                self._init_ally_unit_types(min_unit_type)

            # check if all agents and enemies are created
            all_agents_created = (len(self.agents) == self.n_agents)
            all_enemies_created = (len(self.enemies) == self.n_enemies)

            if all_agents_created and all_enemies_created:  # all good
                return

            try:
                self._controller.step(1)
                self._obs = self._controller.observe()
            except (protocol.ProtocolError, protocol.ConnectionError):
                self.full_restart()
                self.reset()

    def update_platoons(self):
        """ Update platoons after an environment step. """
        n_ally_alive = [0] * self.n_ally_platoons
        n_enemy_alive = 0

        arrive_reward = [False] * self.n_ally_platoons

        # Store previous state
        # self.previous_ally_units = deepcopy(self.agents)
        # self.previous_enemy_units = deepcopy(self.enemies)

        self.previous_ally_platoons = deepcopy(self.ally_platoons)
        self.previous_enemy_platoons = deepcopy(self.enemies)

        if self.hierarchical:
            # count the number of live agents in each platoon
            for pid in range(self.n_ally_platoons):
                for al_id, al_unit in enumerate(self.ally_platoons[pid]):
                    updated = False
                    for unit in self._obs.observation.raw_data.units:
                        # read the units from the env, put them in ally_platoon
                        if al_unit.tag == unit.tag:
                            self.ally_platoons[pid][al_id] = unit
                            updated = True
                            n_ally_alive[pid] += 1
                            break
                    if not updated:  # dead
                        al_unit.health = 0

            # count the alive enemies
            for e_id, e_unit in self.enemies.items():
                updated = False
                for unit in self._obs.observation.raw_data.units:
                    if e_unit.tag == unit.tag:
                        self.enemies[e_id] = unit
                        updated = True
                        n_enemy_alive += 1
                        break

                if not updated:  # dead
                    e_unit.health = 0

            number_arrive = [0] * self.n_ally_platoons  # the count of agents that arrive at the assigned SP
            arrive_lastSP = None

            # Update on 2022-02-18: Calculate the distance of individual agent.
            for pid in range(self.n_ally_platoons):
                for al_id, al_unit in enumerate(self.ally_platoons[pid]):
                    alpha = 3
                    deviation = [0, 0]  # The deviation is adjusted for each agent.

                    if al_id == 0:
                        deviation = [0, alpha]
                    if al_id == 1:
                        deviation = [0, -alpha]
                    if al_id == 2:
                        deviation = [alpha, 0]
                    if al_id == 3:
                        deviation = [-alpha, 0]

                    distance_x = al_unit.pos.x - (self.target_SP_loc[pid][0] + deviation[0])
                    distance_y = al_unit.pos.y - (self.target_SP_loc[pid][1] + deviation[1])
                    distance = math.sqrt(distance_x ** 2 + distance_y ** 2)
                    if distance < 2 and al_unit.health != 0:
                        self.agent_reach_point[pid][al_id] = True  # 只要到达过SP一次即可
                    if al_unit.health == 0:
                        self.agent_reach_point[pid][al_id] = None  # 不考虑死亡的单位的到达记录

                number_arrive[pid] = self.agent_reach_point[pid].count(True)  # 目前到达的单位的数量

                if number_arrive[pid] == n_ally_alive[pid] > 0:  # A platoon reach a SP.
                    arrive_reward[pid] = True

                    self.platoons_move_record[pid][self.target_SP_id[pid]] = 1  # denote the platoon has arrived the sp

                    # TODO: when training on intervals, comment these codes.
                    """
                    In the FALCON demo, even when the first platoon arrive at SP11, the target of other platoons will 
                    still be updated, until all platoons arrive SP 11.
                    """
                    if not (self.target_SP_id[pid] + 1) == len(
                            self.map_sps):  # When the platoon arrive the last SP, don't issue new SP.
                        self.target_SP[pid] = str(int(self.target_SP[pid]) + 1)  # set the target SP as the next one
                        self.target_SP_id[pid] += 1

                    self.target_SP_loc[pid] = self.map_sps[self.target_SP[pid]]  # issue the next SP.

                    # refresh the checklist of next SP
                    self.agent_reach_point[pid] = [False for _ in self.agent_reach_point[pid]]

                    if self.train_on_intervals:
                        arrive_lastSP = True

                    # Todo: [11-08] Trigger the movement of the next platoon
                    if self.FALCON_demo:
                        # self.next_movement_platoon = [1, 2, 0][pid]  # when p0 arrive at the SP, only p1 will move
                        if pid == 0:
                            self.next_movement_platoon = 2
                        else:
                            self.next_movement_platoon = pid - 1

            arrive_sp1 = False
            check_arrive_sp1 = [i[0] for i in self.platoons_move_record]
            if check_arrive_sp1.count(1) == self.n_ally_platoons:
                arrive_sp1 = True

            arrive_sp2 = False
            check_arrive_sp2 = [i[1] for i in self.platoons_move_record]
            if check_arrive_sp2.count(1) == self.n_ally_platoons:
                arrive_sp2 = True

            arrive_sp3 = False
            check_arrive_sp3 = [i[2] for i in self.platoons_move_record]
            if check_arrive_sp3.count(1) == self.n_ally_platoons:
                arrive_sp3 = True

            arrive_sp4 = False
            check_arrive_sp4 = [i[3] for i in self.platoons_move_record]
            if check_arrive_sp4.count(1) == self.n_ally_platoons:
                arrive_sp4 = True

            arrive_sp5 = False
            check_arrive_sp5 = [i[4] for i in self.platoons_move_record]
            if check_arrive_sp5.count(1) == self.n_ally_platoons:
                arrive_sp5 = True

            arrive_sp6 = False
            check_arrive_sp6 = [i[5] for i in self.platoons_move_record]
            if check_arrive_sp6.count(1) == self.n_ally_platoons:
                arrive_sp6 = True

            arrive_sp7 = False
            check_arrive_sp7 = [i[6] for i in self.platoons_move_record]
            if check_arrive_sp7.count(1) == self.n_ally_platoons:
                arrive_sp7 = True

            arrive_sp8 = False
            check_arrive_sp8 = [i[7] for i in self.platoons_move_record]
            if check_arrive_sp8.count(1) == self.n_ally_platoons:
                arrive_sp8 = True
            #
            # arrive_sp9 = False
            # check_arrive_sp9 = [i[8] for i in self.platoons_move_record]
            # if check_arrive_sp9.count(1) == self.n_ally_platoons:
            #     arrive_sp9 = True
            #
            # arrive_sp10 = False
            # check_arrive_sp10 = [i[9] for i in self.platoons_move_record]
            # if check_arrive_sp10.count(1) == self.n_ally_platoons:
            #     arrive_sp10 = True

            #
            # arrive_sp11 = False
            # check_arrive_sp11 = [i[10] for i in self.platoons_move_record]
            # if check_arrive_sp11.count(1) == self.n_ally_platoons:
            #     arrive_sp11 = True

            # todo change the win criteria here.

            if self.map_name == '4t_vs_0t_8SPs':
                if self._episode_steps == self.episode_limit and not arrive_sp8:
                    return -1, arrive_reward  # lost
            elif self.map_name == '4t_vs_0t_8SPs_randomized':
                if self._episode_steps == self.episode_limit and not arrive_lastSP:
                    return -1, arrive_reward  # lost
            elif self.map_name in ['4t_vs_0t_8SPs_RandomEnemy', '4t_vs_0t_8SPs_RandomEnemy_075']:
                if (self._episode_steps == self.episode_limit and not arrive_sp8) or \
                        (sum(n_ally_alive) == 0 and n_enemy_alive > 0 or self.only_medivac_left(ally=True)):
                    return -1, arrive_reward  # lost
            else:
                if (sum(n_ally_alive) == 0 and n_enemy_alive > 0
                        or self.only_medivac_left(ally=True)):
                    return -1, arrive_reward  # lost

            # if (sum(n_ally_alive) > 0 and n_enemy_alive == 0 and arrive_sp1  # TODO: change the required SP to arrive.
            #         or self.only_medivac_left(ally=False)):
            #     return 1, arrive_reward  # won

            if (sum(n_ally_alive) > 0 and (arrive_sp8 or arrive_lastSP)  # and n_enemy_alive == 0
                    or self.only_medivac_left(ally=False)):
                return 1, arrive_reward  # won

            if sum(n_ally_alive) == 0 and n_enemy_alive == 0:
                return 0, arrive_reward

        else:
            # update the alive agents
            for al_id, al_unit in self.agents.items():
                updated = False
                for unit in self._obs.observation.raw_data.units:
                    if al_unit.tag == unit.tag:
                        self.agents[al_id] = unit
                        updated = True
                        n_ally_alive += 1
                        break
                if not updated:  # dead
                    al_unit.health = 0

            # Update the status of enemies
            for e_id, e_unit in self.enemies.items():
                updated = False
                for unit in self._obs.observation.raw_data.units:
                    if e_unit.tag == unit.tag:
                        self.enemies[e_id] = unit
                        updated = True
                        n_enemy_alive += 1
                        break

                if not updated:  # dead
                    e_unit.health = 0

            # TODO: These lines defines the game-end signal
            # The return values here need to be changed.
            if (n_ally_alive == 0 and n_enemy_alive > 0
                    or self.only_medivac_left(ally=True)):
                return -1  # lost

            # HERE to examine if the agents are at the target point
            # ---------------------------------------------------------------------------------------
            # agent_reach_point = True
            # for al_id, al_unit in self.agents.items():
            #     distance = math.sqrt((al_unit.pos.x - self.reward_StrategicPoint_loc[0]) ** 2 + (
            #                 al_unit.pos.y - self.reward_StrategicPoint_loc[1]) ** 2)
            #     if distance > 3:
            #         agent_reach_point = False
            # if (n_ally_alive > 0 and n_enemy_alive == 0 and agent_reach_point == True
            #         or self.only_medivac_left(ally=False)):
            #     return 1  # won
            # ---------------------------------------------------------------------------------------

            # ---------------------------------------------------------------------------------------
            # 2021-09-11: Update the evaluation of winning the game
            for al_id, al_unit in self.agents.items():
                distance = math.sqrt((al_unit.pos.x - self.reward_StrategicPoint_loc[0]) ** 2 + (
                        al_unit.pos.y - self.reward_StrategicPoint_loc[1]) ** 2)
                if distance < 5 and al_unit.health != 0:
                    self.agent_reach_point[al_id] = True  # update arrival information

            number_arrive = self.agent_reach_point.count(True)

            if (n_ally_alive > 0 and n_enemy_alive == 0 and number_arrive == n_ally_alive
                    or self.only_medivac_left(ally=False)):
                return 1  # won
            # ---------------------------------------------------------------------------------------

            # origin
            # if (n_ally_alive > 0 and n_enemy_alive == 0
            #        or self.only_medivac_left(ally=False)):
            #    return 1  # won

            if n_ally_alive == 0 and n_enemy_alive == 0:
                return 0

        return None, arrive_reward

    def update_units(self):
        """Update units after an environment step.
        This function assumes that self._obs is up-to-date.
        """
        # TODO update the units after the movement in env.
        n_ally_alive = 0
        n_enemy_alive = 0

        # Store previous state
        # TODO Store previous company statys
        if self.hierarchical:
            self.previous_ally_platoons = deepcopy(self.ally_platoons)
            self.previous_enemy_units = deepcopy(self.enemies)
        else:
            self.previous_ally_units = deepcopy(self.agents)
            self.previous_enemy_units = deepcopy(self.enemies)

        if self.hierarchical:
            for platoon in self.ally_platoons:
                for al_id, al_unit in enumerate(platoon):
                    updated = False
                    for unit in self._obs.observation.raw_data.units:
                        if al_unit.tag == unit.tag:
                            platoon[al_id] = unit
                            updated = True
                            n_ally_alive += 1
                            break
                    if not updated:  # dead
                        al_unit.health = 0
        else:
            for al_id, al_unit in self.agents.items():
                updated = False
                for unit in self._obs.observation.raw_data.units:
                    if al_unit.tag == unit.tag:
                        self.agents[al_id] = unit
                        updated = True
                        n_ally_alive += 1
                        break

                if not updated:  # dead
                    al_unit.health = 0

        for e_id, e_unit in self.enemies.items():
            updated = False
            for unit in self._obs.observation.raw_data.units:
                if e_unit.tag == unit.tag:
                    self.enemies[e_id] = unit
                    updated = True
                    n_enemy_alive += 1
                    break
            if not updated:  # dead
                e_unit.health = 0

        # TODO: These lines defins the game-end signal
        # The return values here need to be changed.
        if (n_ally_alive == 0 and n_enemy_alive > 0
                or self.only_medivac_left(ally=True)):
            return -1  # lost

        # HERE to examine if the agents are at the target point
        # ---------------------------------------------------------------------------------------
        # agent_reach_point = True
        # for al_id, al_unit in self.agents.items():
        #     distance = math.sqrt((al_unit.pos.x - self.reward_StrategicPoint_loc[0]) ** 2 + (
        #                 al_unit.pos.y - self.reward_StrategicPoint_loc[1]) ** 2)
        #     if distance > 3:
        #         agent_reach_point = False
        # if (n_ally_alive > 0 and n_enemy_alive == 0 and agent_reach_point == True
        #         or self.only_medivac_left(ally=False)):
        #     return 1  # won
        # ---------------------------------------------------------------------------------------

        # ---------------------------------------------------------------------------------------
        # 2021-09-11: Update the evaluation of winning the game
        for al_id, al_unit in self.agents.items():
            distance = math.sqrt((al_unit.pos.x - self.reward_StrategicPoint_loc[0]) ** 2 + (
                    al_unit.pos.y - self.reward_StrategicPoint_loc[1]) ** 2)
            if distance < 5 and al_unit.health != 0:
                self.agent_reach_point[al_id] = True  # update arrival information

        number_arrive = self.agent_reach_point.count(True)

        if (n_ally_alive > 0 and n_enemy_alive == 0 and number_arrive == n_ally_alive
                or self.only_medivac_left(ally=False)):
            return 1  # won
        # ---------------------------------------------------------------------------------------

        # origin
        # if (n_ally_alive > 0 and n_enemy_alive == 0
        #        or self.only_medivac_left(ally=False)):
        #    return 1  # won

        if n_ally_alive == 0 and n_enemy_alive == 0:
            return 0

        return None

    def _init_ally_unit_types(self, min_unit_type):
        """Initialise ally unit types. Should be called once from the
        init_units function.
        """
        self._min_unit_type = min_unit_type
        if self.map_type == "marines":
            self.marine_id = min_unit_type
        elif self.map_type == "stalkers_and_zealots":
            self.stalker_id = min_unit_type
            self.zealot_id = min_unit_type + 1
        elif self.map_type == "colossi_stalkers_zealots":
            self.colossus_id = min_unit_type
            self.stalker_id = min_unit_type + 1
            self.zealot_id = min_unit_type + 2
        elif self.map_type == "MMM":
            self.marauder_id = min_unit_type
            self.marine_id = min_unit_type + 1
            self.medivac_id = min_unit_type + 2
        elif self.map_type == "zealots":
            self.zealot_id = min_unit_type
        elif self.map_type == "hydralisks":
            self.hydralisk_id = min_unit_type
        elif self.map_type == "stalkers":
            self.stalker_id = min_unit_type
        elif self.map_type == "colossus":
            self.colossus_id = min_unit_type
        elif self.map_type == "bane":
            self.baneling_id = min_unit_type
            self.zergling_id = min_unit_type + 1

    def only_medivac_left(self, ally):
        """Check if only Medivac units are left."""
        if self.map_type != "MMM":
            return False

        if ally:
            units_alive = [
                a
                for a in self.agents.values()
                if (a.health > 0 and a.unit_type != self.medivac_id)
            ]
            if len(units_alive) == 0:
                return True
            return False
        else:
            units_alive = [
                a
                for a in self.enemies.values()
                if (a.health > 0 and a.unit_type != self.medivac_id)
            ]
            if len(units_alive) == 1 and units_alive[0].unit_type == 54:
                return True
            return False

    def get_unit_by_id(self, a_id):
        """Get unit by ID."""
        return self.agents[a_id]

    def get_platoon_unit_by_id(self, a_id, p_id):
        """
        Retrieve a unit agent from the a given platoon.
        """
        # return self.agents[a_id]
        return self.ally_platoons[p_id][a_id]

    def get_platoon_by_id(self, p_id):
        """Get unit by ID."""
        return self.ally_platoons[p_id]

    def get_stats(self):
        stats = {
            "battles_won": self.battles_won,
            "battles_game": self.battles_game,
            "battles_draw": self.timeouts,
            "win_rate": self.battles_won / self.battles_game,
            "timeouts": self.timeouts,
            "restarts": self.force_restarts,
        }
        return stats

    ##---------- following are the functions defined by Geng, Minghong -------------------------------------------------
    def get_platoon_current_sp(self, platoon):  # update on 2022-03-16
        agent_nearest_sp = [self.nearest_sp(agent, self.map_sps) for agent in platoon]
        if len(set(agent_nearest_sp)) == 1:
            platoon_current_sp = agent_nearest_sp[0]
        else: # 2 agents have different closest SP
            platoon_current_sp = max(agent_nearest_sp, key=agent_nearest_sp.count)
        return platoon_current_sp

    def nearest_sp(self, agent, SPs):
        distance_to_SPs = [0] * len(SPs)
        for SP in SPs.items():
            distance_to_SP = self.distance(SP[1][0], SP[1][1], agent.pos.x, agent.pos.y)
            #print(distance_to_SPs)
            distance_to_SPs[int(SP[0])] = distance_to_SP
        min_distance = min(distance_to_SPs)
        nearest_sp = distance_to_SPs.index(min_distance)
        return nearest_sp

    def get_target_SP_id(self, platoon_SPs):
        target_SPs = []
        for platoon_SP in platoon_SPs:
            if platoon_SP == 0:
                target_SP = 1
            elif platoon_SP == len(self.map_sps) - 1:
                target_SP = platoon_SP - 1
            else:
                target_SP = np.random.choice(np.array([platoon_SP - 1, platoon_SP + 1]))
            target_SPs.append(target_SP)
        return target_SPs