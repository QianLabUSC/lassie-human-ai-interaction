
import numpy as np

from overcooked_ai_py.planning.planners import MediumLevelActionManager
from overcooked_ai_py.planning.planners import MotionPlanner
"""
    custom MediumLevelActionManager
"""
class LLMMediumLevelActionManager(MediumLevelActionManager):
    def __init__(self, mdp, mlam_params):
        self.mdp = mdp

        self.params = mlam_params
        self.wait_allowed = mlam_params["wait_allowed"]
        self.counter_drop = mlam_params["counter_drop"]
        self.counter_pickup = mlam_params["counter_pickup"]

        self.joint_motion_planner = LLMJointMotionPlanner(mdp, mlam_params)
        self.motion_planner = self.joint_motion_planner.motion_planner

    """
    extend the MediumLevelActionManager to include put_item_on_counter action
    """
    def put_item_on_specific_counter_actions(self,counter_pos):
        valid_empty_counters = [
            c_pos for c_pos in self.counter_drop if c_pos in counter_pos
        ]
        return self._get_ml_actions_for_positions(valid_empty_counters)

# the original overcooked ai implmentation assumes two agents interacting in the game. since we assume 1 agent, we used some hack to ommited lines of code that are computational intensive. 
class LLMJointMotionPlanner(object):
    """A planner that computes optimal plans for a two agents to
    arrive at goal positions and orientations in a OvercookedGridworld.

    Args:
        mdp (OvercookedGridworld): gridworld of interest
    """

    def __init__(self, mdp, params, debug=False):
        self.mdp = mdp

        # Whether starting orientations should be accounted for
        # when solving all motion problems
        # (increases number of plans by a factor of 4)
        # but removes additional fudge factor <= 1 for each
        # joint motion plan
        self.debug = debug
        self.start_orientations = params["start_orientations"]

        # Enable both agents to have the same motion goal
        self.same_motion_goals = params["same_motion_goals"]

        # Single agent motion planner
        self.motion_planner = MotionPlanner(
            mdp, counter_goals=params["counter_goals"]
        )

        # Graph problem that returns optimal paths from
        # starting positions to goal positions (without
        # accounting for orientations)
        # HACK: commented out to avoid computation
        # self.joint_graph_problem = self._joint_graph_from_grid()
        # self.all_plans = self._populate_all_plans()
