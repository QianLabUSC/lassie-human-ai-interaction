from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState, OvercookedGridworld
from overcooked_ai_py.utils import (
    OvercookedException,
    classproperty,
    pos_distance,
    read_layout_dict,
)
import copy
import itertools
import warnings
from collections import Counter, defaultdict
from functools import reduce

import numpy as np

from overcooked_ai_py.mdp.actions import Action, Direction


class LLMOvercookedGridworld(OvercookedGridworld):
    """
    custon gridworld for LLM. 
    I changed event logging
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.human_event_list = []

    @staticmethod
    def from_layout_name(layout_name, **params_to_overwrite):
        """
        Generates a OvercookedGridworld instance from a layout file.

        One can overwrite the default mdp configuration using partial_mdp_config.
        """
        params_to_overwrite = params_to_overwrite.copy()
        base_layout_params = read_layout_dict(layout_name)

        grid = base_layout_params["grid"]
        del base_layout_params["grid"]
        base_layout_params["layout_name"] = layout_name
        if "start_state" in base_layout_params:
            base_layout_params["start_state"] = OvercookedState.from_dict(
                base_layout_params["start_state"]
            )

        # Clean grid
        grid = [layout_row.strip() for layout_row in grid.split("\n")]
        return LLMOvercookedGridworld.from_grid(
            grid, base_layout_params, params_to_overwrite
        )

    @staticmethod
    def from_grid(
        layout_grid, base_layout_params={}, params_to_overwrite={}, debug=False
    ):
        """
        Returns instance of OvercookedGridworld with terrain and starting
        positions derived from layout_grid.
        One can override default configuration parameters of the mdp in
        partial_mdp_config.
        """
        mdp_config = copy.deepcopy(base_layout_params)

        layout_grid = [[c for c in row] for row in layout_grid]
        LLMOvercookedGridworld._assert_valid_grid(layout_grid)

        if "layout_name" not in mdp_config:
            layout_name = "|".join(["".join(line) for line in layout_grid])
            mdp_config["layout_name"] = layout_name

        player_positions = [None] * 9
        for y, row in enumerate(layout_grid):
            for x, c in enumerate(row):
                if c in ["1", "2", "3", "4", "5", "6", "7", "8", "9"]:
                    layout_grid[y][x] = " "

                    # -1 is to account for fact that player indexing starts from 1 rather than 0
                    assert (
                        player_positions[int(c) - 1] is None
                    ), "Duplicate player in grid"
                    player_positions[int(c) - 1] = (x, y)

        num_players = len([x for x in player_positions if x is not None])
        player_positions = player_positions[:num_players]

        # After removing player positions from grid we have a terrain mtx
        mdp_config["terrain"] = layout_grid
        mdp_config["start_player_positions"] = player_positions

        for k, v in params_to_overwrite.items():
            curr_val = mdp_config.get(k, None)
            if debug:
                print(
                    "Overwriting mdp layout standard config value {}:{} -> {}".format(
                        k, curr_val, v
                    )
                )
            mdp_config[k] = v

        return LLMOvercookedGridworld(**mdp_config)

    def log_object_potting(
        self, events_infos, state, old_soup, new_soup, obj_name, player_index
    ):
        """Player added an ingredient to a pot"""
        obj_pickup_key = "potting_" + obj_name
        if player_index == 0:
            self.human_event_list.append(obj_pickup_key)
        if obj_pickup_key not in events_infos:
            raise ValueError("Unknown event {}".format(obj_pickup_key))
        events_infos[obj_pickup_key][player_index] = True

        POTTING_FNS = {
            "optimal": self.is_potting_optimal,
            "catastrophic": self.is_potting_catastrophic,
            "viable": self.is_potting_viable,
            "useless": self.is_potting_useless,
        }

        for outcome, outcome_fn in POTTING_FNS.items():
            if outcome_fn(state, old_soup, new_soup):
                potting_key = "{}_{}_potting".format(outcome, obj_name)
                events_infos[potting_key][player_index] = True
    def log_object_pickup(
        self, events_infos, state, obj_name, pot_states, player_index
    ):
        """Player picked an object up from a counter or a dispenser"""
        obj_pickup_key = obj_name + "_pickup"
        if player_index == 0:
            self.human_event_list.append(obj_pickup_key)

        if obj_pickup_key not in events_infos:
            raise ValueError("Unknown event {}".format(obj_pickup_key))
        events_infos[obj_pickup_key][player_index] = True

        USEFUL_PICKUP_FNS = {
            "tomato": self.is_ingredient_pickup_useful,
            "onion": self.is_ingredient_pickup_useful,
            "dish": self.is_dish_pickup_useful,
        }
        if obj_name in USEFUL_PICKUP_FNS:
            if USEFUL_PICKUP_FNS[obj_name](state, pot_states, player_index):
                obj_useful_key = "useful_" + obj_name + "_pickup"
                events_infos[obj_useful_key][player_index] = True

    def log_object_drop(
        self, events_infos, state, obj_name, pot_states, player_index
    ):
        """Player dropped the object on a counter"""
        obj_drop_key = obj_name + "_drop"
        if player_index == 0:
            self.human_event_list.append(obj_drop_key)

        if obj_drop_key not in events_infos:
            raise ValueError("Unknown event {}".format(obj_drop_key))
        events_infos[obj_drop_key][player_index] = True

        USEFUL_DROP_FNS = {
            "tomato": self.is_ingredient_drop_useful,
            "onion": self.is_ingredient_drop_useful,
            "dish": self.is_dish_drop_useful,
        }
        if obj_name in USEFUL_DROP_FNS:
            if USEFUL_DROP_FNS[obj_name](state, pot_states, player_index):
                obj_useful_key = "useful_" + obj_name + "_drop"
                events_infos[obj_useful_key][player_index] = True

