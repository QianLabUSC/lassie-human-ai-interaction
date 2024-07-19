from collections import defaultdict
def ml_action(self, state):
    player = state.players[self.agent_index]
    other_player = state.players[1 - self.agent_index]
    am = self.mlam
    counter_objects = self.mlam.mdp.get_counter_objects_dict(
        state, list(self.mlam.mdp.terrain_pos_dict["X"])
    )
    pot_states_dict = self.mlam.mdp.get_pot_states(state)
    motion_goals = []
    if not player.has_object():
        ready_soups = pot_states_dict["ready"]
        cooking_soups = pot_states_dict["cooking"]
        soup_nearly_ready = len(ready_soups) > 0 or len(cooking_soups) > 0
        other_has_dish = (
            other_player.has_object() and
            other_player.get_object().name == "dish"
        )

        if soup_nearly_ready and not other_has_dish:
            motion_goals = am.pickup_dish_actions(counter_objects)
        else:
            # Stay in row 1 and do nothing if no cooked soup is available
            motion_goals = []
    else:
        player_obj = player.get_object()
        if player_obj.name == "dish":
            # Place dish on any available counter in row 1
            empty_counters = self.mlam.mdp.get_empty_counter_locations(state)
            row_1_counters = [pos for pos in empty_counters if pos[1] == 1]
            if row_1_counters:
                motion_goals = [am.put_item_on_specific_counter_actions([row_1_counters[0]])]
            else:
                # Stay in row 1 if no empty counter is available
                motion_goals = []
        elif player_obj.name == "soup":
            motion_goals = am.deliver_soup_actions()
        else:
            # Drop any other items on the nearest empty counter in row 1
            empty_counters = self.mlam.mdp.get_empty_counter_locations(state)
            row_1_counters = [pos for pos in empty_counters if pos[1] == 1]
            if row_1_counters:
                motion_goals = [am.put_item_on_specific_counter_actions([row_1_counters[0]])]
            else:
                # Stay in row 1 if no empty counter is available
                motion_goals = []

    motion_goals = [
        mg
        for mg in motion_goals
        if self.mlam.motion_planner.is_valid_motion_start_goal_pair(
            player.pos_and_or, mg
        )
    ]
    if len(motion_goals) == 0:
        # Stay in row 1 if no valid motion goals
        motion_goals = [(player.pos_and_or, player.pos_and_or)]
    return motion_goals