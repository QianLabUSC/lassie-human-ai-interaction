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
            next_order = list(state.all_orders)[0]
            
            needed_ingredients = {
                "onion": next_order.ingredients.count("onion"),
                "tomato": next_order.ingredients.count("tomato")
            }

            soups_ready_to_cook_key = "{}_items".format(
                    len(next_order.ingredients)
                )
            soups_ready_to_cook = pot_states_dict[soups_ready_to_cook_key]
            if soups_ready_to_cook:
                only_pot_states_ready_to_cook = defaultdict(list)
                only_pot_states_ready_to_cook[
                    soups_ready_to_cook_key
                ] = soups_ready_to_cook
                # we want to cook only soups that has same len as order
                motion_goals = am.start_cooking_actions(
                    only_pot_states_ready_to_cook
                )
            else:
                # check the current 2 items key and 1 items key
                if '2_items' in pot_states_dict:
                    pot_pos = pot_states_dict['2_items'][0]
                    pot_contents = state.get_object(pot_pos)
                    current_onion_number = pot_contents.ingredients.count('onion')
                    current_tomato_number = pot_contents.ingredients.count('tomato')
                elif '1_items' in pot_states_dict:
                    pot_pos = pot_states_dict['1_items'][0]
                    pot_contents = state.get_object(pot_pos)
                    current_onion_number = pot_contents.ingredients.count('onion')
                    current_tomato_number = pot_contents.ingredients.count('tomato')
                else:
                    current_onion_number = 0
                    current_tomato_number = 0
                
                still_need_onion = needed_ingredients["onion"] - current_onion_number
                still_need_tomato = needed_ingredients["tomato"] - current_tomato_number
                # Pick up needed ingredients
                if still_need_onion > still_need_tomato:
                    motion_goals = am.pickup_onion_actions(counter_objects)
                elif still_need_tomato > 0:
                    motion_goals = am.pickup_tomato_actions(counter_objects)
    else:
        player_obj = player.get_object()
        if player_obj.name == "onion":
            motion_goals = am.put_onion_in_pot_actions(pot_states_dict)
        elif player_obj.name == "tomato":
            motion_goals = am.put_tomato_in_pot_actions(pot_states_dict)
        elif player_obj.name == "dish":
            motion_goals = am.pickup_soup_with_dish_actions(
                pot_states_dict, only_nearly_ready=True
            )
        elif player_obj.name == "soup":
            motion_goals = am.deliver_soup_actions()

    motion_goals = [
        mg
        for mg in motion_goals
        if self.mlam.motion_planner.is_valid_motion_start_goal_pair(
            player.pos_and_or, mg
        )
    ]
    if len(motion_goals) == 0:
        motion_goals = am.go_to_closest_feature_actions(player)
        motion_goals = [
            mg
            for mg in motion_goals
            if self.mlam.motion_planner.is_valid_motion_start_goal_pair(
                player.pos_and_or, mg
            )
        ]
        assert len(motion_goals) != 0
    return motion_goals