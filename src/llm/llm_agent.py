from overcooked_ai_py.agents.agent import Agent, GreedyHumanModel
import itertools
import types
import numpy as np
import pygame
import re
from openai import OpenAI
from overcooked_ai_py.mdp.actions import Action
from dotenv import load_dotenv
import os
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../../.env'))
from overcooked_ai_py.mdp.actions import Action
import time
from llm.llm_api import Llama, GPT, Llama_with_ollama, Gemma2b_with_ollama, GroqClient,ReplicateClient

from llm.utils import read_from_file, write_to_file

  # implement async api call
from queue import Queue
import threading


class SharedState:
    """Class to manage shared state with thread-safe access."""
    def __init__(self):
        self._lock = threading.Lock()
        self.state = None
        self.current_agent_state = None
        self.other_agent_state = None

    def update_state(self, state):
        with self._lock:
            self.state = state


    def get_state(self):
        with self._lock:
            return self.state
        

        

#base class for LLM models
class LLMModel(GreedyHumanModel):
    def __init__(self,mlam, agent_name,action_prompt_template_with_layout, subtask_system_text, subtask_prompt_template_with_layout, reactive_model="llama", manager_model="gpt", personality=None, debug=True):
        super().__init__(mlam)
        self.agent_name = agent_name
        self.agent_log = []
        print("\nInitializing LLM models\n")

        self.debug = debug
        # self.reactive_model = self.initialize_model(reactive_model)
        self.manager_model = self.initialize_model(manager_model)

        self.action_prompt_layout = read_from_file(action_prompt_template_with_layout)
        self.subtask_prompt_layout = read_from_file(subtask_prompt_template_with_layout)
    
    def initialize_model(self, model):
        print(model)
        if model=="gpt":
            print("check")
            openai_key = os.environ.get("OPENAI_API_KEY")
            model_ = GPT(openai_key)
        elif model=="llama":
            hf_key = os.environ.get("HF_TOKEN")
            model_ = Llama(hf_key)
        elif model == "ollama":
            model_ = Llama_with_ollama()
        elif model == "gemma":
            model_ = Gemma2b_with_ollama()
        elif model == "groq":
            model_ = GroqClient()
        elif model == "replicate":
            load_dotenv()
            replicate_key = os.environ.get("REPLICATE_API_TOKEN")
            print(replicate_key)
            model_ = ReplicateClient(replicate_key)
        else:
            raise ValueError("Model not recognized")
        return model_

    def action(self, state):
        #determine action by querying the model
        raise NotImplementedError("action method must be implemented in subclass")
    
    def format_prompt_given_states(self, prompt: str, world_state, current_agent_state, other_agent_state, grid, avaiable_action=None, current_subtasks = None,task_list=None, human_preference = None):
        """format the prompt given world states

        does not format the task list

        Input:
            prompt: the layout read from the layout file

        Returns: formatted prompt
        """

        # format the prompt

        # current state
        current_state = self.get_agent_state_as_language(current_agent_state, world_state,grid,first_person=True)
        prompt = prompt.replace("{current_state}", current_state)

        # other chef state
        other_chef_state = self.get_agent_state_as_language(other_agent_state,world_state,grid, first_person=False)
        prompt = prompt.replace("{other_chef_state}", other_chef_state)


        # update kitchen state
        # kitchen_overview, kitchen_items = self.get_kitchen_as_language(world_state, current_agent_state, other_agent_state,grid)
        # prompt = prompt.replace("{kitchen_items}", kitchen_items)
        # prompt = prompt.replace("{kitchen_overview}", kitchen_overview)
        
        if current_subtasks:
            prompt = prompt.replace("{current_subtask}", current_subtasks)
        else:
            prompt = prompt.replace("{current_subtask}", "No subtask selected yet.")

        if human_preference:
            prompt = prompt.replace("{human_preferences}", human_preference)
        else:
            prompt = prompt.replace("{human_preferences}", "No preference yet.")

        if task_list:
            prompt = prompt.replace("{task_list}", task_list)
        if avaiable_action:
            prompt = prompt.replace("{feasible_action}", str(avaiable_action))
        else:
            prompt = prompt.replace("{feasible_action}", "1. move right\n2. move left\n3. move up\n4. move down\n5. interact\n6. stay")

        # other_chef_message = self.other_agent_response
        # # check if chef_message is enabled in the prompt
        # if prompt.find("{other_chef_message}") != -1:
        #     # add the chef message to the prompt
        #     prompt = prompt.replace("{other_chef_message}", other_chef_message)


        return prompt


    def get_agent_state_as_language(self, state, world_state,grid,first_person=False):
        """Construct the agent state as a string from a dictionary containing its contents
        """
        
        # pronouns resolver
        if first_person:
            pronouns = ["You are", "Your", "You"]
        else:
            pronouns = ["The other chef is", "Their", "They"]

        # held object resolver
        if state['held_object'] is not None:
            held_object = "a " + state['held_object']['name']
        else:
            held_object = "nothing"
        orientation_to_string = {
            (1,0): "right",
            (-1,0): "left",
            (0,1): "down",
            (0,-1): "up",
            (0,0): "staying"
        }
        grid_to_item = {
            "X": "counter",
            "P": "pot",
            "D": "dish dispenser",
            "O": "onion dispenser",
            "T": "tomato dispenser",
            "S": "delivery location",
            " ": "empty square"
        }
        faced_pos = (state['position'][0] + state['orientation'][0], state['position'][1] + state['orientation'][1])
        # print(f"currently facing {faced_pos}")
        faced_item = grid[faced_pos[1]][faced_pos[0]]
        # print(f"currently facing {grid_to_item[faced_item]}")
        # construct and return state string 
        faced_item_state = grid_to_item[faced_item]
        if faced_item == "X": 
            #add some additional information about the counter
            for counter in world_state:
                if counter["position"] == faced_pos:
                    faced_item_state = f"a counter, it has a {counter['name']} on it."
                    break
        elif faced_item == "P":
            #add some additional information about the pot
            for pot in world_state:
                if pot["name"] == "soup" and pot["position"] == faced_pos:
                    faced_item_state = f"a pot, {self.get_pot_state_as_language(pot)}"
                    break

        return f"""1. {pronouns[0]} at the coordinates {state['position']}
    2. {pronouns[1]} orientation is facing {orientation_to_string[state['orientation']]}
    3. {pronouns[2]} are holding {held_object}
    4. {pronouns[2]} are facing {faced_item_state}
        """            

    def get_kitchen_as_language(self, world_state, current_agent_state, other_agent_state, grid):
        """Construct the kitchen state as a string from a dictionary containing its contents
        """

        # obtain the grid layout from the mdp
        grid_layout = grid

        # construct kitchen overview
        x_dim, y_dim = len(grid_layout[0]), len(grid_layout)
        kitchen_overview = f"The kitchen is a {x_dim}x{y_dim} grid world. The top left corner is (0, 0) and the bottom right corner is ({x_dim-1}, {y_dim-1})."

        # construct kitchen items
        kitchen_items = []
        
        for i in range(len(grid_layout)):
            for j in range(len(grid_layout[i])):
                item = grid_layout[i][j]
                if item == "X":
                    item_name = "Counter"

                    item_state = "The counter is empty."

                    # resolve counter contents (not that efficient)
                    for counter in world_state:
                        if counter["position"] == (j,i):
                            item_state = f"The counter has a {counter['name']} on it."
                            break
                    
                elif item == "P":
                    item_name = "Pot"

                    pot_state = None
                    # find the pot at this position
                    for pot in world_state:
                        if pot["name"] == "soup" and pot["position"] == (j,i):
                            pot_state = pot
                            break

                    # special case resolution for pot
                    # item_state = "The pot is empty."
                    item_state = self.get_pot_state_as_language(pot_state)
                elif item == "D":
                    item_name = "Dish dispenser"
                    item_state = "The dish dispenser has infinite empty dishes."
                elif item == "O":
                    item_name = "Onion dispenser"
                    item_state = "The onion dispenser has infinite onions."
                elif item == "T":
                    item_name = "Tomato dispenser"
                    item_state = "The tomato dispenser has infinite tomatoes."
                elif item == "S":
                    item_name = "Delivery location"
                    # item_state = "The delivery location is empty."
                    item_state = ""
                else:
                    item_name = "Empty square"

                    # resolve state based on where chefs are standing
                    if current_agent_state['position'] == (j,i):
                        item_state = "You are standing here."
                    elif other_agent_state['position'] == (j,i):
                        item_state = "The other chef is currently standing here."
                    else:
                        item_state = "You can stand here."
                
                kitchen_items.append( f"\t({j},{i}): {item_name}. {item_state}" )

        # format with newline operator
        return kitchen_overview, "\n".join(kitchen_items)

    def get_pot_state_as_language(self, pot_state):
        """Construct the pot state as a string from a dictionary containing its contents
        """

        # resolve if pot has no ingredients
        if pot_state is None:
            return "The pot is empty. It has 0 ingredients."

        # obtain the pot state
        number_of_ingredients = len(pot_state['_ingredients'])
        is_ready = pot_state["is_ready"]
        is_cooking = pot_state["is_cooking"]
        cook_time = pot_state["cook_time"]   
        cooking_timer = pot_state["_cooking_tick"]
        soup_ingredients = pot_state["_ingredients"]

        ingredients = []
        for ingredient in soup_ingredients:
            ingredients.append(ingredient['name'])

        ingredients = ", ".join(ingredients)

        pot_items = f"The pot is not empty. There are already {number_of_ingredients} ingredients in the pot: {ingredients}. "

        if not is_cooking:
            if is_ready:
                pot_items += f"The soup is finished cooking. It can be picked up with a dish."
            else:
                pot_items += f"The soup has not started cooking yet."
        else:
            pot_items += f"The soup has already started cooking, but is not finished cooking. It is {cooking_timer} out of {cook_time} ticks cooked."

        return pot_items

        
    
"""
LLM model that query every atomic action
"""
class ManagerReactiveModel(LLMModel):
    def __init__(self, mlam, agent_name, action_prompt_template_with_layout, subtask_system_layout, subtask_prompt_template_with_layout, reactive_model="llama", manager_model="gpt", personality=None, debug=False):
        super().__init__(mlam, agent_name, action_prompt_template_with_layout, subtask_system_layout, subtask_prompt_template_with_layout, reactive_model, manager_model, personality, debug)
        self.agent_response = ""
        self.mode = 3

        # Use the custom queue for each type of result
        self.shared_state = SharedState()
        self.subtask_queue = []
        self.subtask_results = 10
        self.active_threads = []
        self.async_determine_subtask()
        self.action_chose = (0,0)
        self.action_status = 0    #0: done, need to call api, 1: running, need to wait, 2: updated, need to send to game
        self.lock = threading.Lock()
        # manager mind settings
        self.subtasks = {
            1: "Pick up onion",
            2: "Pick up dish",
            3: "Pick up tomato",
            4: "Pick up soup with dish",
            5: "Start cooking pot",
            6: "Place holding object on counter",
            7: "Deliver soup",
            8: "Put onion in pot ",
            9: "Put tomato in pot",
            10: "Do nothing"
        }
        self.low_level_actions = {1:"east", 2:"west", 3:"north", 4:"south", 5:"interact", 6:"stay"}

        self.agent_name = agent_name
        self.motion_goals = []
        self.agent_subtask_response = ""
        self.agent_log = []
        self.human_preference = None
        # self.other_agent_response = "Let's make some soups!"

        self.action_template_file_path = action_prompt_template_with_layout
        self.subtask_template_file_path = subtask_prompt_template_with_layout
        self.subtask_system = subtask_system_layout
        system_overview = read_from_file(self.subtask_template_file_path)
        self.manager_id, initial_reply = self.manager_model.start_new_chat(system_overview)

        self.arrow = pygame.image.load('./data/graphics/arrow.png')
        self.arrow = pygame.transform.scale(self.arrow, (15, 30))
        self.stay = pygame.image.load('./data/graphics/stay.png')
        self.stay = pygame.transform.scale(self.stay, (10, 20))
        self.target = pygame.image.load('./data/graphics/target.png')
        self.target = pygame.transform.scale(self.target, (20, 20))
        self.interact = pygame.image.load('./data/graphics/interact.png')
        self.interact = pygame.transform.scale(self.interact, (20, 20))
        # self.personality = "You are a helpful assistant."

        # if personality:
        #     # if a personality was provided, try to import from the personality subdirectory
        #     try:
        #         personality_prompt = read_from_file(f"llm/personality/{personality}.txt")
        #         self.personality = personality_prompt
        #     except Exception as e:
        #         print(f"Personality not found: {personality}")
        self.debug = False

    def get_subtasks(self):
        """Process the latest results from the subtask queues."""
        with threading.Lock():
            if not self.subtask_queue:
                return 10
            else:
                # print('check if values')
                # print(self.subtask_queue[0])
                return self.subtask_queue[0]
    def set_human_preference(self, human_preference):
        """Set the human preference for the agent."""
        self.human_preference = human_preference

    def update_state(self, state):
        """Update the shared state."""
        self.shared_state.update_state(state)

    def stop_all_threads(self):
        """Stops all running threads. Threads should be designed to stop safely."""
        for thread in self.active_threads:
            thread.join()  # Ensuring threads have completed

    

    def action(self, state, screen):
        #ml_action is determined by llm 
        self.update_state(state)
        possible_motion_goals, if_stay = self.update_ml_action(state)
   
        start_pos_and_or = state.players_pos_and_or[self.agent_index]
        # if possible_motion_goals:
        if start_pos_and_or == possible_motion_goals[0] and (self.subtask_queue):
            with threading.Lock():
                print("finished current task")
                if self.subtask_queue[0] != 10:
                    self.subtask_queue.pop(0)
        # Once we have identified the motion goals for the medium
        # level action we want to perform, select the one with lowest cost
        if if_stay:
            chosen_action = (0, 0)
            action_probs = 1
        else:
            chosen_goal, chosen_action, action_probs = self.choose_motion_goal(
                start_pos_and_or, possible_motion_goals
            )
            # print(chosen_action)

            if (
                self.ll_boltzmann_rational
                and chosen_goal[0] == start_pos_and_or[0]
            ):
                chosen_action, action_probs = self.boltzmann_rational_ll_action(
                    start_pos_and_or, chosen_goal
                )

            if self.auto_unstuck:
                # HACK: if two agents get stuck, select an action at random that would
                # change the player positions if the other player were not to move
                if (
                    self.prev_state is not None
                    and state.players_pos_and_or
                    == self.prev_state.players_pos_and_or
                ):
                    if self.agent_index == 0:
                        joint_actions = list(
                            itertools.product(Action.ALL_ACTIONS, [Action.STAY])
                        )
                    elif self.agent_index == 1:
                        joint_actions = list(
                            itertools.product([Action.STAY], Action.ALL_ACTIONS)
                        )
                    else:
                        raise ValueError("Player index not recognized")

                    unblocking_joint_actions = []
                    for j_a in joint_actions:
                        new_state, _ = self.mlam.mdp.get_state_transition(
                            state, j_a
                        )
                        if (
                            new_state.player_positions
                            != self.prev_state.player_positions
                        ):
                            unblocking_joint_actions.append(j_a)
                    # Getting stuck became a possiblity simply because the nature of a layout (having a dip in the middle)
                    if len(unblocking_joint_actions) == 0:
                        unblocking_joint_actions.append([Action.STAY, Action.STAY])
                    chosen_action = unblocking_joint_actions[
                        np.random.choice(len(unblocking_joint_actions))
                    ][self.agent_index]
                    action_probs = self.a_probs_from_action(chosen_action)

                # NOTE: Assumes that calls to the action method are sequential
                self.prev_state = state

        return chosen_action, {"action_probs": action_probs}

    def ml_action(self,state):
        return self.motion_goals

    def update_ml_action(self, state):
        """select a medium level aciton for the agent's current state by querying an LLM
        """
        if_stay =  False
        if self.debug:
            print(f"updating ML action for {self.agent_name}")
       
        # obtain relevant state information
        player = state.players[self.agent_index]

        # get counter objects and pot states
        am = self.mlam
        counter_objects = self.mlam.mdp.get_counter_objects_dict(
            state, list(self.mlam.mdp.terrain_pos_dict["X"])
        )
        pot_states_dict = self.mlam.mdp.get_pot_states(state)

        
        subtask_index = self.get_subtasks()
        # print(subtask_index)

        # NOTE: new logging system: append the subtask index to the agent log
        self.agent_log.append(subtask_index)
        # construct MLAM motion goals based on the subtask index
        if subtask_index == 1: 
            motion_goals = am.pickup_onion_actions(counter_objects)
            # self.agent_log += "1. Picked up onion "
        elif subtask_index == 2:
            motion_goals = am.pickup_dish_actions(counter_objects)
            # self.agent_log += "2. Picked up dish "
        elif subtask_index == 3:
            motion_goals = am.pickup_tomato_actions(counter_objects)
            # self.agent_log += "3. Picked up tomato "
        elif subtask_index == 4:
            motion_goals = am.pickup_soup_with_dish_actions(pot_states_dict)
            # self.agent_log += "4. Picked up soup with dish "
        elif subtask_index == 5:
            motion_goals = am.start_cooking_actions(pot_states_dict)
            # self.agent_log += "5. Started cooking the pot "
        elif subtask_index == 6:
            motion_goals = am.place_obj_on_counter_actions(state)
            # self.agent_log += "6. Placed object on counter "
        elif subtask_index == 7:
            motion_goals = am.deliver_soup_actions()
            # self.agent_log += "7. Delivered soup "
        elif subtask_index == 8:
            motion_goals = am.put_onion_in_pot_actions(pot_states_dict)
            # self.agent_log += "8. Added onion to the pot "
        elif subtask_index == 9:
            motion_goals = am.put_tomato_in_pot_actions(pot_states_dict)
            # self.agent_log += "9. Added tomato to the pot "
        elif subtask_index == 10:
            motion_goals = am.wait_actions(player)
            if_stay = True
            # self.agent_log += "10. Did nothing "
        else:
            raise ValueError(f"Index {subtask_index} not found in subtasks")
        

        # filter out invalid motion goals
        motion_goals = [
            mg
            for mg in motion_goals
            if self.mlam.motion_planner.is_valid_motion_start_goal_pair(
                player.pos_and_or, mg
            )
        ]
        # print("check if empty:", motion_goals)
        # if no valid motion goals, go to the closest feature
        if motion_goals == []:
             motion_goals = am.go_to_closest_feature_actions(player)
             motion_goals = [mg for mg in motion_goals if self.mlam.motion_planner.is_valid_motion_start_goal_pair(player.pos_and_or, mg)]
        
        # update the motion goals
        self.motion_goals = motion_goals

        if self.debug:
            print(f"Motion Goals for {self.agent_name}:",motion_goals)

        return motion_goals, if_stay
        

    def render_action(self, pygame_surface, action_index,  origin_pos, orientation):
        center_x = origin_pos[0] * 30 + 15
        center_y = origin_pos[1] * 30 + 15
        arrow = self.arrow
        stay = self.stay
        interaction = self.interact
        # Rotate and position the arrow based on the direction
        if action_index == 1:  # right
            arrow_rotated = pygame.transform.rotate(arrow, -90)  # No rotation needed, assuming arrow points right by default
            center_x += 30
        elif action_index == 2:  # left
            arrow_rotated = pygame.transform.rotate(arrow, 90)  # Rotate 180 degrees to point right
            center_x -= 30
        elif action_index == 3:  # up
            arrow_rotated = pygame.transform.rotate(arrow, 0)  # Rotate 90 degrees counter-clockwise to point up
            center_y -= 30
        elif action_index == 4:  # down
            arrow_rotated = pygame.transform.rotate(arrow, 180)  # Rotate 90 degrees clockwise to point down
            center_y += 30
        elif action_index == 6:  #stay
            arrow_rotated = stay
            center_x = center_x
        elif action_index == 5: #interact
            arrow_rotated = interaction
            center_x += 30 * orientation[0]
            center_y += 30 * orientation[1]


        rect = arrow_rotated.get_rect(center=(center_x, center_y))
        pygame_surface.blit(arrow_rotated, rect.topleft)

    def render_target(self, pygame_surface, pos):
        center_x = pos[0] * 30 + 15
        center_y = pos[1] * 30 + 15
        target = self.target
        rect = target.get_rect(center=(center_x, center_y))
        pygame_surface.blit(target, rect.topleft)


    def get_graphics_repre_with_subtasks(self, state, player, current_subtasks, screen, grid, avaiable_action):
        grid_x_num = len(grid[0])
        grid_y_num = len(grid)
        rect = pygame.Rect(0, 140, grid_x_num*30+30, grid_y_num*30+40)
        mdp_surface = screen.subsurface(rect)
        # get the current agent position
        agent_state = player
        for action in avaiable_action:
            self.render_action(mdp_surface, action, (agent_state["position"] * 30), player['orientation'])
        # self.render_arrow(mdp_surface, 1, (agent_state["position"] * 30))
        # find the interaction position based on subgoal. 
        # there should be a function finding the subgoal based on subtask
        # or we could let manager mind directly output
        subgoal = "D"  # D P T S P X
        index = np.where(np.array(grid) == subgoal)
        for i in range(len(index[0])):
            self.render_target(mdp_surface, [index[1][i], index[0][i]] * 30)

            #    elif item == "D":
            #         item_name = "Dish dispenser"
            #         item_state = "The dish dispenser has infinite empty dishes."
            #     elif item == "O":
            #         item_name = "Onion dispenser"
            #         item_state = "The onion dispenser has infinite onions."
            #     elif item == "T":
            #         item_name = "Tomato dispenser"
            #         item_state = "The tomato dispenser has infinite tomatoes."
            #     elif item == "S":
            #         item_name = "Delivery location"
        import datetime
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f'input_{current_time}.png'
        pygame.image.save(mdp_surface, filename)
 
    

    def get_avaiable_action(self, state, player, other_player,all_action=False):
        if all_action:
            return [1,2,3,4,5,6]
        held_object = player['held_object']
        feasible_action = [1,2,3,4,5,6]
        player['position']
        player['orientation']
        grid_layout = self.mdp.terrain_mtx
        player['position']
        # check if it can do interact
        interact_position = (player['position'][0] + player['orientation'][0], 
                                player['position'][1] + player['orientation'][1])
        # print(grid_layout)
        # print(interact_position)
        object = grid_layout[interact_position[1]][interact_position[0]]
        if held_object:       
            if object == 'O' or object == "D":
                feasible_action.remove(5)
        else:
            if object == 'X' or object == ' ':
                feasible_action.remove(5)
        # check the four direction is illegal
        west_position = (player['position'][0] - 1, player['position'][1])
        east_posiiton = (player['position'][0] + 1, player['position'][1])
        north_position = (player['position'][0], player['position'][1] - 1)
        south_position = (player['position'][0], player['position'][1] + 1)

        left_object = grid_layout[west_position[1]][west_position[0]]
        right_object = grid_layout[east_posiiton[1]][east_posiiton[0]]
        up_object = grid_layout[north_position[1]][north_position[0]]
        down_object = grid_layout[south_position[1]][south_position[0]]

        if right_object != ' ' and player['orientation'] == (1,0):
            feasible_action.remove(1)
        if left_object != ' ' and player['orientation'] == (-1,0):
            feasible_action.remove(2)
        if up_object != ' ' and player['orientation'] == (0,-1):
            feasible_action.remove(3)
        if down_object != ' ' and player['orientation'] == (0,1):
            feasible_action.remove(4)

        # print(feasible_action)



        return feasible_action

        
    def determine_action(self,state, current_subtasks, screen):
        """query the model for the optimal atomic action

        Return: action chosen by the model, and the action probabilities
        """ 
        # relevant state information
        grid = self.mdp.terrain_mtx
        
        player = state.players[self.agent_index].to_dict()
        other_player = state.players[1 - self.agent_index].to_dict()
        world_state = state.to_dict().pop("objects")

        avaiable_action = self.get_avaiable_action(world_state, player, other_player,all_action=False)
        image_repre = self.get_graphics_repre_with_subtasks(world_state, player, current_subtasks, screen, grid, avaiable_action)
        action_list = {1:"move right", 2:"move left", 3:"move up", 4:"move down", 5:"interact", 6:"stay"}

        formatted_avaiable_action = "\n".join([f"{i}. {action_list[i]}" for i in avaiable_action])
        prompt_layout = read_from_file(self.action_template_file_path)
        
        # format prompt layout given the current states (this will replace all the placeholders in the prompt layout)
        prompt = self.format_prompt_given_states(prompt_layout, world_state, player, other_player, grid, avaiable_action=formatted_avaiable_action,current_subtasks=current_subtasks, human_preference=self.human_preference)
        # log the prompt generated
        write_to_file(f"llm/log/reactive_mind_prompt_generated_{self.agent_name}.txt", prompt)

        response = self.reactive_model.query(prompt, temp=0.4)
        # response = "analysis(optional) + [1]"
        # parse text surrounded by bracket from response
        action_index = int(re.search(r'\[(.*?)\]', response).group(1))        
        if self.debug:
            print("**********Low level*************")
            print(self.agent_name + ": " + response)
            print("*********^Low level^************")
        if action_index not in self.low_level_actions:
            raise ValueError(f"Index {action_index} not found in low level actions")
        # action index to action mapping
        action_dict = {1: (1,0), 2:(-1,0), 3:(0,-1), 4:(0,1), 5:"interact", 6:(0,0)}
        self.agent_response =f"I chose {action_index}, {self.low_level_actions[action_index]}"
        print(f"ReactiveMind: chose {action_index}, which is {self.low_level_actions[action_index]}")
        return action_dict[action_index]


    def async_determine_subtask(self):
        """Function to asynchronously determine subtask."""
        def task(shared_state):
            while True:
                state = shared_state.get_state()
                # print(state)
                if state is None:
                    time.sleep(0.1)  # Sleep briefly if no state is available
                    continue
                if not (self.subtask_queue):
                    print("querying")
                    self.subtask_results = self.determine_subtask(state)
                    # print(self.subtask_results)
                    self.subtask_queue.append(self.subtask_results)
                    print("current subtasks queue:")
                    for task in self.subtask_queue:
                        print(self.subtasks[task])
                    print("\n")
                time.sleep(0.1)

        thread = threading.Thread(target=task, args=(self.shared_state, ))
        thread.daemon = True
        thread.start()
        self.active_threads.append(thread)

    def determine_subtask(self, state):
        """query the model appropriately for the optimal subtask

        Return: index of the subtask as defined according to self.subtasks
        """
        current_agent_state = state.players[self.agent_index].to_dict()
        other_agent_state = state.players[1 - self.agent_index].to_dict()
        world_state = state.to_dict().pop("objects")

        # obtain prompt layout from file
        prompt_layout = read_from_file(self.subtask_template_file_path)

        task_list, cross_reference = self.get_relevant_subtasks(current_agent_state)

        formatted_task_list = "\n".join([f"\Option {i}: {task}" for i, task in enumerate(task_list, 1)])
        grid = self.mdp.terrain_mtx
        # format prompt layout given the current states (this will replace all the placeholders in the prompt layout)
        prompt = self.format_prompt_given_states(prompt_layout, world_state, current_agent_state, other_agent_state, grid= grid,task_list=formatted_task_list, current_subtasks= self.subtasks[self.subtask_results], human_preference=self.human_preference)
        print(prompt)
        # message_to_other_chef = "Happy to work with you!"

        # query the model given prompt
        response = self.manager_model.query(self.manager_id, "user", prompt, temp=0.4)
        
        # log the prompt generated
        write_to_file(f"llm/log/manager_mind_prompt_generated_{self.agent_name}.txt", prompt)

        if self.debug:
            print("**********manager*************")
            print(self.agent_name + ": " + response)
            print("********************************")
            # print("Other Agent: ", self.other_agent_response)

        # parse response for subtask index and cross reference index to subtasks list
        try: 
            subtask_index = int(re.search(r'\[(.*?)\]', response).group(1))        
        except Exception as e:
            print("Could not find response when parsing")
            subtask_index = 0
        
        subtask_index = cross_reference[subtask_index - 1]
        # print(f"ManagerMind:  selected {subtask_index}, {self.subtasks[subtask_index]}")
        
        # Visual conversation
        # self.agent_subtask_response = f"I selected {subtask_index}, {self.subtasks[subtask_index]}"
        # self.agent_subtask_response = message_to_other_chef
        return subtask_index

    def get_relevant_subtasks(self, current_agent_state):
        """obtain the relevant subtasks given the current agent state

        Return: 
            1) list of subtask indices
            2) indices that cross reference to valid subtask indices in self.subtasks
        """

        # get current held object
        held_object = current_agent_state['held_object']

        if held_object is None:
            # remove subtask that need a held object
            # availlable_subtasks = "1. Pick up onion\n2. Pick up dish\n3. Pick up tomato\n5. Start cooking pot\n10. Do nothing"
            cross_reference = [1, 2, 3, 5]
            task_list = ["Pick up the nearest onion", "Pick up the nearest dish", "Pick up tomato", "Start cooking the nearest pot"]
        else:
            # construct task list based on held object, and add to cross reference with switch case
            task_list = [f"Place {held_object['name']} on the nearest counter"]

            # construct cross reference list

            if held_object['name'] == "onion":
                # remove subtask that cant do with an onion held
                task_list.append(f"Put the {held_object['name']} in the nearest pot")
                cross_reference = [6, 8]
            elif held_object['name'] == "tomato":
                task_list.append(f"Put the {held_object['name']} in the nearest pot")
                cross_reference = [6, 9]

            elif held_object['name'] == "dish":
                # remove subtask that cant do with a dish held
                task_list.append("Pick up soup from the nearest pot with a dish")
                cross_reference = [6, 4]

            elif held_object['name'] == "soup":
                # remove subtask that cant do with a soup held
                task_list.append("Deliver the soup you are holding to the delivery location")
                cross_reference = [6, 7]
            
            
            # TODO: add back bottom once tomato is reintroduced
            # elif held_object['name'] == "tomato":
                # remove subtask that cant do with a tomato held
                # availlable_subtasks = "6. Place object on counter\n9. Put tomato in pot\n10. Do nothing"


        # add the do nothing action at the end (applies to every action subset)
        cross_reference.append(10)
        task_list.append("Do nothing")

        
        return task_list, cross_reference

    def pause_manager_mind(self):
        #TODO: pause threads
        pass
    def resume_manager_mind(self):
        #TODO:pause threads
        pass