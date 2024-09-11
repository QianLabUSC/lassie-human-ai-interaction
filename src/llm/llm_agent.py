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
from llm.llm_api import Llama, GPT, Llama_with_ollama, Gemma2b_with_ollama, GroqClient,ReplicateClient, Llava7b_with_ollma,ReplicateClientLlava,SGLang, reactiveReasoning,managerReasoning, rule
from llm.utils import read_from_file, write_to_file

  # implement async api call
from queue import Queue
import threading

#import random
#base class for LLM models
class LLMModel(GreedyHumanModel):
    def __init__(self,agent_name,action_system_layout,action_prompt_template_with_layout,subtask_system_layout, subtask_prompt_template_with_layout, env, mlam, reactive_model="llama", manager_model="gpt", personality=None, debug=True):
        super().__init__(mlam)
        self.env = env
        self.agent_name = agent_name
        self.agent_log = []
        print("\nInitializing LLM models\n")

        self.debug = debug
        self.reactive_model = self.initialize_model(reactive_model)
        self.manager_model = self.initialize_model(manager_model)

        self.action_prompt_layout = read_from_file(action_prompt_template_with_layout)
        self.subtask_prompt_layout = read_from_file(subtask_prompt_template_with_layout)
    
    def initialize_model(self, model):
        print(f"Initializing {model} model")
        if model=="gpt":
            openai_key = os.environ.get("OPENAI_API_KEY")
            # print(openai_key)
            model_ = GPT(openai_key)
        elif model=="gpt_mini":
            openai_key = os.environ.get("OPENAI_API_KEY")
            # print(openai_key)
            model_ = GPT(openai_key, model_name="gpt-4o-mini-2024-07-18")
        elif model== "rule":
            model_ = rule()
        elif model=="llama":
            hf_key = os.environ.get("HF_TOKEN")
            model_ = Llama(hf_key)
        elif model == "ollama":
            model_ = Llama_with_ollama()
        elif model == "gemma":
            model_ = Gemma2b_with_ollama()
        elif model == "llava":
            model_ = Llava7b_with_ollma()
        elif model == "groq":
            model_ = GroqClient()
        elif model == "replicate-llama":
            load_dotenv()
            replicate_key = os.environ.get("REPLICATE_API_TOKEN")
            print(replicate_key)
            model_ = ReplicateClient(replicate_key)
        elif model == "replicate-llava":
            load_dotenv()
            replicate_key = os.environ.get("REPLICATE_API_TOKEN")
            print(replicate_key)
            model_ = ReplicateClientLlava(replicate_key)
        elif model == "sglang":
            model_ = SGLang(key="")
        else:
            raise ValueError("Model not recognized")
        return model_

    def action(self, state):
        #determine action by querying the model
        raise NotImplementedError("action method must be implemented in subclass")
    
    def format_prompt_given_states(self,
                                   prompt: str,
                                   world_state,
                                   current_agent_state,
                                   other_agent_state,
                                   grid,
                                   prompt_methods="text",
                                   image_path=None,
                                   avaiable_action=None,
                                   current_subtasks=None,
                                   task_list=None,
                                   human_preference=None,
                                   greedy_decision=None,
                                   greedy_path=None,
                                   human_intent=None,
                                   reactive_rules=None,
                                   human_trajectory=None,
                                   robot_trajectory=None,
                                   target_position=None,
                                   tips=None,
                                   coordinated_plan=None):
        """format the prompt given world states

        does not format the task list

        Input:
            prompt: the layout read from the layout file

        Returns: formatted prompt
        """

        # format the prompt

        # current state
        current_state = self.get_agent_state_as_language(
            current_agent_state, world_state, grid, first_person=True)
        prompt = prompt.replace("{current_state}", current_state)

        # other chef state
        other_chef_state = self.get_agent_state_as_language(
            other_agent_state, world_state, grid, first_person=False)
        prompt = prompt.replace("{other_chef_state}", other_chef_state)

        kitchen_overview, kitchen_items = self.get_kitchen_as_language(
            world_state, current_agent_state, other_agent_state, grid, verbose=False)
        ##################
        # print(self.env)
        if prompt_methods == "grid":
            grid_description = "X is counter, P is pot, D is dish dispenser, O is onion dispenser, T is tomato dispenser, S is delivery location, empty square is empty square, 1 is you and 0 is the other human chef, arrow is the direction agents are facing, ø is onion \n"

            kitchen_items = grid_description + str(self.env) + "\n" + kitchen_items 
        elif prompt_methods == "image":
            image_description = "The following image contains the visual state information, the arrows represent your next avaiable action, your goal is to interact with the stuff that locates at the red circle marker, select your next step among the arrows.\n"
            kitchen_items = image_description + \
                "the image is stored at: " + str(image_path)
        # print(kitchen_items)
        ##################
        prompt = prompt.replace("{kitchen_items}", kitchen_items)
        prompt = prompt.replace("{kitchen_overview}", kitchen_overview)

        if current_subtasks:
            prompt = prompt.replace("{current_subtask}", current_subtasks)
        else:
            prompt = prompt.replace(
                "{current_subtask}", "No subtask selected yet.")

        if human_trajectory:
            prompt = prompt.replace("{human_trajectory}", human_trajectory)
        else:
            prompt = prompt.replace("{human_trajectory}", "No trajectory yet.")
        if robot_trajectory:
            prompt = prompt.replace("{robot_trajectory}", robot_trajectory)
        else:
            prompt = prompt.replace("{robot_trajectory}", "No trajectory yet.")
        if human_preference:
            prompt = prompt.replace("{human_preferences}", human_preference)
        else:
            prompt = prompt.replace(
                "{human_preferences}", "No preference yet.")

        if task_list:
            prompt = prompt.replace("{task_list}", task_list)
        if avaiable_action:
            prompt = prompt.replace("{feasible_action}", str(avaiable_action))
        else:
            prompt = prompt.replace(
                "{feasible_action}", "1. move right\n2. move left\n3. move up\n4. move down\n5. interact\n6. stay")
        # print(coordinated_plan)
        if coordinated_plan:
            prompt = prompt.replace("{coordinated_plans}", str(coordinated_plan))
        else:
            prompt = prompt.replace("{coordinated_plans}", "No coordinated plan yet")
        if greedy_decision:
            prompt = prompt.replace("{greedy_decision}", greedy_decision)
        if greedy_path:
            prompt = prompt.replace("{greedy_path}", greedy_path)
        if human_intent:
            prompt = prompt.replace("{human_intent}", human_intent)
        if reactive_rules:
            prompt = prompt.replace("{reactive_rules}", reactive_rules)
        if target_position:
            prompt = prompt.replace("{target_position}", str(target_position))
        if tips:
            prompt = prompt.replace("{tips}", tips)
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
            (0,0): "staying",
            "interact": "interact"
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
    2. {pronouns[2]} are facing {faced_item_state}
    3. {pronouns[2]} are holding {held_object}
        """            
    # 2. {pronouns[1]} orientation is facing {orientation_to_string[state['orientation']]}

    def get_kitchen_as_language(self, world_state, current_agent_state, other_agent_state, grid, verbose=False):
        """
        Construct the kitchen state as a string from a dictionary containing its contents

        if verbose=False, won't populate irrelevant items
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
                necessary = False  # only include necessary information

                item = grid_layout[i][j]
                distance = np.linalg.norm(np.array(current_agent_state['position']) - np.array((j, i)))
                item_state = f"Distance to you: {distance}. "
                if item == "X":
                    item_name = "Empty Counter"
                    
                    
                    necessary = True
                    # resolve counter contents (not that efficient)
                    for counter in world_state:
                        if counter["position"] == (j, i):
                            
                            # item_state = f"The counter has a {counter['name']} on it."
                            # necessary = True
                            if counter["name"] == "onion":
                                item_name = "Onion counter "
                            if counter["name"] == "tomato":
                                item_name = "Tomato counter"
                            if counter["name"] == "dish":
                                item_name = "Dish counter"
                            break

                elif item == "P":
                    necessary = True
                    item_name = "Pot"

                    pot_state = None
                    # find the pot at this position
                    for pot in world_state:
                        if pot["name"] == "soup" and pot["position"] == (j, i):
                            pot_state = pot
                            break

                    # special case resolution for pot
                    # item_state = "The pot is empty."
                    item_state += self.get_pot_state_as_language(pot_state)
                elif item == "D":
                    item_name = "Dish counter"
                    necessary = True
                    # item_state = "The dish dispenser has infinite empty dishes."

                elif item == "O":
                    item_name = "Onion counter"
                    necessary = True
                    # item_state = "The onion dispenser has infinite onions."
                    
                elif item == "T":
                    item_name = "Tomato counter"
                    necessary = True
                    # item_state = "The tomato dispenser has infinite tomatoes."
                    # item_state = ""
                elif item == "S":
                    item_name = "Delivery location"
                    necessary = True
                    # item_state = ""
                else:
                    item_name = "Empty square"

                    # resolve state based on where chefs are standing
                    if current_agent_state['position'] == (j, i):
                        item_state = "You are standing here."
                    elif other_agent_state['position'] == (j, i):
                        item_state += "The other chef is currently standing here."
                    else:
                        item_state += "You can stand here."

                if verbose or necessary:
                    kitchen_items.append(
                        f"\t({j},{i}): {item_name}. {item_state}")
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
    def __init__(self, agent_name,action_system_layout, reactive_prompt_template_with_layout,subtask_system_layout, subtask_prompt_template_with_layout, env, mlam, reactive_model="llama", manager_model="gpt", personality=None, debug=False):
        super().__init__(agent_name, action_system_layout, reactive_prompt_template_with_layout,subtask_system_layout, subtask_prompt_template_with_layout, env, mlam, reactive_model, manager_model, personality, debug)
        self.agent_response = ""
        self.mode = 3
        self.subtask_results = "pick up onion"
        self.subtask_index = 1
        self.human_intention= ""
        self.reactive_target_position = ""
        self.response_plan = ""
    
        self.reactive_rules = ''
        self.manager_target_position = []
        self.active_threads = []
    
        self.action_chose = (0,0)
        self.action_status = 0    #0: done, need to call api, 1: running, need to wait, 2: updated, need to send to game
        self.subtask_status = 0
        self.reactive_status = 0
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
            10: "Do nothing",
            11: "Pick up the closest soup",

        }
        self.subgoals_correspinding = {
            1: "O",
            2: "D",
            3: 'T',
            4: 'P',
            5: 'P',
            6: 'X',
            7: 'S',
            8: 'P',
            9: 'T',
            10: 'P', #revsie later
            11: 'P',
        }
        self.low_level_actions = {1:"east", 2:"west", 3:"north", 4:"south", 5:"interact", 6:"stay"}
        self.action2string = {(-1, 0): "move left", (1, 0): "move right", (0, -1)
                               : "move up", (0, 1): "move down", "interact": "interact", (0, 0): "stay"}
        self.agent_name = agent_name
        self.motion_goals = []
        self.agent_subtask_response = ""
        self.agent_log = [] # log of agent actions and associated states
        self.human_log = []  # log of human actions and associated states
        self.human_preference = None
        # self.other_agent_response = "Let's make some soups!"
        self.suggested_human_tasks = "None"
        self.action_template_file_path = reactive_prompt_template_with_layout
        self.action_system = action_system_layout
        self.subtask_template_file_path = subtask_prompt_template_with_layout
        self.subtask_system = subtask_system_layout
        system_overview = read_from_file(self.subtask_system)
        reactive_overview = read_from_file(self.action_system)
        self.manager_id, initial_reply = self.manager_model.start_new_chat(system_overview)
        self.reactive_id, initial_reactive_reply = self.reactive_model.start_new_chat(reactive_overview)

        
        self.debug = False

        # for query time logging
        self.manager_average_response_time = 0
        self.manager_response_count = 0
        self.reactive_average_response_time = 0
        self.reactive_response_count = 0

        #greedy mid level action manager 
        self.mlam = mlam

    def get_manager_outputs(self):
        """Process the latest results from the subtask queues."""
        # if not self.subtask_queue.empty():
        #     self.subtask_results = self.subtask_queue.get()
        return self.subtask_index, self.manager_target_position
    def set_human_preference(self, human_preference):
        # print('userfeedback from chatui', userfeedback)
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
        
        # print(self.subtask_status)
        if self.subtask_status == 1 : #executing
            with self.lock:
                subtask_index, target_pos = self.get_manager_outputs()
            grid = self.mdp.terrain_mtx

            player = state.players[self.agent_index].to_dict()
            other_player = state.players[1 - self.agent_index].to_dict()
            world_state = state.to_dict().pop("objects")
            
            # print("further check:", target_pos)
            if len(target_pos) < 2:
                greedy_pos_list = self.find_subgoal_position(
                player, grid, subtask_index)  # D P T S P X
            else:
                greedy_pos_list = [(target_pos[0], target_pos[1])]
            
            # print("*******************"*5)
            # print(greedy_pos_list)
            # print(greedy_pos_list)
            greedy_decisions = self.mlam._get_ml_actions_for_positions(
                greedy_pos_list)
            # print("Greedy decision: ", greedy_decisions)
            motion_planner = self.mlam.joint_motion_planner.motion_planner
            start_pos = (state.players[self.agent_index].position,
                        state.players[self.agent_index].orientation)
            # print("Start pos: ", start_pos)

            # find greedy decision with lowest cost
            best_cost = float('inf')
            best_plan = []
            for greedy_decision in greedy_decisions:
                plan, _, cost = motion_planner.get_plan(start_pos, greedy_decision)
                # print("Cost: ", cost)
                # print("Plan: ", plan)
                if cost < best_cost:
                    best_cost = cost
                    best_plan = plan

            # print("best plan: ", best_plan)
            # print("best cost: ", best_cost)
            
            if len(best_plan)>0:
                
                chosen_action = best_plan[0]
                # print('best plan',best_plan[0])
            else:
                # print('best plan is empty []')
                chosen_action = (0, 0)
            action_probs = 1
                
            # auto unstuck
            human_trajectory = self.human_log[-6:]
            human_positions = [human_state.position for human_state, _ in human_trajectory]

            # Use a set to determine the number of unique human positions
            unique_human_positions = set(human_positions)
            
            robot_trajectory = self.agent_log[-6:]
            robot_actions = [action for _, action in robot_trajectory]
            unique_robot_actions = set(robot_actions)
            robot_positions = [robot_state.position for robot_state, _ in robot_trajectory]
            zero_robot_actions = sum(1 for _, action in robot_trajectory if action == (0, 0))
            # Use a set to determine the number of unique robot positions
            unique_robot_positions = set(robot_positions)
            # print(human_trajectory, robot_trajectory)
            # print(unique_robot_positions, unique_human_positions, len(self.agent_log), zero_robot_actions)

            if len(unique_robot_positions) <= 1  and len(unique_human_positions) <= 1 and len(unique_robot_actions) <= 1\
                and self.prev_state is not None and len(self.agent_log) > 6 and len(self.human_log) > 6 \
                and zero_robot_actions < 1:
                print("unstuck")
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
            self.record_agent_log(state.players[1], chosen_action)
        else: #0 finished task and about to query, 2 for querying, we will wait
            chosen_action = (0, 0)
            action_probs = 1

        if chosen_action == "interact":
            with threading.Lock():
                print("finished current task")
                self.subtask_status = 0

        
        
        return chosen_action, {"action_probs": action_probs}

    def ml_action(self,state):
        return self.motion_goals
    

    def find_subgoal_position(self, player, grid, goal_index):
        greedy_pos_list = []
        if goal_index == 10: #do nothing
            greedy_pos_list.append(player['pos'])
        elif goal_index == 6: # put it into counter. 
            greedy_pos_list.append((3,2)) # this should comes from llm
        subgoal = self.subgoals_correspinding[goal_index]
        # print(subgoal)
        # print(grid)
        pos_index = np.where(np.array(grid) == subgoal)
        
        for i in range(len(pos_index[0])):
            greedy_pos_list.append((pos_index[1][i], pos_index[0][i]))
        return greedy_pos_list
    
    def subtasking(self, state):
        ### send suggestion to the overcooked pygame clas
        self.robotfeedback = {
                    "frequent_feedback":{ 
                            "value": self.suggested_human_tasks, # Placeholder for the actual frequency feedback value
                            "is_updated": False # Flag indicating if this feedback has been updated
                        },
                    "hasAgentPaused":False # Used only For mode 3, since in mode 3 At the beginning Agent will pause the game
                }
        if self.subtask_status == 0: #thread free
            # enable for manager mind
            self.subtask_status = 2 # querying
            self.async_determine_subtask(state)
        elif self.subtask_status == 2:
            print("querying")
        else:
            ### send suggestion to the overcooked pygame clas
            self.robotfeedback = {
                    "frequent_feedback":{ 
                            "value": self.suggested_human_tasks, # Placeholder for the actual frequency feedback value
                            "is_updated": True # Flag indicating if this feedback has been updated
                        },
                    "hasAgentPaused":False # Used only For mode 3, since in mode 3 At the beginning Agent will pause the game
                }
            print("executing the current subtask", self.subtask_results)
        return self.robotfeedback
    



        

        
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



    def async_determine_subtask(self, state):
        """Function to asynchronously determine subtask."""
        def task(state):
            self.subtask_results, self.subtask_index, self.manager_target_position, self.suggested_human_tasks\
                  = self.determine_subtask(state)
            self.subtask_status = 1 # executing
           
            

        thread = threading.Thread(target=task, args=(state,))
        thread.start()
        self.active_threads.append(thread)






    def reactive_interactive_query(self, state):
        "query the interactive mind if there is request"
        current_agent_state = state.players[self.agent_index].to_dict()
        other_agent_state = state.players[1 - self.agent_index].to_dict()
        world_state = state.to_dict().pop("objects")
        # obtain prompt layout from file]
        prompt_layout = read_from_file(self.action_template_file_path)
        task_list, cross_reference = self.get_relevant_subtasks(current_agent_state)

        formatted_task_list = "\n".join([f"\Option {i}: {task}" for i, task in enumerate(task_list, 1)])
        grid = self.mdp.terrain_mtx
        # # last 5 human states and actions
        human_trajectory = self.human_log[-5:]
        human_trajectory_in_language = ""
        for human_state, action in human_trajectory:
            human_trajectory_in_language += f"at Position: {human_state.position},human {self.action2string[action]}\n"

        # last 5 human states and actions
        robot_trajectory = self.agent_log[-5:]
        robot_trajectory_in_language = ""
        for robot_state, action in robot_trajectory:
            robot_trajectory_in_language += f"at Position: {robot_state.position},robot {self.action2string[action]}\n"
        #format prompt layout given the current states (this will replace all the placeholders in the prompt layout)
        
        prompt = self.format_prompt_given_states(prompt_layout, 
                                                 world_state, 
                                                 current_agent_state,
                                                 other_agent_state, 
                                                 grid= grid,
                                                 prompt_methods = "grid",
                                                 task_list=formatted_task_list,
                                                 human_preference=self.human_preference, 
                                                 human_trajectory=human_trajectory_in_language, 
                                                 robot_trajectory=robot_trajectory_in_language)

        # message_to_other_chef = "Happy to work with you!"
        # print("promt", prompt)
        # query the model given prompt
        reactive_start_time = time.time()
        response = self.reactive_model.query(self.reactive_id, "user", reactiveReasoning, prompt, temp=0.2)

        self.human_intention = response.human_intention
        self.reactive_target_position = response.reactive_target_position
        self.response_plan = response.response_plan

        reactive_elapsed_time  = time.time() - reactive_start_time
        print(f"ReactiveQuery: took {reactive_elapsed_time} seconds to evaluate")
        

        # log the prompt generated
        write_to_file(f"llm/log/reactive_mind_prompt_generated_{self.agent_name}.txt", prompt)

        if self.debug:
            print("**********manager*************")
            print(self.agent_name + ": ")
            print(response)
            print("********************************")
            # print("Other Agent: ", self.other_agent_response)
        
        
        return self.human_intention, self.reactive_target_position, self.response_plan


    # def reactive_query(self, state):
    #     current_agent_state = state.players[self.agent_index].to_dict()
    #     other_agent_state = state.players[1 - self.agent_index].to_dict()
    #     world_state = state.to_dict().pop("objects")

    #     # obtain prompt layout from file]
    #     prompt_layout = read_from_file(self.subtask_template_file_path)
    #     task_list, cross_reference = self.get_relevant_subtasks(current_agent_state)

    #     formatted_task_list = "\n".join([f"\Option {i}: {task}" for i, task in enumerate(task_list, 1)])
    #     grid = self.mdp.terrain_mtx
    #     # # last 5 human states and actions
    #     human_trajectory = self.human_log[-5:]
    #     human_trajectory_in_language = ""
    #     for human_state, action in human_trajectory:
    #         human_trajectory_in_language += f"at Position: {human_state.position},human {self.action2string[action]}\n"

    #     # last 5 human states and actions
    #     robot_trajectory = self.agent_log[-5:]
    #     robot_trajectory_in_language = ""
    #     for robot_state, action in robot_trajectory:
    #         robot_trajectory_in_language += f"at Position: {robot_state.position},robot {self.action2string[action]}\n"
    #     #format prompt layout given the current states (this will replace all the placeholders in the prompt layout)
    #     prompt = self.format_prompt_given_states(prompt_layout, world_state, current_agent_state, other_agent_state, grid= grid,task_list=formatted_task_list,human_preference=self.human_preference
    #                                              , human_trajectory=human_trajectory_in_language, robot_trajectory=robot_trajectory_in_language)

    #     # message_to_other_chef = "Happy to work with you!"
    #     # print("promt", prompt)
    #     # query the model given prompt
    #     reactive_start_time = time.time()
    #     response = self.reactive_model.query(self.reactive_id, "user", reactiveReasoning, prompt, temp=0.2)

    #     self.human_intention = response.human_intention
    #     self.reactive_rules = response.reactive_adaptive_rules

    #     reactive_elapsed_time  = time.time() - reactive_start_time
    #     print(f"ReactiveQuery: took {reactive_elapsed_time} seconds to evaluate")
        

    #     # log the prompt generated
    #     write_to_file(f"llm/log/reactive_mind_prompt_generated_{self.agent_name}.txt", prompt)

    #     if self.debug:
    #         print("**********manager*************")
    #         print(self.agent_name + ": ")
    #         print(response)
    #         print("********************************")
    #         # print("Other Agent: ", self.other_agent_response)
        
        
    #     return self.human_intention, self.reactive_rules
    

    def determine_subtask(self, state):
        """query the model appropriately for the optimal subtask

        Return: index of the subtask as defined according to self.subtasks
        """
        current_agent_state = state.players[self.agent_index].to_dict()
        other_agent_state = state.players[1 - self.agent_index].to_dict()
        world_state = state.to_dict().pop("objects")

        # obtain prompt layout from file]
        prompt_layout = read_from_file(self.subtask_template_file_path)
        task_list, cross_reference = self.get_relevant_subtasks(current_agent_state)

        formatted_task_list = "\n".join([f"\Option {i}: {task}" for i, task in enumerate(task_list, 1)])
        grid = self.mdp.terrain_mtx
        # # last 5 human states and actions
        human_trajectory = self.human_log[-5:]
        human_trajectory_in_language = ""
        for human_state, action in human_trajectory:
            human_trajectory_in_language += f"at Position: {human_state.position},human {self.action2string[action]}\n"

        # last 5 human states and actions
        robot_trajectory = self.agent_log[-5:]
        robot_trajectory_in_language = ""
        for robot_state, action in robot_trajectory:
            robot_trajectory_in_language += f"at Position: {robot_state.position},robot {self.action2string[action]}\n"
        #format prompt layout given the current states (this will replace all the placeholders in the prompt layout)
        agent_plan = f"human wants to {self.human_intention}, you can, {self.response_plan} and first move to {self.reactive_target_position}"
        prompt = self.format_prompt_given_states(prompt_layout, 
                                                 world_state, 
                                                 current_agent_state, 
                                                 other_agent_state, 
                                                 grid= grid,
                                                 prompt_methods = "grid",
                                                 task_list=formatted_task_list,
                                                 human_preference=self.human_preference, 
                                                 human_trajectory=human_trajectory_in_language, 
                                                 robot_trajectory=robot_trajectory_in_language,
                                                 coordinated_plan=agent_plan)

        # message_to_other_chef = "Happy to work with you!"
        # print("promt", prompt)
        # query the model given prompt
        manager_start_time = time.time()
        response = self.manager_model.query(self.manager_id, "user",managerReasoning, prompt, temp=0.2)

        manager_elapsed_time  = time.time() - manager_start_time
        print(f"ManagerMind: took {manager_elapsed_time} seconds to evaluate")
        self.manager_average_response_time = (self.manager_average_response_time * self.manager_response_count + manager_elapsed_time) / (self.manager_response_count + 1)
        self.manager_response_count += 1

        # log the prompt generated
        write_to_file(f"llm/log/manager_mind_prompt_generated_{self.agent_name}.txt", prompt)

        if self.debug:
            print("**********manager*************")
            print(self.agent_name + ": ")
            print(response)
            print("********************************")
            # print("Other Agent: ", self.other_agent_response)    
        subtask_index = response.final_subtasks_id
        target_pos = response.target_position
        human_tasks = response.human_tasks
        # print("target_pos: ",  target_pos, "subtask_index: ", subtask_index)
        subtask_index = cross_reference[subtask_index - 1]
        print(f"ManagerMind:  selected subtask {subtask_index}, {self.subtasks[subtask_index]}")
        
        # Visual conversation
        # self.agent_subtask_response = f"I selected {subtask_index}, {self.subtasks[subtask_index]}"
        # self.agent_subtask_response = message_to_other_chef
        return self.subtasks[subtask_index], subtask_index, target_pos, human_tasks

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
        # task_list.append("Do nothing")

        
        return task_list, cross_reference

    def pause_manager_mind(self):
        #TODO: pause threads
        pass
    def resume_manager_mind(self):
        #TODO:pause threads
        pass

    def record_human_log(self, human_state, action):
        self.human_log.append((human_state, action))

    def record_agent_log(self, agent_state, action):
        self.agent_log.append((agent_state, action))