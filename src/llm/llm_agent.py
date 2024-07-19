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
        
class LatestItemQueue(Queue):
    def get(self, block=True, timeout=None):
        """Retrieve the latest item from the queue and clear all other items."""
        with self.mutex:
            last_item = None
            if self.empty():
                raise ValueError("Queue is empty")  
            else:
                last_item = super().get(block, timeout)
            self.queue.clear()  # Ensure the queue is indeed empty
            return last_item
        

#base class for LLM models
class LLMModel(Agent):
    def __init__(self,agent_name,action_prompt_template_with_layout, subtask_prompt_template_with_layout, reactive_model="llama", manager_model="gpt", personality=None, debug=True):
        super().__init__()
        self.agent_name = agent_name
        self.agent_log = []
        print("\nInitializing LLM models\n")

        self.debug = debug
        self.reactive_model = self.initialize_model(reactive_model)
        self.manager_model = self.initialize_model(manager_model)

        self.action_prompt_layout = read_from_file(action_prompt_template_with_layout)
        self.subtask_prompt_layout = read_from_file(subtask_prompt_template_with_layout)
    
    def initialize_model(self, model):
        if model=="gpt":
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
        kitchen_overview, kitchen_items = self.get_kitchen_as_language(world_state, current_agent_state, other_agent_state,grid)
        prompt = prompt.replace("{kitchen_items}", kitchen_items)
        prompt = prompt.replace("{kitchen_overview}", kitchen_overview)
        
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
    def __init__(self, agent_name, action_prompt_template_with_layout, subtask_prompt_template_with_layout, reactive_model="llama", manager_model="gpt", personality=None, debug=False):
        super().__init__(agent_name, action_prompt_template_with_layout, subtask_prompt_template_with_layout, reactive_model, manager_model, personality, debug)
        self.agent_response = ""
        self.mode = 3

        # Use the custom queue for each type of result
        self.shared_state = SharedState()
        self.subtask_queue = LatestItemQueue()
        self.subtask_results = "Do nothing"
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
        # if not self.subtask_queue.empty():
        #     self.subtask_results = self.subtask_queue.get()
        return self.subtask_results
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

        with self.lock:
            self.update_state(state)
            current_subtasks = self.get_subtasks()
        
        if self.action_status == 0:
            self.async_determine_action(state, current_subtasks,  screen)
            self.action_status = 1
            return (0,0),{}
        elif self.action_status == 1:
            return (0,0),{} 
        elif self.action_status == 2:
            self.action_status = 0
            # print("action_chose", self.action_chose)
            return self.action_chose, {}
        

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
    def async_determine_action(self, state, current_subtasks, screen):
        """Function to asynchronously determine action."""
        def task(state, current_subtasks, screen):
            new_action = self.determine_action(state, current_subtasks, screen)
            with self.lock:
                self.action_chose = new_action
                self.action_status = 2
            

        thread = threading.Thread(target=task, args=(state, current_subtasks, screen))
        thread.start()
        self.active_threads.append(thread)
    

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
        def task(shared_state, results_queue):
            while True:
                state = shared_state.get_state()
                if state is None:
                    time.sleep(0.1)  # Sleep briefly if no state is available
                    continue
                self.subtask_results = self.determine_subtask(state)
                # 3s for calling the manager mind
                time.sleep(2)  # Simulate processing time

        thread = threading.Thread(target=task, args=(self.shared_state, self.subtask_queue))
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
        prompt = self.format_prompt_given_states(prompt_layout, world_state, current_agent_state, other_agent_state, grid= grid,task_list=formatted_task_list,human_preference=self.human_preference)

        # message_to_other_chef = "Happy to work with you!"

        # query the model given prompt
        response = self.manager_model.query(prompt, temp=0.4)
        
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
        print(f"ManagerMind:  selected {subtask_index}, {self.subtasks[subtask_index]}")
        
        # Visual conversation
        # self.agent_subtask_response = f"I selected {subtask_index}, {self.subtasks[subtask_index]}"
        # self.agent_subtask_response = message_to_other_chef
        return self.subtasks[subtask_index]

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
            cross_reference = [1, 2, 5]
            task_list = ["Pick up the nearest onion", "Pick up the nearest dish", "Start cooking the nearest pot"]
        else:
            # construct task list based on held object, and add to cross reference with switch case
            task_list = [f"Place {held_object['name']} on the nearest counter"]

            # construct cross reference list

            if held_object['name'] == "onion":
                # remove subtask that cant do with an onion held
                task_list.append("Put the onion in the nearest pot")
                cross_reference = [6, 8]

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