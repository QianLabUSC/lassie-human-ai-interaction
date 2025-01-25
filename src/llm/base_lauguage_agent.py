from overcooked_ai_py.agents.agent import Agent, GreedyHumanModel
import numpy as np
from overcooked_ai_py.mdp.actions import Action
# from dotenv import load_dotenv
import os
import time
from llm.llm_api import GPT, Llama_with_ollama, Gemma2b_with_ollama, Llava7b_with_ollma,SGLang, coordinatorReasoning,subtask_managerReasoning, rule
from llm.utils import read_from_file, write_to_file


class LLMModel(GreedyHumanModel):
    def __init__(self,agent_name,action_system_layout,action_prompt_template_with_layout,subtask_system_layout, subtask_prompt_template_with_layout, env, mlam, coordinator_model="gpt", subtask_manager_model="llama", personality=None, debug=True):
        super().__init__(mlam)
        self.env = env
        self.agent_name = agent_name
        self.agent_log = []
        print("\nInitializing LLM models\n")

        self.debug = debug
        self.coordinator_model = self.initialize_model(coordinator_model)
        self.subtask_manager_model = self.initialize_model(subtask_manager_model)

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
        elif model == "ollama":
            model_ = Llama_with_ollama()
        elif model == "gemma":
            model_ = Gemma2b_with_ollama()
        elif model == "llava":
            model_ = Llava7b_with_ollma()
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
                                   coordinator_rules=None,
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

        kitchen_overview, kitchen_items, kitchen_item_pos = self.get_kitchen_as_language(
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
        if coordinator_rules:
            prompt = prompt.replace("{coordinator_rules}", coordinator_rules)
        if target_position:
            prompt = prompt.replace("{target_position}", str(target_position))
        if tips:
            prompt = prompt.replace("{tips}", tips)
        

        return prompt


    def format_graph_generation_prompt_given_states(self,
                                   prompt: str,
                                   world_state,
                                   current_agent_state,
                                   other_agent_state,
                                   grid,
                                   order_list, 
                                   prompt_methods="text",
                                   image_path=None):
        """format the prompt given world states

        does not format the task list

        Input:
            prompt: the layout read from the layout file

        Returns: formatted prompt
        """

        # format the prompt


        kitchen_overview, kitchen_items, kitchen_item_pos = self.get_kitchen_as_language(
            world_state, current_agent_state, other_agent_state, grid, verbose=False)
        ##################
        # print(self.env)
        if order_list:
            orders = []
            for number in range(len(self.order_list)):
                orders.append(f"Recipe {number}: Requires {len(self.order_list[number]['ingredients'])} ingredients: " + ", ".join(self.order_list[number]['ingredients']) + ". The ingredients should be placed in a pot and cooked to make the soup.")
            orders_formatted_in_language = "\n".join(orders)
            prompt = prompt.replace("{recipe_book}", str(orders_formatted_in_language))

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
        example_subtasks = '''example_subtask = {
                "id": int,  # Unique ID of the subtask start from 0
                "name": string,  # Task description, e.g. "Get onion"
                "target_position_id": list[int],  # IDs of target positions selected from provided locations
                "task_type": int,  # Integer representing the task type (e.g., 1 = GETTING, refer to all avaiable types)
                "task_status": int,  # Integer representing the task status (e.g., refer to all avaiable status, but you only to judge if this subtask has been finished, if not, leave unknown, I will handle it based on graph)
                "parent_subtask": list[int]  #  Only list of IDs of parent subtasks that are a must and reprequisite to this task, (leave empty if no required subtasks before this, or other agent can help do this)
            }'''
        prompt = prompt.replace("{example_subtasks}", example_subtasks)

        available_subtask_types = '''0: "PUTTING", 1: "GETTING", 2: "COOKING"'''
        prompt = prompt.replace("{available_subtask_types}", available_subtask_types)

        available_subtask_status = '''0: "UNKNOWN", 1: "READY_TO_EXECUTE", 2: "SUCCESS", 3: "FAIL", 4: "NOT READY"'''
        prompt = prompt.replace("{available_subtask_status}", available_subtask_status)

  
        
        

        return prompt, kitchen_item_pos

    def format_subtask_assign_prompt_given_states(self,
                                   prompt: str,
                                   world_state,
                                   current_agent_state,
                                   other_agent_state,
                                   graph_state,
                                   grid
                                   ):
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
        prompt = prompt.replace("{robot_state}", current_state)

        # other chef state
        other_chef_state = self.get_agent_state_as_language(
            other_agent_state, world_state, grid, first_person=False)
        prompt = prompt.replace("{human_state}", other_chef_state)

        graph_state_lauguage = graph_state.get_state_in_lauguage(current_agent_state, other_agent_state)
        prompt = prompt.replace("{graph_state}", graph_state_lauguage)
        return prompt
       

    
    def format_subtask_status_change(self,
                                   prompt: str,
                                   world_state,
                                   current_agent_state,
                                   other_agent_state,
                                   graph_state,
                                   grid,
                                   human_log,
                                   agent_log
                                   ):
        """format the prompt given world states for solving the issues that robot might interact in a wrong place.

        does not format the task list

        Input:
            prompt: the layout read from the layout file

        Returns: formatted prompt
        """

        # format the prompt
        # current state
        # current_state = self.get_agent_hist_as_lauguage(
        #     agent_log, world_state, grid, first_person=True)
        # prompt = prompt.replace("{robot_state}", current_state)

        # # other chef state
        # other_chef_state = self.get_agent_hist_as_lauguage(
        #     human_log, world_state, grid, first_person=False)
        # prompt = prompt.replace("{human_state}", other_chef_state)
        # grid_layout = grid
        # item_index = 0
        # item_states = []
        # for i in range(len(grid_layout)):
        #     for j in range(len(grid_layout[i])):
                
        #         item = grid_layout[i][j]
        #         item_state = ""
        #         if item == "P":
        #             item_name = "Pot"

        #             pot_state = None
        #             # find the pot at this position
        #             for pot in world_state:
        #                 if pot["name"] == "soup" and pot["position"] == (j, i):
        #                     pot_state = pot
        #                     break
        #             item_state += self.get_pot_state_as_language(pot_state)
        #             item_index += 1
        #             item_states.append(
        #                 f"\t{item_index}: {item_name}. {item_state}")
        # prompt = prompt.replace("{pot_state}", "\n".join(item_states))
        prev_robot_state, prev_world_state, prev_action = agent_log[-1]
        agent_prev_state = self.get_agent_last_state_as_language(prev_action, prev_robot_state, prev_world_state, grid, first_person=True)
        prompt = prompt.replace("{agent_prev_state}", agent_prev_state)

        if human_log:  # Check if human_log is not empty
            prev_human_state, prev_human_world_state, prev_human_action = human_log[-1]
            human_prev_state = self.get_agent_last_state_as_language(prev_human_action, prev_human_state, prev_human_world_state, grid, first_person=True)
        
        else:
            # Handle the empty case
            # prev_human_state = None  # or a default value
            # prev_human_action = None  # or a default value
            human_prev_state = "The human chef has not taken any action yet."
        prompt = prompt.replace("{human_prev_state}", human_prev_state)

        current_state = self.get_agent_state_as_language(
            current_agent_state, world_state, grid, first_person=True)
        prompt = prompt.replace("{robot_state}", current_state)

        # other chef state
        other_chef_state = self.get_agent_state_as_language(
            other_agent_state, world_state, grid, first_person=False)
        prompt = prompt.replace("{human_state}", other_chef_state)


        _, id = graph_state.check_executing_by_agent_id(1)
        robot_executing_task = graph_state.get_node_by_id(id)
        _, id = graph_state.check_executing_by_agent_id(0)
        human_executing_task = graph_state.get_node_by_id(id)
        
        prompt = prompt.replace("{robot_task}", f"{robot_executing_task.name.split('-')[0]}")
        prompt = prompt.replace("{human_task}", f"{human_executing_task.name.split('-')[0]}")
        return prompt


    def get_agent_hist_as_lauguage(self, log, world_state, grid, first_person=False):
       
        agent_trajectory_in_language = ""
        robot_trajectory = log[-5:]
        
        for robot_state, action in robot_trajectory:
            if robot_state.held_object is not None:
                held_object = "a " + robot_state.held_object.name
            else:
                held_object = "nothing"
            agent_trajectory_in_language += f"robot now at Position: {robot_state.position}, holding {held_object}, chose {self.action2string[action]}\n"
        return agent_trajectory_in_language
    
    def get_agent_last_state_as_language(self, action, state, world_state, grid,first_person=False):
        """Construct the agent state as a string from a dictionary containing its contents
        """
        state = state.to_dict()
        # pronouns resolver
        if first_person:
            pronouns = ["Robot is", "robot's", "robot"]
        else:
            pronouns = ["Human is", "His", "He"]

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

        return f"""at last step, 1. {pronouns[0]} at the coordinates {state['position']}
    2. {pronouns[2]} are facing {faced_item_state}
    3. {pronouns[2]} are holding {held_object}
4. {pronouns[2]} chose {self.action2string[action]}
        """            
    # 2. {pronouns[1]} orientation is facing {orientation_to_string[state['orientation']]}

    
    
    def get_agent_state_as_language(self, state, world_state,grid,first_person=False):
        """Construct the agent state as a string from a dictionary containing its contents
        """
        
        # pronouns resolver
        if first_person:
            pronouns = ["Robot is", "robot's", "robot"]
        else:
            pronouns = ["Human is", "His", "He"]

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

        return f"""Now, 1. {pronouns[0]} at the coordinates {state['position']}
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
        kitchen_item_pos = []
        item_index = 0
        for i in range(len(grid_layout)):
            for j in range(len(grid_layout[i])):
                necessary = False  # only include necessary information

                item = grid_layout[i][j]
                distance = np.linalg.norm(np.array(current_agent_state['position']) - np.array((j, i)))
                # item_state = f"Distance to you: {distance}. "
                item_state = ""
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
                    kitchen_item_pos.append([j, i])
                    kitchen_items.append(
                        f"\t{item_index}: {item_name}. {item_state}")
                    item_index += 1
        # format with newline operator
        return kitchen_overview, "\n".join(kitchen_items), kitchen_item_pos

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
        if not is_cooking:
            if is_ready:
                pot_items = f"The soup is finished cooking. "
            else:
                pot_items = f"The soup has not started cooking yet."
        else:
            pot_items = f"The soup has already started cooking, but is not finished cooking. It is {cooking_timer} out of {cook_time} ticks cooked."

        pot_items += f"There are {number_of_ingredients} ingredients in this soup: {ingredients}. "

        
        return pot_items

        