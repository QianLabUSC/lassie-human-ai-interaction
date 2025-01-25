from overcooked_ai_py.agents.agent import Agent, GreedyHumanModel
import itertools
import numpy as np
from openai import OpenAI
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.actions import Action
import time
from llm.llm_api import GPT, Llama_with_ollama, Gemma2b_with_ollama, Llava7b_with_ollma,SGLang, coordinatorReasoning, subtask_managerReasoning, rule, graph_generation, subtaskStatus
from llm.utils import read_from_file, write_to_file
import threading
from llm.subtask_graph import Graph, SubTask
from llm.base_lauguage_agent import LLMModel
    
"""
LLM model that query every atomic action
"""
class HRT(LLMModel):
    def __init__(self, agent_name,action_system_layout, coordinator_prompt_template_with_layout,subtask_system_layout, subtask_prompt_template_with_layout, order_list, env, mode, mlam, coordinator_model="llama", subtask_manager_model="gpt", personality=None, debug=False):
        super().__init__(agent_name, action_system_layout, coordinator_prompt_template_with_layout,subtask_system_layout, subtask_prompt_template_with_layout, env, mlam, coordinator_model, subtask_manager_model, personality, debug)
        self.agent_response = ""
        self.mode = 3
        self.subtask_results = "pick up onion"
        self.subtask_index = 1
        self.human_intention= ""
        self.coordinator_target_position = ""
        self.response_plan = ""
        self.order_list = order_list
        self.coordinator_rules = ''
        self.subtask_manager_target_position = []
        self.active_threads = []
    
        self.action_chose = (0,0)
        self.action_status = 0    #0: done, need to call api, 1: running, need to wait, 2: updated, need to send to game
        self.subtask_status = 0
        self.coordinator_status = 0
        self.lock = threading.Lock()
        self.current_action = (0,0)

        print(f"User Mode at Overcooked: {mode}")     
        self.usermode = mode
        # subtask_manager mind settings
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
        self.action_template_file_path = coordinator_prompt_template_with_layout
        self.action_system = action_system_layout
        self.subtask_template_file_path = subtask_prompt_template_with_layout
        self.subtask_system = subtask_system_layout
        system_overview = read_from_file(self.subtask_system)
        # print(self.subtask_system)
        coordinator_overview = read_from_file(self.action_system)
        self.subtask_manager_id, initial_reply = self.subtask_manager_model.start_new_chat(system_overview)
        self.coordinator_id, initial_coordinator_reply = self.coordinator_model.start_new_chat(coordinator_overview)

        
        self.debug = False

        # for query time logging
        self.subtask_manager_average_response_time = 0
        self.subtask_manager_response_count = 0
        self.coordinator_average_response_time = 0
        self.coordinator_response_count = 0

        #greedy mid level action subtask_manager 
        self.mlam = mlam

        self.graph_state = Graph(mlam=mlam)
        self.agent_subtask_id, self.human_subtask_id = None, None
        
    def initilize_Graph(self, state):
        print("initialize graph")
        response, pos_list = self.query_subtask_list(state)
        self.graph_state.generate_graph_from_subtask_objects(response, pos_list)

    def query_subtask_list(self, state):
        
        current_agent_state = state.players[self.agent_index].to_dict()
        other_agent_state = state.players[1 - self.agent_index].to_dict()
        world_state = state.to_dict().pop("objects")
        grid = self.env.mdp.terrain_mtx
        # obtain prompt layout from file]
        prompt_layout = read_from_file(f"llm/layout/HRT/HRT_generate_graph.txt")
        prompt, pos_list = self.format_graph_generation_prompt_given_states(
                                                prompt_layout,
                                                world_state,
                                                current_agent_state,
                                                other_agent_state,
                                                grid, 
                                                self.order_list,
                                                prompt_methods = "grid")
        system_prompt = read_from_file(f"llm/layout/HRT/HRT_generate_graph_system.txt")
        coordinator_start_time = time.time()
        response = self.coordinator_model.query_direct(graph_generation, system_prompt, prompt, temp=0.2)

        print(response)
        

        coordinator_elapsed_time  = time.time() - coordinator_start_time
        print(f"generating graph: took {coordinator_elapsed_time} seconds to evaluate")
        

        # log the prompt generated
        write_to_file(f"llm/log/graph_generation_prompt_generated_{self.agent_name}.txt", prompt)
        
        return response, pos_list
    def calculate_subtasks_cost_for_recipe(self, recipe, grid):
        # WE CAN DEFINE THE SCORE OF INGREDIENTS AS MUCH AS WE WANT
        # STATIC Value
        ingredient_scores = {
            'onion': 2,
            'tomato': 2,
            # You can add more ingredients and their corresponding scores here
        }

        """Calculate the total number of subtasks needed to complete a given recipe on the current map."""
        total_subtasks_cost = 0

       
       # TODO: if repeated ingredeinent dont count
        for ingredient in recipe['ingredients']:
            ingredient_pos = self.find_ingredient_position_in_given_map(ingredient, grid)
            if ingredient_pos:
                # Add the score for the ingredient
                total_subtasks_cost += ingredient_scores.get(ingredient, 0)  # Default score of 0 if ingredient not in the map

        print(f"Total subtasks cost for recipe: {total_subtasks_cost}")
        return total_subtasks_cost

    # finding the coordinates of a given ingredient on the kitchen map
    def find_ingredient_position_in_given_map(self, ingredient, grid):
        """Find the position of the given ingredient on the map."""
        ingredient_symbol = {
            'onion': 'O',
            'tomato': 'T',
            'dish': 'D'
        }
        ingredient_code = ingredient_symbol.get(ingredient, ' ')

        # Search the grid for the ingredient
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if grid[i][j] == ingredient_code:
                    return (j, i)
        return None
    
    def get_subtask_manager_outputs(self):
        """Process the latest results from the subtask queues."""
        # if not self.subtask_queue.empty():
        #     self.subtask_results = self.subtask_queue.get()
        return self.subtask_index, self.subtask_manager_target_position



    def update_state(self, state):
        """Update the shared state."""
        self.shared_state.update_state(state)

    def stop_all_threads(self):
        """Stops all running threads. Threads should be designed to stop safely."""
        for thread in self.active_threads:
            thread.join()  # Ensuring threads have completed

    

    def action(self, state):
        
    
        grid = self.mdp.terrain_mtx

        player = state.players[self.agent_index].to_dict()
        other_player = state.players[1 - self.agent_index].to_dict()
        world_state = state.to_dict().pop("objects")
        
       
        start_pos = (state.players[self.agent_index].position,
                    state.players[self.agent_index].orientation)
        

        if self.robot_subtask is not None:
            
            robot_costs, robot_actions = self.graph_state.calculate_distance_to_pos(start_pos, self.robot_subtask)

        
            
            if len(robot_costs)>0:
                # find the index of minimum cost in robot costs
                min_cost_index = np.argmin(np.array(robot_costs))
                chosen_action = robot_actions[min_cost_index][0]
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
            robot_actions = [action for _, _, action in robot_trajectory]
            unique_robot_actions = set(robot_actions)
            robot_positions = [robot_state.position for robot_state, _, _ in robot_trajectory]
            zero_robot_actions = sum(1 for _,_, action in robot_trajectory if action == (0, 0))
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
            self.record_agent_log(state.players[1], state.to_dict().pop("objects"), chosen_action)


  
        else:
            chosen_action = (0,0)
            action_probs = 1
        self.current_action  = chosen_action
        print(chosen_action)
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
    
    def subtasking(self, state, ui):
        # print("check if graph task status has changed and trigger coordinating or task reassign based on mode")
        agent_executing, _ = self.graph_state.check_executing_by_agent_id(1)
        human_executing, _ = self.graph_state.check_executing_by_agent_id(0) # 0 is human
        

        if (agent_executing and human_executing):
            print("both are executing")
            # update graph and chech failure
            # when user interacts and has reached the goal, we need to check the status, if the task has been finished
            robot_pos = (state.players[self.agent_index].to_dict()['position'], state.players[self.agent_index].to_dict()['orientation'])
            cost, _ = self.graph_state.calculate_distance_to_pos(robot_pos, self.robot_subtask)
            if self.graph_state.checking_time_out_fail(1, True):
                pass # time out failure call back # ask gpt to diagnose the failure
            elif self.current_action == "interact" and np.min(cost) == 1:
                response = self.query_subtask_status(state)
                if not self.graph_state.update_status_by_agent_id(0, response.human_status):
                    # consider to recall the gpt to diagnose the failure, no id is find on executing. 
                    pass
                    
                if not self.graph_state.update_status_by_agent_id(1, response.agent_status):
                    # consider to recall the gpt to diagnose the failure, no id is find on executing. 
                    pass

                #  update status given success change
                self.graph_state.update_node_status()
                self.robot_subtask = None

        else:
            print("one of them is not executing")
            self.agent_subtask_id, self.human_subtask_id = self.determine_subtask(state)
            print(self.agent_subtask_id, self.human_subtask_id)
            self.robot_subtask = self.graph_state.assign_task(self.agent_subtask_id, 1)
            self.human_subtask = self.graph_state.assign_task(self.human_subtask_id, 0)


        
            
    def query_subtask_status(self, state):
        
        current_agent_state = state.players[self.agent_index].to_dict()
        other_agent_state = state.players[1 - self.agent_index].to_dict()
        world_state = state.to_dict().pop("objects")
        grid = self.mdp.terrain_mtx
        # obtain prompt layout from file]
        prompt_layout = read_from_file(f"llm/layout/HRT/HRT_assign_subtask_status_query.txt")
        prompt = self.format_subtask_status_change(prompt_layout, 
                                                 world_state, 
                                                 current_agent_state, 
                                                 other_agent_state, 
                                                 self.graph_state,
                                                 grid= grid,
                                                 human_log=self.human_log,
                                                 agent_log=self.agent_log
                        
                                                )
        system_prompt = read_from_file(f"llm/layout/HRT/HRT_assign_subtask_status_query_system.txt")
        coordinator_start_time = time.time()
        response = self.subtask_manager_model.query_direct(subtaskStatus, system_prompt, prompt, temp=0.2)

        print(response)
        

        coordinator_elapsed_time  = time.time() - coordinator_start_time
        print(f"status analysis: took {coordinator_elapsed_time} seconds to evaluate")
        

        # log the prompt generated
        write_to_file(f"llm/log/subtaskanalysis_prompt_generated_{self.agent_name}.txt", prompt)
        
        return response
        # CHECK IF current executing task is finished or failed
        
    def checking_time_out_fail(self):
        print("failure monitor")
        pass
    # functions for communciating
    def launch_conversation(self, state, ui, message):
        # launch a converation for continure improvement
        pass

    def end_conversation(self, conversation_id):
        pass


    def reply_to_human(self, human_message):
        # reply human messages
        return "hh"


    # functions to if the robot should launch conversation
    def coordinating(self, state, ui):
        print("no coordinating for now")
        bool_coordinate = self._check_if_need_coordinating(state, self.mode)
        if bool_coordinate:
            message = "coordinate message"
            self.launch_conversation(state, ui, message)
            # human_intention, coordinator_pos, response_plan = self.agent2.coordinator_interactive_query(self.env.state)
            # ui.robot_generate_callback(f"human wants to {human_intention}, I will, {response_plan} and first move to {coordinator_pos}")   
            
    def _check_if_need_coordinating(state, mode):
        pass

        

        
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






    def coordinator_interactive_query(self, state):
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
        for robot_state, _, action in robot_trajectory:
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
        coordinator_start_time = time.time()
        response = self.coordinator_model.query(self.coordinator_id, "user", coordinatorReasoning, prompt, temp=0.2)

        self.human_intention = response.human_intention
        self.coordinator_target_position = response.coordinator_target_position
        self.response_plan = response.response_plan

        coordinator_elapsed_time  = time.time() - coordinator_start_time
        print(f"coordinatorQuery: took {coordinator_elapsed_time} seconds to evaluate")
        

        # log the prompt generated
        write_to_file(f"llm/log/coordinator_mind_prompt_generated_{self.agent_name}.txt", prompt)

        if self.debug:
            print("**********subtask_manager*************")
            print(self.agent_name + ": ")
            print(response)
            print("********************************")
            # print("Other Agent: ", self.other_agent_response)
        
        
        return self.human_intention, self.coordinator_target_position, self.response_plan


    # def coordinator_query(self, state):
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
    #     coordinator_start_time = time.time()
    #     response = self.coordinator_model.query(self.coordinator_id, "user", coordinatorReasoning, prompt, temp=0.2)

    #     self.human_intention = response.human_intention
    #     self.coordinator_rules = response.coordinator_adaptive_rules

    #     coordinator_elapsed_time  = time.time() - coordinator_start_time
    #     print(f"coordinatorQuery: took {coordinator_elapsed_time} seconds to evaluate")
        

    #     # log the prompt generated
    #     write_to_file(f"llm/log/coordinator_mind_prompt_generated_{self.agent_name}.txt", prompt)

    #     if self.debug:
    #         print("**********subtask_manager*************")
    #         print(self.agent_name + ": ")
    #         print(response)
    #         print("********************************")
    #         # print("Other Agent: ", self.other_agent_response)
        
        
    #     return self.human_intention, self.coordinator_rules
    

    def determine_subtask(self, state):
        """query the model appropriately for the optimal subtask

        Return: index of the subtask as defined according to self.subtasks
        """
        current_agent_state = state.players[self.agent_index].to_dict()
        other_agent_state = state.players[1 - self.agent_index].to_dict()
        world_state = state.to_dict().pop("objects")

        # obtain prompt layout from file
        prompt_layout = read_from_file(f"llm/layout/HRT/HRT_assign_subtask.txt")
        
        grid = self.mdp.terrain_mtx
        # # # last 5 human states and actions
        # human_trajectory = self.human_log[-5:]
        # human_trajectory_in_language = ""
        # for human_state, action in human_trajectory:
        #     human_trajectory_in_language += f"at Position: {human_state.position},human {self.action2string[action]}\n"

        # # last 5 human states and actions
        # robot_trajectory = self.agent_log[-5:]
        # robot_trajectory_in_language = ""
        # for robot_state, action in robot_trajectory:
        #     robot_trajectory_in_language += f"at Position: {robot_state.position},robot {self.action2string[action]}\n"
        #format prompt layout given the current states (this will replace all the placeholders in the prompt layout)
        # agent_plan = f"human wants to {self.human_intention}, you can, {self.response_plan} and first move to {self.coordinator_target_position}"
        prompt = self.format_subtask_assign_prompt_given_states(prompt_layout, 
                                                 world_state, 
                                                 current_agent_state, 
                                                 other_agent_state, 
                                                 self.graph_state,
                                                 grid= grid
                        
                                                )

        # message_to_other_chef = "Happy to work with you!"
        # print("promt", prompt)
        # query the model given prompt
        subtask_manager_start_time = time.time()
        system_prompt = read_from_file(f"llm/layout/HRT/HRT_assign_subtask_system.txt")
        response = self.subtask_manager_model.query_direct(subtask_managerReasoning, system_prompt, prompt, temp=0.2)

        subtask_manager_elapsed_time  = time.time() - subtask_manager_start_time
        print(f"subtask_managerMind: took {subtask_manager_elapsed_time} seconds to evaluate")
        self.subtask_manager_average_response_time = (self.subtask_manager_average_response_time * self.subtask_manager_response_count + subtask_manager_elapsed_time) / (self.subtask_manager_response_count + 1)
        self.subtask_manager_response_count += 1

        # log the prompt generated
        write_to_file(f"llm/log/subtask_manager_mind_prompt_generated_{self.agent_name}.txt", prompt)

        if self.debug:
            print("**********subtask_manager*************")
            print(self.agent_name + ": ")
            print(response)
            print("********************************")
            # print("Other Agent: ", self.other_agent_response)    
        
        return response.agent_subtask_id, response.human_subtask_id

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

    def pause_subtask_manager_mind(self):
        #TODO: pause threads
        pass
    def resume_subtask_manager_mind(self):
        #TODO:pause threads
        pass

    def record_human_log(self, human_state, world_state, action):
        self.human_log.append((human_state,  world_state, action))

    def record_agent_log(self, agent_state,  world_state, action):
        self.agent_log.append((agent_state,  world_state, action))