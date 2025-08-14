"""
an universal dialogue manager for all envirionments: overcooked, science explore, etc.
"""
from llm.subtask_graph import SubTask, Graph
from llm.base_lauguage_agent import LLMModel
from llm.utils import read_from_file, write_to_file
from llm.llm_api import graph_generation,BaseLLM
import re
import json
from openai import OpenAI
import time
import os
from json.decoder import JSONDecodeError
from llm.llm_api import GPT
from pydantic import BaseModel
from typing import List

#helper for parsing json from string
def parse_json(self, json_response: str):
    try:
        data = json.loads(json_response)
        return data
    except json.JSONDecodeError:
        raise ValueError("LLM response is not in JSON format.")

#TODO: add pydantic schema to a single file 
class coordinationType(BaseModel):
    query_type: int
    message_to_human: str
class VertexItem(BaseModel): # pydantic schema for vertex
    id: int
    name: str
    target_position: List[List[int]]
    task_type: str
    parent_subtasks: List[int]
    next_subtasks: List[int]
    status: str
    notes: str
class GraphResponse(BaseModel): # pydantic schema for graph response
    vertex: List[VertexItem]
    edge: List[List[str]]
class GraphResponseHuman(BaseModel): # pydantic schema for graph response along with a message to human
    graph: GraphResponse
    message_to_human: str
class ActiveSuggestion(BaseModel):
   suggestions: str

    
class DialogueManager:
    """
    Orchestrates the dialogue and maintain a subtask graph.:
    - Holds the node graph
    - Delegates to different LLMs
    - Maintains conversation state
    """
    def __init__(self, model : BaseLLM , node_graph: Graph,max_history_length=10):
 
        self.model  = model
        self.node_graph = node_graph #init_graph
        self.max_history_length = max_history_length
        self.conversation_history = []  #TODO:maybe store a list of chats

    # def generate_dialogue_id(self):
    #     return str(uuid.uuid4())
    def init_dialogue(self):
        #TODO: add sysyem prompt to the dialogue
       
        example_subtasks = '''
                Available_subtask_types:
                0: "PUTTING", 1: "GETTING", 2: "COOKING"

                Available_subtask_status:
                0: "unknown", 1: "ready_to_execute", 2: "success", 3: "failure", 4: "not_ready", 5: emergency
                example_subtask = {
                "id": int,  # Unique ID of the subtask start from 0
                "name": string,  # Task description, e.g. "Get onion"
                "target_position_id": list[int],  # IDs of target positions selected from provided locations
                "task_type": int,  # Integer representing the task type (e.g., 1 = GETTING, refer to all avaiable types)
                "task_status": int,  # Integer representing the task status (e.g., refer to all avaiable status, but you only to judge if this subtask has been finished, if not, leave unknown, I will handle it based on graph)
                "notes": str, # if human has some preferences related to this subtask, you should write it in a very short sentences here, e.g. human preferes to do this task
                "parent_subtask": list[int]  #  Only list of IDs of parent subtasks that are a must and reprequisite to this task, (leave empty if no required subtasks before this, or other agent can help do this)
            }'''
        self.conversation_history.append({"role": "system", "content": "You are an robot assistant that breaks down tasks into steps and checks with the user when you are unsure about any part. At each step, if you are uncertain or the information is incomplete, ask the user for clarification or confirmation before proceeding."})
            
        self.conversation_history.append({"role": "system", "content": "You are now collaborates with human, and you are responsible for make the subtask graph for achieving the recipe, which contains tasks for both human and you, the graph should follow this form, {example_subtasks} \
                                          human will query you with some lauguage instructions, for every human query, first return the coordination type as \
                                          1 if human wants to change the coordination graph that will hold for continous collaborating\n \
                                          2 if human want to indicate their preference\n \
                                          3 if you human want to assign temporary tasks\n \
                                          you should return 0 if you are even a little bit uncertain, you should send a short message to explicitly ask which type."})
        


    def active_query(self, prompt):
        graph_j = self.node_graph.to_json()
        prompt = prompt.replace("{current_graph}", str(graph_j))
        system_prompt = read_from_file(f"llm/layout/HRT/HRT_active_suggestion_system.txt")
        response = self.model.query_direct(ActiveSuggestion, [{"role": "system", "content":system_prompt},{"role": "user", "content": prompt}], temp=0.38)

        print(response)
        
        

        # log the prompt generated
        write_to_file(f"llm/log/activesuggestion.txt", prompt)
        
        return response


    def process_conversation(self):
        """
        process user conversation, modify node graph and 
        return user readable message 
        """
        # print("----conversation history----")
        print(self.conversation_history)
        response = self.model.query_direct(
            coordinationType,
            self.conversation_history
        )
        write_to_file(f"llm/log/conversation.txt", str(self.conversation_history))
        if response.query_type == 0:
            self.conversation_history.append({"role": "assistant", "content": response.message_to_human})
            return response.message_to_human
        elif response.query_type == 1:
            graph_j = self.node_graph.to_json()
            self.conversation_history.append({"role": "user", "content": "Here is the current graph:" + str(graph_j)})
            self.conversation_history.append({"role": "user", "content": "human want to change the graph to create a long coordinating strategy, you \
                                                should update the graph and adding or inserting nodes based on needs."})
            response = self.model.query_direct(
                GraphResponseHuman,
                self.conversation_history,
            )
<<<<<<< HEAD
            write_to_file(f"llm/log/conversation_type1.txt", str(self.conversation_history))
            print("----response----")
            print(response)
=======
            # print("----response----")
            # print(response)
>>>>>>> ede815e598eb982e728a13be9defe2ed0fc7a520
            self.update_graph(response.graph)
        elif response.query_type == 2:
            graph_j = self.node_graph.to_json()
            self.conversation_history.append({"role": "user", "content": "Here is the current graph:" + str(graph_j)})
            self.conversation_history.append({"role": "user", "content": "human report preferences for different tasks, you should add to notes for later task assignment,  keep in mind that you are the robot."})
            response = self.model.query_direct(
                GraphResponseHuman,
                self.conversation_history,
            )
            write_to_file(f"llm/log/conversation_type2.txt", str(self.conversation_history))
            print("----response----")
            print(response)
            self.update_graph(response.graph)
        elif response.query_type == 3:
            graph_j = self.node_graph.to_json()
            self.conversation_history.append({"role": "user", "content": "Here is the current graph:" + str(graph_j)})
            self.conversation_history.append({"role": "user", "content": "human querys a temporary subtask, you should add a node with status emergency on the basis of original graph, dont change original graph\
                                        add in notes who should execute, keep in mind that you are the robot."})
            response = self.model.query_direct(
                GraphResponseHuman,
                self.conversation_history,
            )
            write_to_file(f"llm/log/conversation_type3.txt", str(self.conversation_history))
            print("----response----")
            print(response)
            self.update_graph(response.graph)
        else:
            return response.message_to_human

        return response.message_to_human
        
    def receive_message(self, message: str) -> str:
        """
        Send user question to the LLM in JSON mode and return the response (JSON text). 
        """        
        self.conversation_history.append({"role": "user", "content": message})
        self._trim_history()  # ensure we don't exceed max_history_length

    def robot_message(self, message: str) -> str:
        """
        Send user question to the LLM in JSON mode and return the response (JSON text). 
        """        
        self.conversation_history.append({"role": "assistant", "content": message})
        self._trim_history()  # ensure we don't exceed max_history_length
        

    def set_node_graph(self, node_graph: Graph):
        self.node_graph = node_graph
 
    def get_node_graph(self):
        return self.node_graph
            # self.node_graph.load_from_json(data)
    def _trim_history(self):
        """Keep only the last `max_history_length` messages in conversation_history."""
        if len(self.conversation_history) > self.max_history_length:
            # Slice from the end
            self.conversation_history = self.conversation_history[-self.max_history_length:]

    def clear_dialog(self):
        self.conversation_history = []

    # def initialize_from_file(self, file_path: str):
    #     """
    #     for debugging purposes,
    #     Initialize the dialogue manager from a file.
    #     """
    #     self.llm_agent.initilize_Graph(state)
    #     self.node_graph = Graph(mlam)

    def update_graph(self, GraphResponse: GraphResponse):
        print("update graph based on conversation")
        return self.node_graph.load_from_json(GraphResponse.model_dump())
        
    def interactive_loop(self):
        """
        Example interactive loop:
        1. Prompt the user for input about strategy or tasks.
        2. Send user input plus current graph state to the LLM.
        3. Parse LLM's response to update the node graph.
        4. (Optional) Re-visualize or print the updated node graph.
        5. Repeat until user decides to exit.
        """
        print("\n=== Strategy Planning Dialogue (type 'quit' to end) ===")
        while True:
            user_input = input("USER >> ")
            if user_input.lower() in ["quit", "exit"]:
                print("Ending dialogue.")
                break
 

            # Construct a simple prompt that includes user_input + a textual summary of the node graph
            # If your node_graph object has a method to summarize as text, use it:
            # 2) Query the LLM
            response = self.change_node_query(self.node_graph,model_prompt)
            print(f" 😄: {response}")
 
            # 4) Optionally parse or interpret the LLM's text to figure out how to update the graph
            #    e.g., look for certain keywords or JSON structures. 
            #    Here’s a placeholder parse function:
            self._update_graph_from_response(response.json())
            self.node_graph.draw_graph("updated_graph" + str(time.time()) + ".png")
            print(f"[Graph updated] \n")


    # def interactive_loop(self):
    #         """
    #         Runs an interactive CLI loop:
    #         - ask user for input
    #          - get answer
    #         - update knowledge graph
    #         """
    #         print("Welcome to the QA dialogue system! Type 'exit' to quit.")

    #         while True:
    #             user_input = input("\n > ")
    #             if user_input.lower() in ["exit", "quit"]:
    #                 print("Exiting...")
    #                 break
 
    #             # Call the LLM
    #             

    #             # Update node graph
    #             self.update_graph(response)
    #             print(f"[NodeGraph Status] {self.node_graph}")



if __name__ == "__main__":
    from config import initialize_config_from_args, StudyConfig
    from planning.planners import LLMMediumLevelActionManager
    from llm.llm_agent import HRT
    from overcooked_ai_py.agents.agent import Agent
    from logger import Logger
    from overcooked_pygame import OvercookedPygame


    study_config = initialize_config_from_args()
    print("Study Configuration Initialized:")
    print(f"Participant ID: {study_config.participant_id}")
    print(f"User Mode: {study_config.user_mode}")
    print(f"Layout: {study_config.layout_name}")
    FULL_PARAMS = {
        "start_orientations": False,
        "wait_allowed": True,
        "counter_goals": study_config.world_mdp.terrain_pos_dict["X"],
        "counter_drop": study_config.world_mdp.terrain_pos_dict["X"],
        "counter_pickup": study_config.world_mdp.terrain_pos_dict["X"],
        "same_motion_goals": True,
    }
    mlam = LLMMediumLevelActionManager(study_config.world_mdp,FULL_PARAMS)
    # # populate details
    agent_details = study_config.agent_details
    # dm = DialogueManager(GPT(key="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"))
    # print(study_config.world_mdp.get_standard_start_state())
    # # print(study_config.mlam)
    agent1 = HRT(
        agent_details['agent_name'],
        agent_details['action_system'],
        agent_details['action_prompt_template'],
        agent_details['subtask_system'],
        agent_details['subtask_prompt_template'],
        order_list = agent_details['order_list'],
        coordinator_model="gpt",
        subtask_manager_model="gpt",
        env=study_config.base_env,
        mode=study_config.user_mode,
        mlam=mlam,
        )
    agent1.set_agent_index(0)

    # agent.initilize_Graph(study_config.world_mdp.get_standard_start_state())
    # dm.set_node_graph(agent.graph_state)
    
    # dm.interactive_loop()
    # print(dm.conversation_history)
    
    
    study_config = initialize_config_from_args()
    print("Study Configuration Initialized:")
    print(f"Participant ID: {study_config.participant_id}")
    print(f"User Mode: {study_config.user_mode}")
    print(f"Layout: {study_config.layout_name}")
    FULL_PARAMS = {
        "start_orientations": False,
        "wait_allowed": True,
        "counter_goals": study_config.world_mdp.terrain_pos_dict["X"],
        "counter_drop": study_config.world_mdp.terrain_pos_dict["X"],
        "counter_pickup": study_config.world_mdp.terrain_pos_dict["X"],
        "same_motion_goals": True,
    }
    agent1.initilize_Graph(study_config.world_mdp.get_standard_start_state())    
    
    dm = DialogueManager(GPT(key="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"),node_graph=agent1.graph_state)
    dm.node_graph.draw_graph("init_graph_b")
    # --- Node Graph & Dialogue Setup ---
 
    # Interactive loop: user & LLM plan strategy, update the node graph
    print("\n[BEGIN STRATEGY DIALOGUE BEFORE GAME STARTS]")
    dm.interactive_loop()
    print("[END STRATEGY DIALOGUE]\n")
    # --- Node Graph & Dialogue Setup ---


