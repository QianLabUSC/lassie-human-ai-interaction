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
class GPT(BaseLLM):
    def __init__(self,key, model_name="gpt-4o-2024-11-20"):
        # create model using gpt-4
        super().__init__(key, model_name)
        self.client = OpenAI(
            api_key=key,  
        )
 
    def query(self, conversation_messages,temp: int = 0.4, max_tokens=1024):
        chat_completion = self.client.chat.completions.create(
            messages=conversation_messages,
            model=self.model_name,
            temperature=temp,
            max_tokens=max_tokens,
        )
        return chat_completion.choices[0].message.content
    def query_json(self, conversation_messages, response_format, temp: int = 0.4,max_tokens=1024):
        request_reply = self.client.beta.chat.completions.parse(
            model=self.model_name,
            messages=conversation_messages,
            temperature=temp,
            response_format=response_format,
            max_tokens=max_tokens,
        )
        return request_reply.choices[0].message.parsed

class DialogueManager:
    """
    Orchestrates the dialogue and maintain a subtask graph.:
    - Holds the node graph
    - Delegates to different LLMs
    - Maintains conversation state
    """
    def __init__(self, model : GPT, node_graph: Graph = None,max_history_length=10):
 
        self.model  = model
        # self.node_graph = Graph()
        self.node_graph = node_graph
        self.conversation_history = []  
        self.max_history_length = max_history_length
    def set_node_graph(self, node_graph: Graph):
        self.node_graph = node_graph
 
    def get_node_graph(self):
        return self.node_graph
        
    def _update_graph_from_response(self, json_response: str):
        """
        Example: parse the LLM text to see if we should add or remove nodes.
        You can define your own format. For instance, if the LLM says:
        'Add a node with ID=5 and label=Chop Onions' 
        or 
        'Remove node 3 because it is irrelevant.'
        you can do that here.
        """
                #save as json to graph.json 

        try:
            data = json.loads(json_response)
            self.node_graph.load_from_json(data)
            print(data)
        except json.JSONDecodeError:
            raise ValueError("LLM response is not in JSON format.")
    def _trim_history(self):
        """Keep only the last `max_history_length` messages in conversation_history."""
        if len(self.conversation_history) > self.max_history_length:
            # Slice from the end
            self.conversation_history = self.conversation_history[-self.max_history_length:]

    # def initialize_from_file(self, file_path: str):
    #     """
    #     for debugging purposes,
    #     Initialize the dialogue manager from a file.
    #     """
    #     self.llm_agent.initilize_Graph(state)
    #     self.node_graph = Graph(mlam)
    def ask_question_json(self, question: str) -> str:
        """
        Send user question to the LLM in JSON mode and return the response (JSON text).
        """
        from pydantic import BaseModel
        from typing import List

        class VertexItem(BaseModel):
            id: int
            name: str
            target_position: List[List[int]]
            task_type: str
            parent_subtasks: List[int]
            next_subtasks: List[int]
            status: str

        class GraphResponse(BaseModel):
            vertex: List[VertexItem]
            edge: List[List[str]]

        # 1) Append user message
        self.conversation_history.append({"role": "user", "content": question})
        self._trim_history()  # ensure we don't exceed max_history_length

        # 2) Query the model in JSON format
        answer = self.model.query_json(
            self.conversation_history,
            GraphResponse,
            temp=0.4
        )

        # 3) Append assistant message
        self.conversation_history.append({"role": "assistant", "content": str(answer)})
        self._trim_history()

        return answer

    def update_graph(self, text_response: str):
        print("update graph")
        #TODO: update
        """
        based on response and update node_graph with llm .
        """
        
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
            current_graph_json  = self.node_graph.to_json()
            model_prompt = f"The current node graph is:\n{current_graph_json}\nUser says: {user_input}\n, based on what user said, update the graph. output in json format" 
            print(f"model input: {model_prompt}")
            # 2) Query the LLM
            response = self.ask_question_json(model_prompt)
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


