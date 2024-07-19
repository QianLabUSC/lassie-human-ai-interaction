import os
import sys
from time import time
import json

from config import StudyConfig, initialize_config_from_args
# from llm.llm_agent import LLMModel
from logger import Logger
from overcooked_pygame import OvercookedPygame

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "overcooked_ai/src"))
)

import json

from overcooked_ai_py.agents.agent import GreedyHumanModel,RandomAgent
from planning.planners import LLMMediumLevelActionManager
from llm.utils import write_to_file
from llm.llm_agent import LLMModel

class ActionLogger:
    def __init__(self):
        """simple data logger class for actions"""

        self.action_history = []

    def log_action(self, mlam, state, action_taken, agent_index):
        """log action taken by agent

        Args:
            log_action (str): action taken by agent
            agent_index (int): index of agent
        """

        # obtain relevant information
        player = state.players[agent_index].to_dict()
        other_player = state.players[1 - agent_index].to_dict()
        world_state = state.to_dict().pop("objects")
        grid = mlam.mdp.terrain_mtx

        # construct new action entry
        self.action_history.append(
            {
                "action": action_taken,
                "agent": player,
                "other_agent": other_player,
                "world_state": world_state,
                "grid": grid,
                # "order_list": state.order_list,
            }
        )

    # def export_to_json(self, file_path):
    #     """export results to a json file"""
    #     contents = json.dumps(self.action_history, indent=4)
    #     write_to_file(file_path, contents)
    def subtask_export_to_json(self,file_path):
        """
        add subtask to exported json
        """
        self.action_history = rawdata_add_subtask(self.action_history)
        contents = json.dumps(self.action_history, indent=4)
        write_to_file(file_path, contents)
        

format_llm_agent = LLMModel("formatter","llm/layout/reactive_with_analysis_gpt.txt","llm/layout/manager_06172024v1.txt","gemma","gemma")

def main(args):
    study_config = initialize_config_from_args()

    FULL_PARAMS = {
        "start_orientations": False,
        "wait_allowed": False,
        "counter_goals": study_config.world_mdp.terrain_pos_dict["X"],
        "counter_drop": study_config.world_mdp.terrain_pos_dict["X"],
        "counter_pickup": study_config.world_mdp.terrain_pos_dict["X"],
        "same_motion_goals": False,
    }
    mlam = LLMMediumLevelActionManager(study_config.world_mdp, FULL_PARAMS)

    # action logger
    actionlogger = ActionLogger()

    # create 2 greedy agents
    agent1 = RandomAgent()

    agent2 = GreedyHumanModel(mlam, auto_unstuck=False, actionlogger=actionlogger)

    agent1.set_agent_index(0)
    agent2.set_agent_index(1)

    #special agent for formatting prompt
    format_llm_agent.set_mdp(study_config.world_mdp)

    study_config.world_mdp
    # Initialize logging
    logger = Logger(
        study_config, study_config.log_file_name, agent1=agent1, agent2=agent2
    )

    # Run the game
    start_time = time()
    theApp = OvercookedPygame(
        study_config, agent1, agent2, logger, gameTime=study_config.total_time
    )
    theApp.on_execute()
    print("It took {} seconds for playing the entire level".format(time() - start_time))

    experiment_name = "greedy+random"

    actionlogger.subtask_export_to_json(f"greedy_data/{experiment_name}.json")
    log_name = "greedy+random_prompt+response"
    convert_to_prompt("greedy_data/greedy+random.json",f"greedy_data/{log_name}.json")
#import util files
action2string= {
        "[1, 0]": "move_right",
        "[-1, 0]":"move left",
        "[0, -1]": "move up",
        "[0, 1]": "move downn",
        "interact": "interact",
        "[0, 0]": "stay",
    }


def rawdata_add_subtask(data):
    #add subtask_in_mind in ecery frame 
    for frame in data:        
        # appending the data
        frame.update({"subtask_in_mind":"none"})
    prev_subtask = "Do nothing"
    subtask= "Do nothing"

    for frame in reversed(data):
        
        agent_pos = frame["agent"]["position"]
        # print(f"pos: {agent_pos}")
        agent_ori = frame["agent"]["orientation"]
        # print(agent_ori)
        held_object = frame["agent"]["held_object"]
        if held_object:
            held_obect_name = held_object["name"]
        # print(held_obect_name)
        faced_grid_pos = [agent_pos[0] + agent_ori[0],agent_pos[1] + agent_ori[1]]
        # print(f"faced{faced_grid_pos}") 
        world_state = frame["world_state"]
        grid_layout = frame["grid"]
        item = grid_layout[faced_grid_pos[1]][faced_grid_pos[0]]

        if item == "X":
            item_name = "Counter"
            for counter in world_state:
                if counter["position"][0] == faced_grid_pos[1] and counter["position"][1] == faced_grid_pos[0]:
                    item_name = counter['name']
                    print(item_name)
                    # onion, tomato, dish, soup
        elif item == "P":
            item_name = "Pot"
        elif item == "D":
            item_name = "Dish dispenser"
        elif item == "O":
            item_name = "Onion dispenser"
        elif item == "T":
            item_name = "Tomato dispenser"
        elif item == "S":
            item_name = "Delivery location"
        else:
            item_name = "Empty square"

        if frame["action"] == "interact":
            subtask = determine_subtask(agent_pos,agent_ori,item_name,held_object)
            prev_subtask =subtask
        else:
            subtask = prev_subtask
        frame.update({"subtask_in_mind":subtask})
    return data

def determine_subtask(pos,ori,faced_item_name,held_object):
    if held_object is None:
        if faced_item_name == "onion" or faced_item_name == "Onion dispenser":
            return "Pick up onion"
        elif faced_item_name == "tomato" or faced_item_name == "Tomato dispenser":
            return "Pick up tomato"
        elif faced_item_name == "soup":
            return "Pick up soup"
        elif faced_item_name == "Dish dispenser" or faced_item_name == "dish":
            return "Pick up dish"
        elif faced_item_name == "Pot":
            return "Start cooking pot"            
    else:
        held_object_name = held_object["name"]
        # print(f"faced item name: {faced_item_name}")
        # print(f"held object name: {held_object_name}")
        if held_object["name"] == "onion" and faced_item_name == "Pot":
                return "Put onion in pot"
        elif held_object["name"] == "tomato" and faced_item_name == "Pot":
            return "Put tomato in pot"
        elif held_object["name"] == "soup" and faced_item_name == "Delivery location":
            return "Deliver soup"
        elif held_object["name"] == "dish" and faced_item_name == "Pot":
            return "Pick up soup with dish"
        elif faced_item_name == "Counter":
            return "Place holding object on counter"
        else:
            return "Do nothing"
        
def read_from_file(file_path):
    """helper function to read content from a file"""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

def convert_to_prompt(input_file_path, output_file_path):
        # load data
    def load_data(data_path):
        with open(data_path, 'r') as file:
            data = json.load(file)
        return data
    data = load_data(input_file_path)
    # print(data)
    log = []
    for dict in data: # i in range(len(dataset['train'])):
        # dataset.append()
        pl = read_from_file("llm/layout/reactive_with_analysis_gpt.txt")
        pl = pl.replace("{recipe_book}", "    Recipe 0: Requires 3 ingredients: onion, onion, onion. The ingredients should be placed in a pot and cooked to make the soup.")
        grid_layout=dict["grid"]
        #convert format
        orix = dict['agent']['orientation'][0]
        oriy = dict['agent']['orientation'][1]
        dict['agent']['orientation'] = (orix,oriy)
        orix = dict['other_agent']['orientation'][0]
        oriy = dict['other_agent']['orientation'][1]
        dict['other_agent']['orientation'] = (orix,oriy)
        #format prompt

        pl = format_llm_agent.format_prompt_given_states(pl,dict['world_state'],dict['agent'],dict['other_agent'],current_subtasks=dict['subtask_in_mind'],grid=grid_layout)
        # print(pl)
        action2string= {
            "[1, 0]": "move right",
            "[-1, 0]":"move left",
            "[0, -1]": "move up",
            "[0, 1]": "move down",
            "interact": "interact",
            "[0, 0]": "stay",
        }
        action2index ={
            "[1, 0]" :"[1]",
            "[-1, 0]": "[2]",
            "[0, -1]":"[3]",
            "[0, 1]":"[4]",
            "interact":"[5]",
            "[0, 0]":"[6]",

        }

        expected_response = f"{action2string[str(dict['action'])]} {action2index[str(dict['action'])]}"

        log.append(
                {
                    "prompt": pl,
                    "response": expected_response,
                }
            )
    contents = json.dumps(log, indent=4)
    write_to_file(output_file_path, contents)


if __name__ == "__main__":
    # obtain args from argparser
    args = {"experiment_name": "test2"}
    main(args)


