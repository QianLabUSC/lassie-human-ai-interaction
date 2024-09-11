import os
import shutil
import argparse
import datetime
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'overcooked_ai/src')))
from time import time
from llm.utils import read_from_file, write_to_file
from mdp.llm_overcooked_mdp import LLMOvercookedGridworld
from overcooked_ai_py.utils import read_layout_dict
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv

from constants import MAX_STEPS

class StudyConfig:
    def __init__(self, args):
        # end at subtask mode
        self.end_at_subtask = args.end_at_subtask
        # obtain args
        self.participant_id = args.participant_id
        self.user_mode=args.user_mode
        self.layout_name = args.layout
        layout_file_name = self.layout_name + ".layout"
        self.log_file_name = args.log_file_name
        self.record_video = args.record_video
        self.total_time = args.total_time
        script_dir = os.path.dirname(os.path.abspath(__file__))

        self.reactive_model = args.reactive_model
        self.manager_model = args.manager_model
        # Copy layout to Overcooked AI code base
        path_from = os.path.join(
            script_dir, "data", "layout", layout_file_name)

        # Construct the absolute path by joining the script directory with the relative path
        path_to = os.path.join(script_dir, "..", "overcooked_ai", "src", 
                               "overcooked_ai_py", "data", "layouts", layout_file_name)
        shutil.copy(path_from, path_to)

        # create overcooked environments
        self.world_mdp = LLMOvercookedGridworld.from_layout_name(
                self.layout_name)

        self.base_env = OvercookedEnv.from_mdp(
            self.world_mdp)

            ### chage LLM description file to include the layout
        base_layout_params = read_layout_dict(self.layout_name)

        # obtain the language formatted orders
        order_list = base_layout_params["start_all_orders"]
        orders = []
        for number in range(len(order_list)):
            orders.append(f"Recipe {number}: Requires {len(order_list[number]['ingredients'])} ingredients: " + ", ".join(order_list[number]['ingredients']) + ". The ingredients should be placed in a pot and cooked to make the soup.")
        orders_formatted_in_language = "\n".join(orders)



        # partially fill in the prompt template with base layout information, and populate to prompt template file
        action_file_path = f"llm/layout/{ args.reactive_prompt}.txt" 
        action_prompt = read_from_file(action_file_path)

        # replace base prompt from original file
        action_prompt = action_prompt.replace("{recipe_book}", orders_formatted_in_language)
        write_to_file(f'llm/prompts/action_prompt_template_with_layout_{args.agent_name}.txt', action_prompt)

        subtask_file_path = f"llm/layout/{ args.manager_prompt}.txt" # 'llm/layout/05292024v3.txt'

        subtask_prompt = read_from_file(subtask_file_path)

        # replace base prompt from original file
        subtask_prompt = subtask_prompt.replace("{recipe_book}", orders_formatted_in_language)
        write_to_file(f'llm/prompts/subtask_prompt_template_with_layout_{args.agent_name}.txt', subtask_prompt)

        subtask_system_path = f"llm/layout/{ args.manager_system}.txt"
        action_system_path = f"llm/layout/{ args.reactive_system}.txt"
        self.agent_details = {
                "agent_name": args.agent_name,
                "subtask_prompt_template": f'llm/prompts/subtask_prompt_template_with_layout_{args.agent_name}.txt',
                "action_prompt_template": f'llm/prompts/action_prompt_template_with_layout_{args.agent_name}.txt',
                "subtask_system": subtask_system_path,
                "action_system": action_system_path,
            }

def validate_mode(value):
    value = int(value)
    if value < 1 or value > 4:
        raise argparse.ArgumentTypeError("Mode must be between 1 and 4")
    return value

def initialize_config_from_args():
    parser = argparse.ArgumentParser(
        description='Initialize configurations for a human study.')
    
    ### Args for the game setup ###
    parser.add_argument('--user_mode', type=validate_mode, default=1,
                        help='Please select the mode in which you want to run experiment, Modes from (1-4)')

    ### Args for the game setup ###
    parser.add_argument('--layout', type=str, default='0_trial_option_coordination',
                        help='List of tasks to be performed in the study')
    parser.add_argument('--total_time', type=int, default=MAX_STEPS,
                        help='Total time to given to complete the game')

    # The following game config options are still undergoing construction
    # parser.add_argument('--served_in_order', type=bool, help='Complete the order list in order')
    # parser.add_argument('--single_player', type=bool, help='Single player mode: one human controlled agent collaborating with a modeled greedy agent')

    # add argument for the layouts
    parser.add_argument('--reactive_prompt', type=str, default='reactive_user_prompt_chain_of_thought',
                        help='action prompt layout for agent ')
    parser.add_argument('--reactive_system', type=str, default='reactive_system_prompt')    
    parser.add_argument('--manager_prompt', type=str, default='manager_user_prompt',
                        help='subtask prompt layout for agent')
    
    parser.add_argument('--manager_system', type=str, default='manager_system_prompt',
                        help='subtask prompt layout for agent')
    
    # args for model selection
    parser.add_argument('--manager_model', type=str, default='rule',
                        help='LLM model selection')
    parser.add_argument('--reactive_model', type=str, default='gpt',
                        help='LLM model selection')
    # parser.add_argument('--manager_model', type=str, default='ollama',
    #                     help='LLM model selection')
    # parser.add_argument('--reactive_model', type=str, default='ollama',
    #                     help='LLM model selection')
    # Deprecated: No agent 1 since we use human for agent 1
    # # args for agent names (primarily logging purposes)
    # parser.add_argument('--agent1_name', type=str, default='human',
    #                     help='Name of the human')
    parser.add_argument('--agent_name', type=str, default='agent',
                        help='Name of the second agent')

    ### Args for the study ###
    parser.add_argument('--participant_id', type=int,
                        help='ID of participants in the study', default=0)
    parser.add_argument('--log_file_name', type=str,
                        default='', help='Log file name')
    parser.add_argument('--record_video', dest='record_video',
                        action='store_true', help='Record video during replay')
    parser.add_argument('--no-record_video', dest='record_video',
                        action='store_false', help='Do not record video during replay')
    # parser.add_argument('--comm',dest='comm', action='store_true', help='Use communication between agents')
    parser.add_argument('--end_at_subtask', dest='end_at_subtask',
                        action='store_true', help='End the game when the subtask is completed')
    args = parser.parse_args()

    if args.log_file_name == '':
        x = datetime.datetime.now()
        datestr = x.strftime("%m-%d-%f")
        print(datestr)
        args.log_file_name = '-'.join([str(args.participant_id), args.layout, datestr])

    return StudyConfig(args)