
from config import initialize_config_from_args, StudyConfig
from logger import Logger
from overcooked_pygame import OvercookedPygame
from llm.llm_agent import ManagerReactiveModel
from planning.planners import LLMMediumLevelActionManager
from overcooked_ai_py.agents.agent import Agent
from time import time
import os 
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'overcooked_ai/src')))

if __name__ == "__main__" : 
    study_config = initialize_config_from_args()
    print("Study Configuration Initialized:")
    print(f"Participant ID: {study_config.participant_id}")
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
    # populate details
    agent_details = study_config.agent_details

    agent1 = Agent()
    agent2 = ManagerReactiveModel(
        mlam,
        agent_details['agent_name'],
        agent_details['action_prompt_template'],
        agent_details['subtask_prompt_template'],
        reactive_model=study_config.reactive_model,
        manager_model=study_config.manager_model
        )
    # print(agent2.shared_state)
    agent1.set_agent_index(0)
    agent2.set_agent_index(1)
    
    # Initialize logging
    logger = Logger(study_config, study_config.log_file_name, agent1=agent1, agent2=agent2)

    # Run the game
    start_time = time()
    theApp = OvercookedPygame(study_config, agent1, agent2,logger, gameTime=study_config.total_time)
    
    theApp.on_execute()
    print("It took {} seconds for playing the entire level".format(time() - start_time))