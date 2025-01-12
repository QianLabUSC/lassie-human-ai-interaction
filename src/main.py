
from config import initialize_config_from_args, StudyConfig
from logger import Logger
from overcookedgamebackup import OvercookedPygame
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
    # populate details
    agent_details = study_config.agent_details
    print(agent_details['subtask_prompt_template'])
    agent1 = Agent()
    agent2 = ManagerReactiveModel(
        agent_details['agent_name'],
        agent_details['action_system'],
        agent_details['action_prompt_template'],
        agent_details['subtask_system'],
        agent_details['subtask_prompt_template'],
        reactive_model=study_config.reactive_model,
        manager_model=study_config.manager_model,
        env=study_config.base_env,
        mode=study_config.user_mode,
        mlam=mlam,
        )
    # Access the extracted order_list directly from study_config
    order_list = study_config.order_list
    print("Order list extracted from the layout:")
    print(order_list) #[{'ingredients': ['onion', 'onion', 'tomato']}]

    # For now test with these recipe and after that get receipe from study_config
    recipe = {
        "ingredients": order_list[0]['ingredients']
    }
    
    # test_recipe = study_config.recipe
    
    # Calculate the total cost of subtasks for the recipe
    total_subtasks_cost = agent2.calculate_subtasks_cost_for_recipe(recipe, study_config.world_mdp.terrain_mtx)
    print(f"Total subtasks required to complete the recipe: {total_subtasks_cost}")

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