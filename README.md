# ChatOverCook 
<p align="center">
  <img src="./images/layout.png" width="100%"> 
  <i>Overcooked AI chat interface</i>
</p>

## Introduction
This is implementation for a final project from course CSCI641 [Computational Human-Robot Interaction](https://www.stefanosnikolaidis.net/comphri2024.html). We used existing implementation of the [Overcooked AI](https://github.com/HumanCompatibleAI/overcooked_ai) and incorporated a chat interface for human to interact with LLMs.  

## Installation
Create a Conda environment with Python 3.10 

``` 
conda create -n overcooked_ai python=3.10
conda activate overcooked_ai
```

Clone the repo recursively
```
git clone --recursive https://github.com/QianLabUSC/llm-flowingtheory
```

To use overcooked ai environment
```
cd overcooked_llm/overcooked_ai
pip install -e .
```
Install required dependency
```
cd overcooked_llm
pip install -r requirements.txt
```

To use Open AI API, create your own API key and create an .env file located in the overcooked_llm folder contains:
```
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"
```


## Testing out ChatOverCooked Interface

python main.py --model gpt --low-level-query --prompt_layout_agent1 06042024v1_atomic_action --prompt_layout_agent2 06042024v1_atomic_action --layout random --record_video
To test with different agent level (default, passive, active)
```
cd src/
python user_study.py --agent_level default --layout 0_trial_option_coordination --participant_id 001 --total_time 100 --record_video
python user_study.py --agent_level passive_mentee --layout 0_trial_option_coordination --participant_id 001 --total_time 100 --record_video
python user_study.py --agent_level active_supervisor --layout 0_trial_option_coordination --participant_id 001 --total_time 100 --record_video
```
To test with different layout (easy, medium, hard)
```
python user_study.py --agent_level default --layout 1_cramped_room_easy --participant_id 001 --total_time 100 --record_video
python user_study.py --agent_level default --layout 2_narrow_corridor_medium --participant_id 001 --total_time 100 --record_video
python user_study.py --agent_level default --layout 3_spiral_hard --participant_id 001 --total_time 100 --record_video
```

the user study log and video will be saved under overcooked_llm/src/user_study

## Game Introduction 
The game is between a LLM-driven agent with a human player. Similiar to Original overcooked ai, we added a chat interface where human will be able to chat with another agent and change agent behavior based on their preferences. An example instruction would be: "you should only make onion soup, I will serve dishes". 
### Agent styles: 
There are three type of agents: active Supervisor, Passive mentee, and default receptive agent. 

- Default: a greedy agent collaborates with humans without any communication.
- Passive leader-follower style: agent receives and acts upon instructions based on human preferences.
- Active peer-to-peer style: beside receive and acts upon instructions. it also provide feedback based on human past behavior.
  
### Interface: 
You will be control blue hat agent. use up/down/left/right key to move and space bar to interact with environment. 
in case of communicate with another agent, click on pause button, then type in your instruction in the text input box. afterwards, agent will be updated once the game is resumed. chat session is shown on the right. Vertical coordinates indicating row(x) numbers and  horizontal coordinates indicating column(y) numbers. 


## Code Structure Overview

`src` contains:

`data/`:

- `config/`: contains Overcooked game initial configuration files
- `layout/`: contains different layout files

`llm/`:

- `task/`: LLM prompts. des_1_layout.txt contains the prompt that is actually being fed to LLM.
- `llm_agent.py`: customized greedy human model. injected with LLM modified code.
- `llm_api.py`: LLM API calls with text/code extraction. with code reflection implementation.
- `utils.py`: LLM prompt tips.
  
`mdp/`:

- `llm_overcooked_mdp.py`: LLM agent related code inheritance. added human action history logging for active agent. 

`user_study/`:

- `log/`: contains folders of logs/video for each participants.
  
`planning/`:

- `planners.py`: customized MediumLevelActionManager for LLM. enable more actions to use.

`user_study.py`: script for start a chatovercook pygame session.
`data_process.ipynb`: post experiment data process script.  


Feel free to reach out to @boshenzh for any questions. 