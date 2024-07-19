#!/bin/bash

# Configuration 0
python user_study.py --agent_level default --layout 0_trial_option_coordination --participant_id 001 --total_time 100 --record_video
python user_study.py --agent_level passive_mentee --layout 0_trial_option_coordination --participant_id 001 --total_time 100 --record_video
python user_study.py --agent_level active_supervisor --layout 0_trial_option_coordination --participant_id 001 --total_time 100 --record_video

# Configuration 1
python user_study.py --agent_level default --layout 1_cramped_room_easy --participant_id 001 --total_time 100 --record_video
python user_study.py --agent_level passive_mentee --layout 1_cramped_room_easy --participant_id 001 --total_time 100 --record_video
python user_study.py --agent_level active_supervisor --layout 1_cramped_room_easy --participant_id 001 --total_time 100 --record_video


# Configuration 2
python user_study.py --agent_level default --layout 2_narrow_corridor_medium --participant_id 001 --total_time 100 --record_video
python user_study.py --agent_level passive_mentee --layout 2_narrow_corridor_medium --participant_id 001 --total_time 100 --record_video
python user_study.py --agent_level active_supervisor --layout 2_narrow_corridor_medium --participant_id 001 --total_time 100 --record_video
 

# Configuration 3
python user_study.py --agent_level default --layout 3_spiral_hard --participant_id 001 --total_time 100 --record_video
python user_study.py --agent_level passive_mentee --layout 3_spiral_hard --participant_id 001 --total_time 100 --record_video
python user_study.py --agent_level active_supervisor --layout 3_spiral_hard --participant_id 001 --total_time 100 --record_video

python main.py --model gpt --prompt_layout_agent2 06042024v1_atomic_action