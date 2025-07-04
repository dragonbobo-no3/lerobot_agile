export PYTHONPATH=/home/agx/price/lerobot_agile:$PYTHONPATH

python replay_mojoco.py --robot.port=can0 --robot.type=mujoco --dataset.repo_id=/home/agx/jedata/test_0618a --dataset.episode=0

python -m lerobot.scripts.visualize_dataset \
    --repo-id test_0704c \
    --root /home/agx/jedata \
    --episode-index 1
