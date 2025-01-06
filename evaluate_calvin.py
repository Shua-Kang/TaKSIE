import yaml
from taksie.taksie_wrapper import taksie_wrapper
import argparse
import logging
from pathlib import Path
import sys

from taksie.evaluate_policy import evaluate_policy
from calvin_env.envs.play_table_env import get_env

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on multistep sequences with language goals.")
    parser.add_argument(
        "--running_config",
        type=str
    )
    parser.add_argument("--eval_log_dir", default="log_evaluate_lh_policy",
                        type=str, help="Where to log the evaluation results.")
    parser.add_argument("--device", default=0, type=int, help="CUDA device")

    args = parser.parse_args()

    with open(args.running_config, 'r') as file:
        running_config = yaml.safe_load(file)

    task_d_env_path = "taksie/task_d_env"
    env = get_env(task_d_env_path, show_gui=False)
    model = taksie_wrapper(running_config)
    evaluate_policy(model, env, epoch=0, eval_log_dir=args.eval_log_dir,
                    debug=False, create_plan_tsne=False)


if __name__ == "__main__":
    main()
