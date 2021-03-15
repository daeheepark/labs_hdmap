import argparse
import json
import os

from nuscenes import NuScenes
from nuscenes.eval.prediction.config import load_prediction_config
from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from nuscenes.prediction import PredictHelper
from nuscenes.prediction.models.physics import ConstantVelocityHeading, PhysicsOracle

version = 'v1.0-trainval'
data_root = '../nus_dataset/original_small/v1.0-trainval'
split_name = 'val'
config_name = 'predict_2020_icra.json'

nusc = NuScenes(version=version, dataroot=data_root)
helper = PredictHelper(nusc)
dataset = get_prediction_challenge_split(split_name, dataroot=data_root)
config = load_prediction_config(helper, config_name)

oracle = PhysicsOracle(config.seconds, helper)
cv_heading = ConstantVelocityHeading(config.seconds, helper)

cv_preds = []
oracle_preds = []
for token in dataset:
    cv_preds.append(cv_heading(token).serialize())
    oracle_preds.append(oracle(token).serialize())

# json.dump(cv_preds, open(os.path.join(output_dir, "cv_preds.json"), "w"))
# json.dump(oracle_preds, open(os.path.join(output_dir, "oracle_preds.json"), "w"))