import argparse
from time import time
from datetime import timedelta
from tqdm import tqdm

import pandas as pd
import torch
import torch.nn.functional as F

from utils import get_test_config, get_logger, roc_curve

def main(args):
    config = get_test_config(args) # Get config
    logger = get_logger(config) # Get logger 
    start_time = time() # Time
    
    # Logging config
    logger.info(f"Experiment (Testing Phase): {config['name']}")
    logger.info(f"Model : \n{config['model']}") 
    logger.info(f"Devices : {config['device']}")
    logger.info(f"Devices ID : {config['device_ids']}")
    logger.info(f"See detail of experiment at {config['config_file_path']}")

    # Computing distances
    config['model'].eval()
    distances, labels, batch_pairs = [], [], []
    with torch.no_grad():
        for (data, label) in tqdm(config['pair_data_loader'], desc="Computing pair distance"):
            data[0][0] = data[0][0].to(config['device']) # (batch_size, num_patches, patch_size, patch_size)
            data[1][0] = data[1][0].to(config['device']) # (batch_size, num_patches, patch_size, patch_size)
            label = label.to(config['device'])
            
            # Get encoded vectors and normalize
            hi, hj, z_i, z_j = config['model'](data[0][0], data[1][0])
            hi = F.normalize(hi, dim=1)
            hj = F.normalize(hj, dim=1)

            # Distance
            distances.append(config['distance'](hi, hj))
            labels.append(label.int())
            batch_pairs.append((data[0][1], data[1][1]))

    # Save (pairs, distances , labels)
    distances = torch.concat(distances, dim=0).unsqueeze(dim=1)
    labels = torch.concat(labels, dim=0).unsqueeze(dim=1)
    pairs = []
    for batch in batch_pairs:
        pairs += [(batch[0][i], batch[1][i]) for i in range(len(batch[0]))]
    
    # Save distance results
    distance_result = pd.concat([
        pd.DataFrame(torch.concat([distances, labels], dim=1).cpu(), columns=["distance", "label"]),
        pd.DataFrame(pairs, columns=["image1", "image2"])
    ], axis=1)
    distance_result.to_csv(config["save_dir"] / "distance_result.csv")

    # Predicts
    metric_results = []
    labels = torch.Tensor(distance_result.label).to(torch.int32)
    for threshold in tqdm(config['thresholds'], desc="Predicts"):
        predicts = torch.Tensor([1 if row.distance <= threshold else 0 for _, row in distance_result.iterrows()]).to(torch.int32)
        metric_results.append([met(predicts, labels) for i, met in enumerate(config['metrics'])])
    metric_results = pd.DataFrame(metric_results, columns=[met.__name__ for met in config['metrics']])
    metric_results = pd.concat([metric_results, pd.DataFrame(config['thresholds'], columns=["threshold"])], axis=1)
    metric_results.to_csv(config["save_dir"] / "metric_result.csv")
    
    # Visualization
    roc_curve(metric_results, config['save_dir'] / 'roc_curve.png')

    # Best case
    best_f1_score_idx = metric_results.f1_score.idxmax()
    best_f1_score_threshold = config['thresholds'][best_f1_score_idx]
    predicts = torch.Tensor([1 if row.distance <= best_f1_score_threshold else 0 for _, row in distance_result.iterrows()]).to(torch.int32)
    best_result = {
        met.__name__: met(predicts, labels) for met in config['metrics']
    }
    logger.info(best_result)
    best_case_detail = pd.concat([
        distance_result,
        pd.DataFrame(predicts, columns=['predict'])
    ], axis=1)
    best_case_detail.to_csv(config["save_dir"] / "best_case_predicts.csv")

    # Time
    end_time = time()
    logger.info("Experiment time " + str(timedelta(seconds=end_time - start_time)).split(".")[0])


if __name__ == "__main__":
    # Argument Parsing
    parser = argparse.ArgumentParser(description="Near-Duplicate Image Retrieval Test")
    parser.add_argument('-c', '--config', default=None, type=str, help='config file path (default: None)')

    #####################################################
    #####################################################
    ####      TODO: Tensorboard visualization        ####
    #####################################################
    #####################################################
    parser.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
    
    # TODO : TensorboardX Option
    args = parser.parse_args()
    main(args)