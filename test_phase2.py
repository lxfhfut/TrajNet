import os
from evaluate import calculate_metrics, parse_csv

ground_truth = parse_csv("/Users/lxfhfut/Dropbox/Garvan/CBVCC/videos/test_phase1/test_phase1.csv")
predictions = parse_csv("/Users/lxfhfut/Dropbox/Garvan/CBVCC/TrajNet/results/preds.csv")

eval_results = calculate_metrics(ground_truth, predictions)
for k, v in eval_results.items():
    print(f"{k}: {v:.4f}")


for k, v in predictions.items():
    if k not in ground_truth.keys():
        print(f"{k} was not found in ground truth.")
    else:
        if int(ground_truth[k]) != int(float(v) > 0.5):
            print(f"{k} was predicted as {int(float(v) > 0.5)}, while the ground-truth is {int(ground_truth[k])}")