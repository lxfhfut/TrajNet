import os
import csv
import torch
import numpy as np
import pandas as pd
from model import VideoClassifier
from dataloader import TrajPointDataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def parse_csv(file):
    data = {}
    with open(file, 'r') as handle:
        # Read the first few lines to check for headers
        sample = handle.read(2048)
        handle.seek(0)  # Reset file pointer to beginning

        # Use csv.Sniffer to detect if there's a header
        sniffer = csv.Sniffer()
        has_header = sniffer.has_header(sample)

        csv_reader = csv.reader(handle)

        # Skip header if it exists
        if has_header:
            next(csv_reader)

        for row in csv_reader:
            # Skip empty rows
            if not row:
                continue

            # Ensure row has at least 2 columns
            if len(row) < 2:
                print(f"Warning: Skipping invalid row: {row}")
                continue

            try:
                # Try to convert second column to float to ensure it's a valid number
                if not row[0].endswith("avi"):
                    data[row[0]+".avi"] = float(row[1])
                else:
                    data[row[0]] = float(row[1])
            except ValueError:
                print(f"Warning: Skipping row with invalid numeric value: {row}")
                continue

    if not data:
        raise ValueError(f"No valid data found in {file}")

    return data


def calculate_metrics(ground_truth, predictions):

    # Get unique thresholds from predictions
    thresholds = sorted(set(float(val) for val in predictions.values()))

    # Calculate ROC points for different thresholds
    roc_points = []
    for th in thresholds:
        TP = FP = FN = TN = 0
        for id_, true_class in ground_truth.items():
            if id_ not in predictions:
                continue

            predicted_class = float(predictions[id_])
            true_class = float(true_class)

            # Calculate TP, FP, FN, TN
            if true_class >= 0.5 and predicted_class >= th:
                TP += 1
            elif true_class >= 0.5 and predicted_class < th:
                FN += 1
            elif true_class < 0.5 and predicted_class >= th:
                FP += 1
            elif true_class < 0.5 and predicted_class < th:
                TN += 1

        # Calculate TPR and FPR for this threshold
        tpr = TP / (TP + FN) if (TP + FN) > 0 else 0
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
        roc_points.append({'FPR': fpr, 'TPR': tpr})

    # Calculate AUC using trapezoidal rule
    auc = 0
    for i in range(1, len(roc_points)):
        x1 = roc_points[i - 1]['FPR']
        y1 = roc_points[i - 1]['TPR']
        x2 = roc_points[i]['FPR']
        y2 = roc_points[i]['TPR']
        auc += abs(x2 - x1) * (y1 + y2) / 2

    # Calculate metrics for default threshold (0.5)
    TP = FP = FN = TN = 0
    default_threshold = 0.5

    for id_, true_class in ground_truth.items():
        if id_ not in predictions:
            continue

        predicted_class = float(predictions[id_])
        true_class = float(true_class)

        if true_class >= 0.5 and predicted_class >= default_threshold:
            TP += 1
        elif true_class >= 0.5 and predicted_class < default_threshold:
            FN += 1
        elif true_class < 0.5 and predicted_class >= default_threshold:
            FP += 1
        elif true_class < 0.5 and predicted_class < default_threshold:
            TN += 1

    # Calculate final metrics
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    balanced_accuracy = (sensitivity + specificity) / 2
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    return {
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'TN': TN,
        'Sensitivity': round(sensitivity, 4),
        'Specificity': round(specificity, 4),
        'Accuracy': round(accuracy, 4),
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'Balanced Accuracy': round(balanced_accuracy, 4),
        'AUC': round(auc, 4),
        'Score': round(0.4*auc+0.2*(precision+recall+balanced_accuracy), 4)
    }


def predict_and_save(model, dataloader, csv_file):
    model.eval()
    all_preds = []
    all_labels = []
    all_vids = []
    all_outs = []

    with torch.no_grad():
        for padded_trajectories, traj_lengths, video_lengths, batch_labels, video_ids in dataloader:
            outputs = model((padded_trajectories, traj_lengths, video_lengths))

            # Convert outputs to predictions
            outputs = torch.atleast_1d(outputs.squeeze())
            preds = (outputs >= 0.5).float()
            batch_preds = preds.cpu().numpy()
            batch_labels = batch_labels.cpu().numpy()
            all_outs.extend(outputs.cpu().numpy())
            all_vids.extend(video_ids)
            all_preds.extend(batch_preds)
            all_labels.extend(batch_labels)
            wrong_idx = np.where(np.abs(batch_preds - batch_labels) > 0.5)[0]
            for idx in wrong_idx:
                print(f"Video ID: {video_ids[idx]}")
                print(f"Predicted: {batch_preds[idx]:.0f} (confidence: {outputs[idx].item():.3f})")
                print(f"Ground Truth: {batch_labels[idx]}")
                print("-" * 30)

    # Add predictions for videos without tracking results, which will be classified as class '0'
    test_df = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(dataloader.dataset.data_dir)), 'test.csv'))
    remaining_vids = set(test_df['id'].to_numpy()) - set(all_vids)
    num_missing = 0
    for r_vid in remaining_vids:
        if r_vid not in all_vids:
            all_vids.extend(r_vid)
            all_outs.extend(0)
            num_missing += 1
    if num_missing:
        print(f"Found {num_missing} videos, setting their class label to default {0}")

    df = pd.DataFrame({"video": [vid+".avi" for vid in all_vids], "predict": all_outs})
    df.to_csv(csv_file, header=False, index=False, float_format="%.3f")
    print(f"Predictions have been save to {csv_file}")

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0)
    }

    return metrics


if __name__ == "__main__":
    pass
    # data_dir = "/Users/lxfhfut/Dropbox/Garvan/CBVCC/tif/cyto_retrained"
    # anno_dir = "/Users/lxfhfut/Dropbox/Garvan/CBVCC/dataset/"
    #
    # tst_dataset = TrajPointDataset(data_dir, split="testing")
    # tst_dataloader = DataLoader(tst_dataset, batch_size=4, shuffle=False, collate_fn=tst_dataset.collate_fn)
    #
    # best_model = VideoClassifier(n_features=64, max_sequence_length=20, sample_attention="single")
    # best_model.load_state_dict(torch.load("ckpts/best_model_20241129_110427tick.pt", weights_only=True)["model_state_dict"])
    #
    # predict_and_save(best_model, tst_dataloader, "results/preds.csv")
    #
    # missing_pred = 0
    # ground_truth_file = os.path.join(anno_dir, "test.csv")
    # predictions_file = 'results/preds.csv'
    #
    # ground_truth = parse_csv(ground_truth_file)
    # predictions = parse_csv(predictions_file)
    #
    # eval_results = calculate_metrics(ground_truth, predictions)
    # for k, v in eval_results.items():
    #     print(f"{k}: {v:.4f}")
