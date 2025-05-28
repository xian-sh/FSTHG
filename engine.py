import os
import logging
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import random
from sklearn.metrics import f1_score
import torch.nn.functional as F

from utils import set_logging_config, dict_to_markdown, set_seed
from datasets import DEAPDataset
from datasets.transforms import (BandDifferentialEntropy, Compose, ToGrid, ToTensor, To2d,
                                 Select, Binary, BaselineRemoval)
from datasets.constants import DEAP_CHANNEL_LOCATION_DICT, DEAP_LOCATION_LIST, STANDARD_1005_CHANNEL_LOCATION_DICT, \
    STANDARD_1020_CHANNEL_LOCATION_DICT
from model_selection import KFoldPerSubjectCrossTrial, KFoldPerSubjectGroupbyTrial, KFoldPerSubject
import time
import torch
import numpy as np
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
import os


# 定义训练函数
def train_one_epoch(model, dataloader, optimizer, device):
    model = model.to(device)
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = F.cross_entropy(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(y).sum().item()
        total += y.size(0)

    accuracy = correct / total
    return total_loss / len(dataloader), accuracy


# 定义验证函数
def validate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    start_time = time.time()  # 记录开始时间
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = F.cross_entropy(outputs, y)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(y).sum().item()
            total += y.size(0)

            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    # 记录推理时间
    end_time = time.time()
    inference_time = end_time - start_time  # 总推理时间
    time_per_sample = inference_time / total  # 每个样本的推理时间

    accuracy = correct / total
    f1 = f1_score(all_labels, all_preds, average='macro')  # 计算 F1 分数
    return total_loss / len(dataloader), accuracy, f1, inference_time, time_per_sample


def train_model_per_subject(model, dataset, optimizer, cv, subject_num, max_epochs, batch_size, device, logger, log_dir,
                            sub_idx=None):
    if sub_idx is None:
        sub_idx = list(range(1, subject_num + 1))

    # 循环计算每个被试的结果
    final_results_dict = {
        'Subject': [],
        'Mean_ACC': [],
        'Variance_ACC': [],
        'Mean_F1': [],
        'Variance_F1': []
    }

    for i in sub_idx:
        scores = []
        f1_scores = []
        logger.info('-' * 50)

        subject_dir = os.path.join(log_dir, f'subject_{i:02d}')
        os.makedirs(subject_dir, exist_ok=True)

        for j, (train_dataset, val_dataset) in enumerate(cv.split(dataset, subject=f"s{i:02d}.dat")):
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            logger.info('-' * 50)

            best_f1 = 0.0
            best_acc = 0.0
            best_model_path = os.path.join(subject_dir, f'best_model_fold_{j:02d}.pth')

            for epoch in range(max_epochs):
                # 记录时间和内存
                epoch_start_time = time.time()
                train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
                val_loss, val_acc, val_f1, inference_time, time_per_sample = validate(model, val_loader, device)
                epoch_end_time = time.time()

                # 计算内存占用
                allocated_memory = torch.cuda.memory_allocated(device) / 1024 ** 2  # MB
                reserved_memory = torch.cuda.memory_reserved(device) / 1024 ** 2  # MB
                epoch_duration = epoch_end_time - epoch_start_time  # 计算 epoch 时长

                logger.info(
                    f'Subject {i:02d}, Fold {j:02d}, Epoch {epoch + 1}/{max_epochs}, '
                    f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                    f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}, '
                    f'Epoch Time: {epoch_duration:.2f}s, Inference Time: {inference_time:.2f}s, '
                    f'Time per Sample: {time_per_sample:.6f}s, '
                    f'Memory (Allocated/Reserved): {allocated_memory:.2f}MB/{reserved_memory:.2f}MB')

                # 保存最佳模型
                if val_f1 > best_f1:  # 根据 F1 分数保存最佳模型
                    best_f1 = val_f1
                    best_acc = val_acc
                    torch.save(model.state_dict(), best_model_path)
                    logger.info(f"Saved best model for Subject {i:02d}, Fold {j:02d} with Val F1: {best_f1:.4f}")

            scores.append(best_acc)
            f1_scores.append(best_f1)
            logger.info(f'Subject {i:02d}, Fold {j:02d}, Best Val Acc: {best_acc:.4f}, Best Val F1: {best_f1:.4f}')

        # 计算当前被试的均值和方差
        mean_score = np.mean(scores)
        variance_score = np.var(scores)
        mean_f1 = np.mean(f1_scores)
        variance_f1 = np.var(f1_scores)

        final_results_dict['Subject'].append(i)
        final_results_dict['Mean_ACC'].append(mean_score)
        final_results_dict['Variance_ACC'].append(variance_score)
        final_results_dict['Mean_F1'].append(mean_f1)
        final_results_dict['Variance_F1'].append(variance_f1)

    return final_results_dict


def save_result(final_results_dict, final_results_csv_path, filtered_results_csv_path, exclude_counts=[3, 5, 7, 10],
                logger=None):
    """
    保存测试结果，并支持移除部分表现最差的被试，重新计算结果
    :param final_results_dict: 所有被试的结果字典
    :param final_results_csv_path: 保存完整结果的路径
    :param filtered_results_csv_path: 保存筛选后结果的路径
    :param exclude_counts: 移除最差被试的数量列表
    :param logger: 日志记录器
    """
    # 保存完整结果
    final_results_df = pd.DataFrame(final_results_dict)
    final_results_df.to_csv(final_results_csv_path, index=False)
    logger.info(f'All subjects mean results saved to {final_results_csv_path}')

    # 计算总体均值和方差
    mean_acc, variance_acc = calculate_statistics(final_results_df, 'Mean_ACC')
    mean_f1, variance_f1 = calculate_statistics(final_results_df, 'Mean_F1')

    logger.info(f"Overall Results:\n"
                f"  Mean_ACC: {mean_acc:.4f}, Variance_ACC: {variance_acc:.4f}\n"
                f"  Mean_F1: {mean_f1:.4f}, Variance_F1: {variance_f1:.4f}")

    # 对最终结果按 Mean_ACC 排序
    sorted_results_df = final_results_df.sort_values(by='Mean_ACC', ascending=False).reset_index(drop=True)

    # 初始化去掉最差被试后的结果字典
    filtered_results_dict = {
        'Excluded_Subjects': [],
        'Remaining_Subjects': [],
        'Mean_ACC': [],
        'Variance_ACC': [],
        'Mean_F1': [],
        'Variance_F1': []
    }

    # 循环计算去掉最差被试后的结果
    if exclude_counts:
        exclude_counts = [count for count in exclude_counts if isinstance(count, int) and count > 0]
        if not exclude_counts:
            logger.warning("No valid exclude counts provided.")
            return

        for exclude_count in exclude_counts:
            if exclude_count >= len(sorted_results_df):
                logger.warning(f"Cannot exclude {exclude_count} subjects; total subjects: {len(sorted_results_df)}")
                continue

            # 保留剩余的被试
            remaining_results_df = sorted_results_df.iloc[:-exclude_count]

            # 计算均值和方差
            mean_acc, variance_acc = calculate_statistics(remaining_results_df, 'Mean_ACC')
            mean_f1, variance_f1 = calculate_statistics(remaining_results_df, 'Mean_F1')

            # 更新结果字典
            filtered_results_dict['Excluded_Subjects'].append(exclude_count)
            filtered_results_dict['Remaining_Subjects'].append(len(remaining_results_df))
            filtered_results_dict['Mean_ACC'].append(mean_acc)
            filtered_results_dict['Variance_ACC'].append(variance_acc)
            filtered_results_dict['Mean_F1'].append(mean_f1)
            filtered_results_dict['Variance_F1'].append(variance_f1)

            logger.info(
                f"Excluded {exclude_count} subjects:\n"
                f"  Remaining subjects: {len(remaining_results_df)}\n"
                f"  Mean_ACC: {mean_acc:.4f}, Variance_ACC: {variance_acc:.4f}\n"
                f"  Mean_F1: {mean_f1:.4f}, Variance_F1: {variance_f1:.4f}"
            )

        pd.DataFrame(filtered_results_dict).to_csv(filtered_results_csv_path, index=False)
        logger.info(f'Filtered results saved to {filtered_results_csv_path}')
    else:
        logger.info(f"No Excluded subjects.")


def calculate_statistics(df, metric_name):
    """
    计算指定列的均值和方差
    :param df: DataFrame
    :param metric_name: 需要计算的列名
    :return: 均值和方差
    """
    if len(df) > 1:
        mean_value = df[metric_name].mean()
        variance_value = df[metric_name].var()
    else:
        mean_value = df[metric_name].mean() if len(df) > 0 else 0.0
        variance_value = 0.0
    return mean_value, variance_value