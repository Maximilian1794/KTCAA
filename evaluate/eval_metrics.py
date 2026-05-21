from __future__ import print_function, absolute_import
import numpy as np
"""Cross-Modality ReID"""
import pdb

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def get_files_in_folder(folder_path):
    """返回文件夹中所有文件的列表（包含子文件夹）"""
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
            if os.path.isfile(os.path.join(folder_path, f))]


def visualize_topk(distmat, q_pids, g_pids, q_paths, g_paths, k=5, save_dir='./topk_results'):
    """
    可视化每个查询样本的top-k相似图像
    
    参数:
        distmat: 距离矩阵 [num_q, num_g]
        q_pids: 查询集人员ID [num_q]
        g_pids: 图库集人员ID [num_g]
        q_paths: 查询图像路径列表 [num_q]
        g_paths: 图库图像路径列表 [num_g]
        k: 要显示的top-k数量
        save_dir: 结果保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    num_q, num_g = distmat.shape
    k = min(k, num_g)
    indices = np.argsort(distmat, axis=1)  # 每行按距离从小到大排序
    
    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_path = q_paths[q_idx]
        
        # 获取top-k索引
        topk_idx = indices[q_idx][:k]
        topk_paths = [g_paths[i] for i in topk_idx]
        topk_pids = g_pids[topk_idx]
        
        # 创建可视化图像
        plt.figure(figsize=(15, 3))
        
        # 显示查询图像
        plt.subplot(1, k+1, 1)
        img = Image.open(q_path)
        plt.imshow(img)
        plt.title(f'Query\nPID: {q_pid}')
        plt.axis('off')
        
        # 显示top-k图像
        for i, (path, pid) in enumerate(zip(topk_paths, topk_pids)):
            plt.subplot(1, k+1, i+2)
            img = Image.open(path)
            plt.imshow(img)
            match_status = '✓' if pid == q_pid else '✗'
            plt.title(f'Top-{i+1}\nPID: {pid} {match_status}')
            plt.axis('off')
        
        # 调整布局并保存
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'q{q_idx}_pid{q_pid}_top{k}.jpg')
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f'Saved: {save_path}')

    print(f"All top-{k} results saved to {save_dir}")


def eval(distmat, q_pids, g_pids, max_rank = 20):
    # q_paths = get_files_in_folder('/home/yongjie/workspace/subjectivity-sketch-reid/market1k/photo/query')
    # g_paths = get_files_in_folder('/home/yongjie/workspace/subjectivity-sketch-reid/market1k/photo/train')
    # visualize_topk(distmat, q_pids, g_pids, q_paths, g_paths, k=5)

    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0. # number of valid query
    
    # only two cameras
    q_camids = np.ones(num_q).astype(np.int32)
    g_camids = 2* np.ones(num_g).astype(np.int32)
    
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        raw_cmc = matches[q_idx][keep] # binary vector, posits inowith value 1 are correct matches
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = raw_cmc.cumsum()

        # compute mINP
        # refernece Deep Learning for Person Re-identification: A Survey and Outlook
        pos_idx = np.where(raw_cmc == 1)
        pos_max_idx = np.max(pos_idx)
        inp = cmc[pos_max_idx]/ (pos_max_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)
    return all_cmc, mAP, mINP