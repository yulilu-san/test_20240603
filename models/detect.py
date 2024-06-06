import os

import torch

from models.ssd import SSD
from models.ssd_predictions import SSDPredictions

WEIGHTS_DIR = "./weights/"

ssd_cfg = {
    "classes_num": 3,  # 背景クラスを含めた合計クラス数
    "input_size": 300,  # 画像の入力サイズ
    "dbox_num": [4, 6, 6, 6, 4, 4],  # DBoxのアスペクト比の種類
    "feature_maps": [38, 19, 10, 5, 3, 1],  # 各sourceの画像サイズ
    "steps": [8, 16, 32, 64, 100, 300],  # DBOXの大きさを決める
    "min_sizes": [30, 60, 111, 162, 213, 264],  # DBOXの大きさを決める
    "max_sizes": [60, 111, 162, 213, 264, 315],  # DBOXの大きさを決める
    "aspect_ratios": [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
}


def detect_kinotake(
    img, weights_file, confidence_threshold=0.6, disp_score=False, disp_counter=True
):
    """きのこ・たけのこを検出し、バウンディングボックスとラベルを表示する

    Parameters:
        img(ndarray): 入力画像 (BGR)
        weights_file(str): 重みファイル名
        confidence_threshold(float): 確信度の閾値
        disp_score(bool): スコアの表示 (False)
        disp_counter(bool): カウンターの表示 (True)

    Returns:
        img_result(ndarray): 出力画像 (RGB)
    """
    net = SSD(phase="test", cfg=ssd_cfg)
    net_weights = torch.load(
        os.path.join(WEIGHTS_DIR, weights_file), map_location={"cuda:0": "cpu"}
    )
    net.load_state_dict(net_weights)

    kinoko_takenoko_labels = ["kinoko", "takenoko"]
    color_mean = [164, 172, 176]  # 色平均値(BGR)
    label_colors = [(255, 15, 15), (0, 128, 0)]  # ラベルごとの色

    ssd = SSDPredictions(
        eval_categories=kinoko_takenoko_labels,
        net=net,
        input_size=300,
        color_mean=color_mean,
    )

    img_rgb_numpy, predict_bbox, predict_label_index, scores = ssd.ssd_predict(
        img, confidence_threshold=confidence_threshold
    )

    img_result, counts = ssd.draw_labels(
        img_rgb_numpy,
        bbox=predict_bbox,
        label_index=predict_label_index,
        scores=scores,
        label_names=kinoko_takenoko_labels,
        label_colors=label_colors,
        disp_score=disp_score,
        disp_counter=disp_counter,
    )

    return img_result
