import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from models.kinoko_takenoko import DataTransform


FONT = "./fonts/NotoSansJP-Bold.otf"


class SSDPredictions:
    """SSDモデルで物体検出を行うクラス

    Attributes:
        eval_categories(list): クラス名(str)
        net(object): SSDモデル
        transform(object): 前処理クラス
    """

    def __init__(self, eval_categories, net, input_size, color_mean):
        """クラスの初期化

        Parameters:
            eval_categories(list): クラス名のリスト
            net(SSD): SSDモデル
            input_size(int): 画像の入力サイズ(300×300)
            color_mean(tuple): データの色の平均値(BGR)
        """
        # クラス名のリストを取得
        self.eval_categories = eval_categories
        # SSDモデル
        self.net = net
        # 前処理を行うDataTransformオブジェクトを生成
        self.transform = DataTransform(input_size, color_mean)

    def ssd_predict(self, img, confidence_threshold=0.5):
        """SSDで物体検出を行い、確信度が高いBBoxの情報を返す

        Parameters:
            img(ndarray): 入力画像(BGR)
            confidence_threshold(float): 確信度の閾値

        Returns: 1画像中で物体を検出したBBoxの情報
            rgb_img(ndarray): 画像のRGB値
            predict_bbox(list): 物体を検出したBBoxの情報
            pre_dict_label_index(list): 物体を検出したBBoxが予測する正解ラベル
            scores(list): 各BBoxごとの確信度
        """

        # ［高さ］, ［幅］, ［RGB値］の要素数をカウントして画像のサイズとチャネル数を取得
        height, width, _ = img.shape
        # BGRからRGBへ変換
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 画像の前処理
        phase = "val"
        img_transformed, boxes, labels = self.transform(
            img, phase, "", ""  # OpneCV2で読み込んだイメージデータ  # 'val'  # アノテーションは存在しないので''
        )
        # img_transformed(ndarray)の形状は(高さのピクセル数,幅のピクセル数,3)
        # 3はBGRの並びなのでこれをRGBの順に変更
        # (3, 高さのピクセル数, 幅のピクセル数)の形状の3階テンソルにする
        img = torch.from_numpy(img_transformed[:, :, (2, 1, 0)]).permute(2, 0, 1)

        # 学習済みSSDモデルで予測
        self.net.eval()  # ネットワークを推論モードにする
        x = img.unsqueeze(0)  # imgの形状をミニバッチの(1,3,300,300)にする
        # detections: 1枚の画像の各物体に対するBBoxの情報が格納される
        # (1, 21(クラス), 200(Top200のBBox), 5)
        # 最後の次元の5は[BBoxの確信度, xmin, ymin, width, height]
        detections = self.net(x)

        # confidence_threshold:
        predict_bbox = []
        pre_dict_label_index = []
        scores = []
        detections = detections.cpu().detach().numpy()

        # 予測結果から物体を検出したとする確信度の閾値以上のBBoxのインデックスを抽出
        # find_index(tuple): (［0次元のインデックス］,
        #                     ［1次元のインデックス],
        #                     [2次元のインデックス],
        #                     [3次元のインデックス],)
        find_index = np.where(detections[:, 0:, :, 0] >= confidence_threshold)

        # detections: (閾値以上のBBox数, 5)
        detections = detections[find_index]

        # find_index[1]のクラスのインデックスの数(21)回ループする
        for i in range(len(find_index[1])):
            if (find_index[1][i]) > 0:  # クラスのインデックス0以外に対して処理する
                sc = detections[i][0]  # detectionsから確信度を取得
                # BBoxの座標[xmin, ymin, width, height]のそれぞれと
                # 画像の[width, height, width, height]をかけ算する
                bbox = detections[i][1:] * [width, height, width, height]
                # find_indexのクラスの次元の値から-1する(背景0を引いて元の状態に戻す)
                lable_ind = find_index[1][i] - 1

                # BBoxのリストに追加
                predict_bbox.append(bbox)
                # 物体のラベルを追加
                pre_dict_label_index.append(lable_ind)
                # 確信度のリストに追加
                scores.append(sc)

        # 1枚の画像のRGB値、BBox、物体のラベル、確信度を返す
        return rgb_img, predict_bbox, pre_dict_label_index, scores

    def draw_labels(
        self,
        img_rgb,
        bbox,
        label_index,
        scores,
        label_names,
        label_colors,
        disp_score=False,
        disp_counter=True,
    ):
        """物体検出の予測結果を描画する

        Parameters:
            img_rgb (ndarray): 画像のRGB値
            bbox(list): 物体を検出したBBoxのリスト
            label_index(list): 検出した物体ラベルへのインデックス
            scores(list): 物体の確信度
            label_names(list): ラベル名のリスト
            label_colors(list): ラベルごとの色指定
                [tuple(R, G, B)]
            disp_score(bool): スコアを表示するか (False)
            disp_counter(bool): カウンターを表示するか (True)

        Returns:
            img_result (ndarray): 描画結果の画像RGB値
            counts(list): ラベルごとの検出オブジェクト数
        """

        img_size = min(img_rgb.shape[0], img_rgb.shape[1])
        thickness = int(img_size * 0.004)
        font_label = ImageFont.truetype(FONT, size=int(img_size * 0.025))
        font_count = ImageFont.truetype(FONT, size=int(img_size * 0.0375))

        img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(img)

        counts = [0] * len(label_names)

        # 検出したBBoxループ
        for i, bb in enumerate(bbox):
            label_name = label_names[label_index[i]]
            color = label_colors[label_index[i]]

            counts[label_index[i]] += 1

            if disp_score:
                sc = scores[i]
                display_txt = f"{label_name}: {sc:.2f}"
            else:
                display_txt = label_name

            draw.rectangle((bb[0], bb[1], bb[2], bb[3]), outline=color, width=thickness)
            w, h = font_label.getsize(display_txt)
            draw.rectangle((bb[0], bb[1] - h, bb[0] + w, bb[1]), fill=color)
            draw.text((bb[0], bb[1] - h), display_txt, font=font_label, fill="white")

        # counter
        if disp_counter:
            x = y = 10  # 座標
            # ラベルごとにカウント表示
            for i, label_name in enumerate(label_names):
                display_txt = f"{label_name}: {str(counts[i])}"
                draw.text((x, y), display_txt, font=font_count, fill=label_colors[i])
                w, h = font_count.getsize(display_txt)
                x = x + w + 10

        img_result = np.array(img)
        return img_result, counts
