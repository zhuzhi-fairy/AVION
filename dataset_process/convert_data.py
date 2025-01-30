import multiprocessing
import os

import ffmpeg


def process_video(args):
    input_path, output_path = args
    if not os.path.isfile(output_path):
        print(input_path)
        min_dimension = 320
        # FFmpegで動画のメタデータを取得
        probe = ffmpeg.probe(input_path)
        video_stream = next(
            (
                stream
                for stream in probe["streams"]
                if stream["codec_type"] == "video"
            ),
            None,
        )
        # 動画の幅と高さを取得
        width = int(video_stream["width"])
        height = int(video_stream["height"])
        # 最短辺を320ピクセルに設定するためのスケーリング係数を計算
        scale_factor = min_dimension / min(width, height)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        if new_width % 2 == 1:
            new_width += 1
        if new_height % 2 == 1:
            new_height += 1
        # FFmpegで動画の回転情報を除去し、リサイズを実行
        (
            ffmpeg.input(input_path)
            .filter("transpose", 0)  # 回転情報を除去（必要であれば調整可能）
            .filter("scale", new_width, new_height)  # サイズ変更
            .output(output_path, codec="libx264", movflags="faststart")
            .run(overwrite_output=True, quiet=True)
        )


def load_hktk555():
    with open("datasets/hktk555_pt/train_cp.txt") as f:
        lines = f.readlines()
    files = [l.split()[0] for l in lines]
    root_path = "/mnt/data/share/sss/annofab-data/"
    args_ls = []
    for n in range(len(files)):
        input_video = root_path + files[n]
        output_video = os.path.join("sss/annofab-data", files[n])
        os.makedirs(os.path.dirname(output_video), exist_ok=True)
        args_ls.append((input_video, output_video))
        # process_video(input_video, output_video)
    return args_ls


def load_k400_val():
    input_folder = "/mnt/data/share/kinetics-dataset/k400/val"
    files = os.listdir(input_folder)
    files = [f for f in files if f.endswith(".mp4")]
    output_folder = "datasets/Kinetics/val_320px"
    args_ls = []
    for file in files:
        args_ls.append(
            (
                os.path.join(input_folder, file),
                os.path.join(output_folder, file),
            )
        )
    return args_ls


def load_ego4d():
    input_folder = "/mnt/data/share/ego4d_data/v2/full_scale"
    files = os.listdir(input_folder)
    files = [f for f in files if f.endswith(".mp4")]
    output_folder = "datasets/Ego4D/video_320px"
    os.makedirs(output_folder, exist_ok=True)
    args_ls = []
    for file in files:
        args_ls.append(
            (
                os.path.join(input_folder, file),
                os.path.join(output_folder, file),
            )
        )
    return args_ls


if __name__ == "__main__":
    # args_ls = load_hktk555()
    # args_ls = load_k400_val()
    args_ls = load_ego4d()
    # with multiprocessing.Pool(32) as p:
    #     p.map(process_video, args_ls)
    for args in args_ls:
        process_video(args)
