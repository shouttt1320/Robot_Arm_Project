#!/usr/bin/env python3
"""SO-101 HDF5 데이터셋 ➔ LeRobot 표준 데이터셋(v3) 변환 스크립트.

기존 data/so101_pick_place 디렉터리의 HDF5 파일들을 읽어,
SmolVLA 학습에 바로 사용할 수 있는 LeRobot 포맷(비디오 mp4 + parquet)으로 자동 변환합니다.
"""

import argparse
import glob
import os
import shutil
import h5py
import numpy as np
from tqdm import tqdm


def convert_hdf5_to_lerobot(
    input_dir: str,
    output_dir: str,
    repo_id: str = "local/so101_pick_place",
    task_description: str = "Pick up the red block and place it into the bin.",
    fps: int = 30,
    overwrite: bool = False,
):
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        raise ImportError(
            "LeRobot이 설치된 가상환경이 필요합니다!\n"
            "먼저 'source ~/lerobot_env/bin/activate'를 실행한 후 다시 시도하세요."
        )

    hdf5_files = sorted(glob.glob(os.path.join(input_dir, "episode_*.hdf5")))
    if not hdf5_files:
        raise FileNotFoundError(f"'{input_dir}' 경로에 episode_*.hdf5 파일이 없습니다.")

    print(f"📦 총 {len(hdf5_files)}개의 HDF5 에피소드를 발견했습니다.")
    print(f"📁 출력 경로: {output_dir}")
    print(f"💬 기본 언어 명령(Task): \"{task_description}\"")

    if os.path.exists(output_dir):
        if overwrite:
            print(f"⚠️ 기존 출력 폴더('{output_dir}')를 삭제하고 새로 생성합니다.")
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(
                f"출력 경로 '{output_dir}'가 이미 존재합니다. 덮어쓰려면 --overwrite 옵션을 사용하세요."
            )

    # LeRobot 피처 스키마 정의 (6-DOF 로봇 팔 + Top/Wrist 카메라)
    features = {
        "action": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"],
        },
        "observation.images.top": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.images.wrist": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channel"],
        },
    }

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=output_dir,
        robot_type="so101_follower",
    )

    print("\n🚀 LeRobot 데이터셋 변환 시작...")
    for ep_idx, file_path in enumerate(tqdm(hdf5_files, desc="Converting Episodes")):
        with h5py.File(file_path, "r") as f:
            actions = f["action"][:]
            qpos = f["observations/qpos"][:]
            top_imgs = f["observations/images/top"][:]
            wrist_imgs = f["observations/images/wrist"][:]

            num_steps = len(actions)
            for t in range(num_steps):
                frame = {
                    "action": actions[t].astype(np.float32),
                    "observation.state": qpos[t].astype(np.float32),
                    "observation.images.top": top_imgs[t],
                    "observation.images.wrist": wrist_imgs[t],
                    "task": task_description,
                }
                dataset.add_frame(frame)

            dataset.save_episode()

    dataset.finalize()
    print(f"\n🎉 변환 완료! 총 {len(hdf5_files)}개 에피소드가 '{output_dir}'에 저장되었습니다.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SO-101 HDF5 to LeRobot Dataset Converter")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="/home/jjj/Documents/Robot_Arm_Project/data/so101_pick_place",
        help="입력 HDF5 디렉터리 경로",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/jjj/Documents/Robot_Arm_Project/data/lerobot_so101_pick_place",
        help="출력 LeRobot 데이터셋 디렉터리 경로",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="local/so101_pick_place",
        help="LeRobot 내부 식별자 repo_id",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="Pick up the red block and place it into the bin.",
        help="VLA 모델이 학습할 자연어 지시문 프롬프트",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="수집 FPS (기본값: 30)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="출력 디렉터리가 이미 존재할 경우 덮어쓰기",
    )
    args = parser.parse_args()

    convert_hdf5_to_lerobot(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        repo_id=args.repo_id,
        task_description=args.task,
        fps=args.fps,
        overwrite=args.overwrite,
    )
