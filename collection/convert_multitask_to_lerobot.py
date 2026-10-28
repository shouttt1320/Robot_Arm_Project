"""SO-101 그린/핑크 멀티태스크 HDF5 데이터셋 ➔ LeRobot 통합 데이터셋 변환 스크립트.

data/so101_pick_green 과 data/so101_pick_pink 에피소드들을 읽어,
각각의 자연어 태스크 라벨을 부여하고 하나의 LeRobot 표준 데이터셋(v3)으로 결합합니다.
"""

import argparse
import glob
import os
import shutil
import h5py
import numpy as np
from tqdm import tqdm


def convert_multitask_to_lerobot(
    output_dir: str = "/home/jjj/Documents/Robot_Arm_Project/data/lerobot_so101_two_balls",
    repo_id: str = "local/so101_two_balls",
    fps: int = 30,
    overwrite: bool = True,
):
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        raise ImportError(
            "LeRobot 환경 활성화가 필요합니다!\n"
            "'source ~/lerobot_env/bin/activate' 후 다시 실행하세요."
        )

    task_configs = [
        {
            "name": "pick_green",
            "dir": "/home/jjj/Documents/Robot_Arm_Project/data/so101_pick_green",
            "task_str": "Pick up the green ball and place it in the box",
        },
        {
            "name": "pick_pink",
            "dir": "/home/jjj/Documents/Robot_Arm_Project/data/so101_pick_pink",
            "task_str": "Pick up the pink ball and place it in the box",
        },
    ]

    all_episodes = []
    for cfg in task_configs:
        files = sorted(glob.glob(os.path.join(cfg["dir"], "episode_*.hdf5")))
        print(f"📦 [{cfg['name']}] 에피소드 {len(files)}개 발견 ➔ Task: \"{cfg['task_str']}\"")
        for f in files:
            all_episodes.append((f, cfg["task_str"]))

    if not all_episodes:
        raise FileNotFoundError("변환할 HDF5 파일이 없습니다!")

    print(f"\n총 결합 에피소드: {len(all_episodes)}개")
    print(f"출력 경로: {output_dir}")

    if os.path.exists(output_dir):
        if overwrite:
            print(f"⚠️ 기존 출력 폴더('{output_dir}')를 삭제하고 새로 생성합니다.")
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(f"'{output_dir}'가 이미 존재합니다.")

    # 6-DOF 로봇 + Dual Camera 스키마
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

    print("\n🚀 멀티태스크 LeRobot 데이터셋 변환 중...")
    for ep_idx, (file_path, task_desc) in enumerate(tqdm(all_episodes, desc="Converting Episodes")):
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
                    "task": task_desc,
                }
                dataset.add_frame(frame)

            dataset.save_episode()

    dataset.finalize()
    print(f"\n🎉 변환 완료! 총 {len(all_episodes)}개 멀티태스크 에피소드가 '{output_dir}'에 저장되었습니다.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multitask HDF5 to LeRobot Converter")
    parser.add_argument("--output-dir", type=str, default="/home/jjj/Documents/Robot_Arm_Project/data/lerobot_so101_two_balls")
    parser.add_argument("--repo-id", type=str, default="local/so101_two_balls")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    convert_multitask_to_lerobot(output_dir=args.output_dir, repo_id=args.repo_id, fps=args.fps, overwrite=True)
