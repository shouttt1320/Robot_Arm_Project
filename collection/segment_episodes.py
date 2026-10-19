"""SO-101 픽앤플레이스 데이터셋 서브태스크 자동 분할기 (Phase 5 Extension).

원본 HDF5 데이터는 절대 수정하지 않고, 각 에피소드를 다음 3단계로 분할하여
새로운 폴더(data/so101_subtasks/{reach, grasp, place}/)에 별도 저장합니다:
  1. reach : 홈 포즈 -> 공 상단(Pre-grasp) 도달
  2. grasp : 공 상단 하강 -> 파지(Grasp) -> 살짝 리프트(Lift)
  3. place : 공 파지 상태 -> 목표 바구니 이동 -> 그리퍼 개방(Release) -> 홈 복귀
"""

import argparse
import glob
import os
import sys
import h5py
import numpy as np


def detect_subtask_cutpoints(qpos: np.ndarray) -> tuple:
    """그리퍼 및 관절 궤적을 분석하여 reach/grasp/place의 시작 및 종료 인덱스 산출.
    
    1. Reach: 홈 대기 정지(약 30스텝)를 제거하고 이륙 10스텝 전부터 시작.
              공 상단 도달 & 그리퍼 개방(Pre-grasp) 안착 완료 시점(약 100~130스텝)까지 추출.
    2. Grasp: 공 상단 하강 -> 파지(Grasp) -> 살짝 리프트(Lift).
    3. Place: 공 파지 상태 -> 목표 바구니 이동 -> 그리퍼 개방(Release).
    """
    T = len(qpos)
    j2 = qpos[:, 1]       # 2번 어깨 관절 (홈 자세: 약 -1.75 rad)
    gripper = qpos[:, 5]  # 6번 그리퍼 (닫힘: 약 -1.14 rad, 열림: > -0.6 rad)

    # 1. 홈 포즈 이륙 감지 (Joint 2가 위로 들리기 시작하는 시점)
    takeoff_cands = np.where(j2 > -1.60)[0]
    t_takeoff = takeoff_cands[0] if len(takeoff_cands) > 0 else 35
    t_reach_start = max(0, t_takeoff - 10)  # 무의미한 초기 30스텝 정지 제거 (10스텝 여유)

    # 2. 공 상단 도달 & 그리퍼 최대 개방 감지 (에피소드 전반부 50% 내 탐색)
    search_limit = int(T * 0.50)
    open_cands = np.where(gripper[:search_limit] > -0.65)[0]
    if len(open_cands) > 0:
        t_open = open_cands[0]
    else:
        t_open = int(np.argmax(gripper[:search_limit]))

    # 공 상단에 안정적으로 안착한 시점 (개방 후 15스텝)
    t_reach_end = min(t_open + 15, T)

    # 3. Grasp 구간 (하강 -> 파지 -> 리프트)
    t_grasp_start = max(0, t_open - 5)
    t_grasp_end = min(t_grasp_start + 50, T)

    # 4. Place 구간 (바구니 이동 -> 개방)
    t_place_start = max(0, t_grasp_end - 10)
    t_place_end = T

    reach_slice = (t_reach_start, t_reach_end)
    grasp_slice = (t_grasp_start, t_grasp_end)
    place_slice = (t_place_start, t_place_end)

    return reach_slice, grasp_slice, place_slice


def slice_and_save_hdf5(src_file: str, dst_file: str, start_idx: int, end_idx: int, subtask_name: str):
    """지정된 구간(start_idx ~ end_idx)의 데이터를 슬라이싱하여 새 HDF5로 저장."""
    os.makedirs(os.path.dirname(dst_file), exist_ok=True)

    with h5py.File(src_file, "r") as src, h5py.File(dst_file, "w", rdcc_nbytes=1024**2 * 2) as dst:
        total_steps = end_idx - start_idx
        dst.attrs["sim"] = src.attrs.get("sim", False)
        dst.attrs["total_timesteps"] = total_steps
        dst.attrs["fps"] = src.attrs.get("fps", 30)
        dst.attrs["subtask"] = subtask_name
        dst.attrs["parent_episode"] = os.path.basename(src_file)

        # 1. Action (T_sub, 6)
        actions = src["action"][start_idx:end_idx]
        dst.create_dataset("action", data=actions.astype(np.float32))

        # 2. Observations 그룹
        obs = dst.create_group("observations")
        qpos = src["observations/qpos"][start_idx:end_idx]
        obs.create_dataset("qpos", data=qpos.astype(np.float32))

        if "observations/qvel" in src:
            qvel = src["observations/qvel"][start_idx:end_idx]
            obs.create_dataset("qvel", data=qvel.astype(np.float32))

        # 3. Images 그룹
        img_grp = obs.create_group("images")
        for cam_name in src["observations/images"].keys():
            cam_data = src[f"observations/images/{cam_name}"][start_idx:end_idx]
            img_grp.create_dataset(
                cam_name,
                data=cam_data,
                chunks=(1, 480, 640, 3),
                compression="gzip",
                compression_opts=2,
            )


def main():
    parser = argparse.ArgumentParser(description="SO-101 에피소드 서브태스크 자동 분할 도구")
    parser.add_argument("--src-dir", type=str, default="data/so101_pick_place", help="원본 에피소드 폴더")
    parser.add_argument("--dst-root", type=str, default="data/so101_subtasks", help="분할 데이터 저장 루트")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.src_dir, "episode_*.hdf5")))
    if not files:
        print(f"❌ {args.src_dir} 에서 HDF5 파일을 찾을 수 없습니다.")
        sys.exit(1)

    print("\n" + "=" * 70)
    print(" ✂️  [SO-101 픽앤플레이스 서브태스크 자동 분할 시작]")
    print(f" • 원본 데이터 경로 (보존) : {args.src_dir} (총 {len(files)}개 에피소드)")
    print(f" • 신규 분할 저장 루트     : {args.dst_root}")
    print("   1) reach (이동) ➔ data/so101_subtasks/reach/")
    print("   2) grasp (잡기) ➔ data/so101_subtasks/grasp/")
    print("   3) place (놓기) ➔ data/so101_subtasks/place/")
    print("=" * 70 + "\n")

    subtasks = ["reach", "grasp", "place"]

    for i, file_path in enumerate(files):
        fname = os.path.basename(file_path)
        with h5py.File(file_path, "r") as root:
            qpos = root["observations/qpos"][()]

        reach_s, grasp_s, place_s = detect_subtask_cutpoints(qpos)
        slices = [reach_s, grasp_s, place_s]

        # 각 서브태스크별 저장
        for task_name, (s_idx, e_idx) in zip(subtasks, slices):
            out_file = os.path.join(args.dst_root, task_name, fname)
            slice_and_save_hdf5(file_path, out_file, s_idx, e_idx, task_name)

        sys.stdout.write(
            f"\r  📦 [{i+1:2d}/{len(files)}] {fname} 분할 완료 "
            f"(Reach: {reach_s[0]:3d}~{reach_s[1]:3d} | Grasp: {grasp_s[0]:3d}~{grasp_s[1]:3d} | Place: {place_s[0]:3d}~{place_s[1]:3d})"
        )
        sys.stdout.flush()

    print("\n\n" + "=" * 70)
    print(" 🎉 [서브태스크 분할 완료!] 원본은 손상 없이 보존되었으며, 신규 데이터셋이 구축되었습니다.")
    for task_name in subtasks:
        cnt = len(glob.glob(os.path.join(args.dst_root, task_name, "*.hdf5")))
        print(f"  • {task_name.upper():5s} 태스크: {os.path.join(args.dst_root, task_name)} ({cnt}개 에피소드)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
