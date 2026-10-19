"""SO-101 학습 및 추론 실험 비교 분석 도구 (A/B Test Comparator).

여러 하이퍼파라미터로 학습한 모델 결과와, 실 로봇에서 추론한 결과를
종합 집계하여 터미널 표로 비교 출력합니다.
"""

import csv
import glob
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def print_train_experiments():
    csv_path = os.path.join(BASE_DIR, "ckpt", "all_train_experiments.csv")
    print("\n" + "=" * 90)
    print(" 📊 [학습 실험 모델 비교 분석표 (Training Experiments)]")
    print("=" * 90)

    if not os.path.exists(csv_path):
        print("  ℹ️ 아직 기록된 학습 실험 요약(ckpt/all_train_experiments.csv)이 없습니다.")
        # 폴더 직접 탐색 (폴더 내 train_summary.csv 확인)
        exp_dirs = glob.glob(os.path.join(BASE_DIR, "ckpt", "experiments", "*"))
        exp_dirs += glob.glob(os.path.join(BASE_DIR, "ckpt", "so101_subtasks", "*"))
        if exp_dirs:
            print("  📂 발견된 모델 체크포인트 폴더:")
            for d in exp_dirs:
                if os.path.isdir(d):
                    cfg_file = os.path.join(d, "train_config.json")
                    cfg_desc = ""
                    if os.path.exists(cfg_file):
                        with open(cfg_file, "r") as f:
                            cfg = json.load(f)
                            cfg_desc = f"(LR: {cfg.get('lr')}, KL: {cfg.get('kl_weight')}, Epochs: {cfg.get('epochs')})"
                    print(f"   • {os.path.basename(d):25s} ➔ {d} {cfg_desc}")
        print("=" * 90 + "\n")
        return

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    if not reader:
        print("  ℹ️ 학습 실험 기록이 비어있습니다.")
        print("=" * 90 + "\n")
        return

    header = f"{'일시':16s} | {'실험명':20s} | {'에포크':6s} | {'최적 Val Loss':13s} | {'L1 오차':8s} | {'KL 손실':8s} | {'LR':8s} | {'KL Wt':6s}"
    print(header)
    print("-" * 95)
    for r in reader:
        print(
            f"{r.get('timestamp', '')[5:16]:16s} | "
            f"{r.get('exp_name', '')[:20]:20s} | "
            f"{r.get('epochs', ''):6s} | "
            f"{r.get('min_val_loss', ''):13s} | "
            f"{r.get('final_val_l1', ''):8s} | "
            f"{r.get('final_val_kl', ''):8s} | "
            f"{r.get('lr', ''):8s} | "
            f"{r.get('kl_weight', ''):6s}"
        )
    print("=" * 90 + "\n")


def print_deploy_experiments():
    csv_path = os.path.join(BASE_DIR, "logs", "all_deploy_runs.csv")
    print("=" * 90)
    print(" 🤖 [실 로봇 자율 추론 실행 이력 (Deployment Runs)]")
    print("=" * 90)

    if not os.path.exists(csv_path):
        print("  ℹ️ 아직 기록된 추론 실행 이력(logs/all_deploy_runs.csv)이 없습니다.")
        print("=" * 90 + "\n")
        return

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    if not reader:
        print("  ℹ️ 추론 기록이 비어있습니다.")
        print("=" * 90 + "\n")
        return

    header = f"{'실행 일시':16s} | {'사용 체크포인트':25s} | {'스텝수':6s} | {'평균 루프':10s} | {'Agg m':6s} | {'Chunk':6s} | {'타임라인 이미지':25s}"
    print(header)
    print("-" * 110)
    for r in reader:
        ckpt_name = os.path.basename(os.path.normpath(r.get('ckpt_dir', '')))
        vis_png = os.path.basename(r.get('visual_png', ''))
        eff_chunk = r.get('effective_chunk', '50')
        print(
            f"{r.get('timestamp', ''):16s} | "
            f"{ckpt_name[:25]:25s} | "
            f"{r.get('steps', ''):6s} | "
            f"{r.get('mean_loop_ms', '') + 'ms':10s} | "
            f"{r.get('temporal_agg_m', ''):6s} | "
            f"{eff_chunk:6s} | "
            f"{vis_png:25s}"
        )
    print("=" * 90 + "\n")


def main():
    print_train_experiments()
    print_deploy_experiments()


if __name__ == "__main__":
    main()
