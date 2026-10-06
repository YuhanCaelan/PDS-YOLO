from ultralytics import YOLO

# ============================================================
# Fixed evaluation configuration
# ============================================================

MODEL_PATH = "weights/best.pt"
DATA_YAML = "data/roboflow.yaml"

DEVICE = "0"
IMGSZ = 640
BATCH = 16

# Fixed validation settings
CONF = 0.001
IOU = 0.7

# Output directory
PROJECT = "evaluation"
NAME = "fixed_eval"


def main():
    # Load trained model
    model = YOLO(MODEL_PATH)

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------
    metrics = model.val(
        data=DATA_YAML,
        # Fixed evaluation parameters
        conf=CONF,
        iou=IOU,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        # Do not use test-time augmentation
        augment=False,
        # Save validation predictions
        save_json=True,
        save_txt=True,
        save_conf=True,
        # Output
        project=PROJECT,
        name=NAME,
        exist_ok=True,
        # Deterministic validation configuration
        plots=True,
        verbose=True,
    )

    # --------------------------------------------------------
    # Print main metrics
    # --------------------------------------------------------
    print("\n========== Evaluation Results ==========")

    print(f"Precision     : {metrics.box.mp:.4f}")
    print(f"Recall        : {metrics.box.mr:.4f}")
    print(f"mAP@0.5       : {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95  : {metrics.box.map:.4f}")

    print("========================================")


if __name__ == "__main__":
    main()
