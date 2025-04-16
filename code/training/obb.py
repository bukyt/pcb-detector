import os
import json
import math
from glob import glob
from PIL import Image
from tqdm import tqdm

# ========== CONFIG ==========
label_dir = "newdata_868/data/labels/train"
image_dir = "newdata_868/data/images/train"
output_dir = "obb_data"

class_name = "capacitor"
val_size = 50
image_ext = ".PNG"
# =============================

def parse_line(line):
    parts = list(map(float, line.strip().split()))
    kp1_x, kp1_y = parts[5], parts[6]
    kp2_x, kp2_y = parts[8], parts[9]
    return (kp1_x, kp1_y), (kp2_x, kp2_y)

def compute_obb(kp_top, kp_bot):
    cx = (kp_top[0] + kp_bot[0]) / 2
    cy = (kp_top[1] + kp_bot[1]) / 2
    dx = kp_bot[0] - kp_top[0]
    dy = kp_bot[1] - kp_top[1]
    height = math.sqrt(dx**2 + dy**2)
    width = height  # make square

    angle = math.atan2(dy, dx)
    angle_deg = math.degrees(angle)

    # Get OBB vector representation
    t = height / 2
    b = height / 2
    r = width / 2
    l = width / 2
    return [t, r, b, l, width, height], cx, cy

def load_image_size(image_path):
    with Image.open(image_path) as img:
        return img.size  # (width, height)

def create_coco_dict():
    return {
        "images": [],
        "annotations": [],
        "categories": [{"id": 0, "name": class_name}]
    }

def main():
    all_labels = sorted(glob(os.path.join(label_dir, "*.txt")))
    val_labels = all_labels[:val_size]
    train_labels = all_labels[val_size:]

    for subset, label_files in zip(["train", "val"], [train_labels, val_labels]):
        coco = create_coco_dict()
        ann_id = 0
        for img_id, label_file in enumerate(tqdm(label_files)):
            base = os.path.splitext(os.path.basename(label_file))[0]
            image_path = os.path.join(image_dir, base + image_ext)

            if not os.path.exists(image_path):
                continue

            width, height = load_image_size(image_path)
            coco["images"].append({
                "id": img_id,
                "file_name": base + image_ext,
                "width": width,
                "height": height
            })

            with open(label_file) as f:
                for line in f:
                    (kp_top, kp_bot) = parse_line(line)
                    obb, cx, cy = compute_obb(kp_top, kp_bot)

                    coco["annotations"].append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": 0,
                        "bbox": obb,  # Custom OBB format
                        "center": [cx, cy],
                        "keypoints": [kp_top[0], kp_top[1], 2, kp_bot[0], kp_bot[1], 2],
                        "num_keypoints": 2,
                        "iscrowd": 0,
                        "area": obb[4] * obb[5]  # width * height
                    })
                    ann_id += 1

        output_path = os.path.join(output_dir, f"{subset}.json")
        with open(output_path, "w") as f:
            json.dump(coco, f, indent=2)
        print(f"Wrote {subset} annotations to: {output_path}")

if __name__ == "__main__":
    main()
