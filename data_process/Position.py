# Processing data for the task Relative Position Prediction
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2 
import matplotlib.pyplot as plt
import h5py
import random
import os
import random
import time

x_diff = 150
z_diff = 0.25

def read_pgm_any(path):
    # img source from Megadepth, which uses .h5 file to record depth information
    with h5py.File(path, 'r') as f:
        img = f['depth'][:]    
    if img is None:
        raise IOError(f"无法读取 PGM: {path}")
    return img.astype(np.float32)

def normalize_depth(depth):
    d = depth.copy()
    mask = d > 0
    if d[mask].max()==d[mask].min():
        d[mask] = 1
        d[~mask] = np.nan
    else:
        d[mask] = (d[mask] - d[mask].min()) / (d[mask].max() - d[mask].min())
        d[~mask] = np.nan
    return d

#mark the image
def mark_with_matplotlib(rgb_path, centers, out_path='result.jpg'):
    jpg = plt.imread(rgb_path)

    plt.figure(figsize=(8, 6))
    plt.imshow(jpg)
    plt.axis('off')

    order = [1, 2]
    random.shuffle(order)

    # 画 1 和 2
    for idx, (y, x) in enumerate(centers):
        plt.text(x, y, str(order[idx]),
                 color='black',
                 fontsize=12,
                 ha='center', va='center',
                 bbox=dict(boxstyle="circle,pad=0.25",
                           facecolor='white',
                           edgecolor='black',
                           lw=1.2))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    #plt.show()
    #print(f'结果已保存至 {out_path}')
    plt.close()
    return order


'''
Find two blocks near in depth
center_margin: Only consider the blocks within image center ±center_margin
min_dist: assure that the two blocks aren't too close on the image
'''
def find_two_blocks_near(depth, block=15, inner_thr=0.15, inter_thr=z_diff,
                    min_dist=x_diff, center_margin=0.45):
    h, w = depth.shape
    min_dist = min(h,w) // 8
    half = block // 2

    # Only consider blocks no too far from image center
    yc, xc = h // 2, w // 2
    y_low = int(yc - center_margin * h)
    y_high = int(yc + center_margin * h)
    x_low = int(xc - center_margin * w)
    x_high = int(xc + center_margin * w)

    # Collect block candidates
    blocks = []
    for y in range(max(half, y_low), min(h - half, y_high)):
        for x in range(max(half, x_low), min(w - half, x_high)):
            roi = depth[y-half:y+half+1, x-half:x+half+1]
            if np.all(np.isnan(roi)):
                continue
            roi_min = np.nanmin(roi)
            roi_max = np.nanmax(roi)
            if roi_max - roi_min <= inner_thr:
                dist_to_center = np.hypot(y - yc, x - xc)
                blocks.append((dist_to_center, y, x, roi_min, roi_max))

    # sort by the distance to the center, prioritize the blocks close to the center
    blocks.sort(key=lambda t: t[0])
    s = time.time()
    # Find two blocks whose depth difference less than 0.25 and x distance larger than 150 px (See Appendix A.6)
    n = len(blocks)
    for i in range(n):
        d1, y1, x1, min1, max1 = blocks[i]
        for j in range(i+1, n):
            d2, y2, x2, min2, max2 = blocks[j]
            if abs(max1 - min2) <= inter_thr and abs(max2 - min1) <= inter_thr and abs(y2-y1)<30 and abs(x1-x2)>=min_dist:
                return [(y1, x1), (y2, x2)]
        if time.time()-s>15:
            break
    return []

'''
Find two blocks far in depth
center_margin: Only consider the blocks within image center ±center_margin
min_dist: assure that the two blocks aren't too close on the image
'''
def find_two_blocks_far(depth, block=15, inner_thr=0.15, inter_thr=z_diff,
                    min_dist=150, center_margin=0.45):
    h, w = depth.shape
    min_dist = min(h,w) // 8
    half = block // 2

    # Only consider blocks no too far from image center
    yc, xc = h // 2, w // 2
    y_low = int(yc - center_margin * h)
    y_high = int(yc + center_margin * h)
    x_low = int(xc - center_margin * w)
    x_high = int(xc + center_margin * w)

    # Collect block candidates
    blocks = []
    for y in range(max(half, y_low), min(h - half, y_high)):
        for x in range(max(half, x_low), min(w - half, x_high)):
            roi = depth[y-half:y+half+1, x-half:x+half+1]
            if np.all(np.isnan(roi)):
                continue
            roi_min = np.nanmin(roi)
            roi_max = np.nanmax(roi)
            if roi_max - roi_min <= inner_thr:
                dist_to_center = np.hypot(y - yc, x - xc)
                blocks.append((dist_to_center, y, x, roi_min, roi_max))

    # sort by the distance to the center, prioritize the blocks close to the center
    blocks.sort(key=lambda t: t[0])

    # Find two blocks whose depth difference larger than 0.25 (See Appendix A.6) and also not too close
    n = len(blocks)
    s = time.time()
    for i in range(n):
        d1, y1, x1, min1, max1 = blocks[i]
        for j in range(i+1, n):
            d2, y2, x2, min2, max2 = blocks[j]
            if max(min1, min2)-min(max1, max2)>=inter_thr:
                if np.hypot(y1 - y2, x1 - x2) >= min_dist:
                    return [(y1, x1), (y2, x2)]
        if time.time()-s>15:
            break
    return []

def one_sample_far(depth_path, rgb_path):
    if not os.path.exists(rgb_path):
        return 0, [], [], []
    depth_raw = read_pgm_any(depth_path)
    depth_norm = normalize_depth(depth_raw)
    centers = find_two_blocks_far(depth_norm)
    if not centers:
        print("Can't find two valid blocks on the image.")
        return 0, [], [], []
    else:
        save_path = f"Position1.jpg"
        od = mark_with_matplotlib(rgb_path, centers, save_path)

        x = centers[0][1]
        y = centers[0][0]
        half = 7 #15//2=7
        roi = depth_raw[y-half:y+half+1, x-half:x+half+1]
        roi_max1 = np.nanmax(roi)

        x = centers[1][1]
        y = centers[1][0]
        half = 7
        roi = depth_raw[y-half:y+half+1, x-half:x+half+1]
        roi_max2 = np.nanmax(roi)
        return 1, od, centers, [roi_max1.item(), roi_max2.item()]

def one_sample_near(depth_path, rgb_path):
    if not os.path.exists(rgb_path):
        return 0, [], []
    depth_raw = read_pgm_any(depth_path)
    depth_norm = normalize_depth(depth_raw)

    centers = find_two_blocks_near(depth_norm)
    if not centers:
        print("Can't find two valid blocks on the image.")
        return 0, [], []
    else:
        save_path = f"Position2.jpg"
        od = mark_with_matplotlib(rgb_path, centers, save_path)
        return 1, od, centers

def get_near_prompt(od, centers):
    choice = random.choices([0, 1, 2, 3], weights=[30, 30, 30, 10], k=1)[0]
    options = ["Left", "Right", "Front", "Back"]
    xuhao = ["A", "B", "C", "D"]
    random.shuffle(options)
    view = [0, 1]
    random.shuffle(view) #od[view[0]]: camera, od[view[1]]: target object
    option_str = ""
    for i in range(4):
        option_str += f" {xuhao[i]}. {options[i]}"
    prompt = ""
    gt = "Left"
    if choice==0: #camera facing left
        prompt = f"<image>I've taken an image and there are two regions marked as 1, and 2 on the image. Assume that there is a camera at position '{str(od[view[0]])}' and it's facing to the left of the image. According to the camera, where is the region marker '{str(od[view[1]])}'?{option_str}. Consider cues such as depth, orientation, and 3D spatial relationship. The final answer should be chosen from 'A', 'B', 'C', and 'D'."
        delta_x_2d = centers[view[1]][1] - centers[view[0]][1] #Relative x-coordinate
        if delta_x_2d>=0:
            gt = "Back"
        else:
            gt = "Front"

    elif choice==1: #camera facing right
        prompt = f"<image>I've taken an image and there are two regions marked as 1, and 2 on the image. Assume that there is a camera at position '{str(od[view[0]])}' and it's facing to the right of the image. According to the camera, where is the region marker '{str(od[view[1]])}'?{option_str}. Consider cues such as depth, orientation, and 3D spatial relationship. The final answer should be chosen from 'A', 'B', 'C', and 'D'."
        delta_x_2d = centers[view[1]][1] - centers[view[0]][1]
        if delta_x_2d>=0:
            gt = "Front"
        else:
            gt = "Back"

    elif choice==2: #camera facing to me
        prompt = f"<image>I've taken an image and there are two regions marked as 1, and 2 on the image. Assume that there is a camera at position '{str(od[view[0]])}' and it's facing to me. According to the camera, where is the region marker '{str(od[view[1]])}'?{option_str}. Consider cues such as depth, orientation, and 3D spatial relationship. The final answer should be chosen from 'A', 'B', 'C', and 'D'."
        delta_x_2d = centers[view[1]][1] - centers[view[0]][1] 
        if delta_x_2d>=0:
            gt = "Left"
        else:
            gt = "Right"

    elif choice==3: #camera facing forward
        prompt = f"<image>I've taken an image and there are two regions marked as 1, and 2 on the image. Assume that there is a camera at position '{str(od[view[0]])}' and it's facing to away from me. According to the camera, where is the region marker '{str(od[view[1]])}'?{option_str}. Consider cues such as depth, orientation, and 3D spatial relationship. The final answer should be chosen from 'A', 'B', 'C', and 'D'."
        delta_x_2d = centers[view[1]][1] - centers[view[0]][1]
        if delta_x_2d>=0:
            gt = "Right"
        else:
            gt = "Left"

    correct_option = xuhao[options.index(gt)]
    return prompt, correct_option

def get_far_prompt_same_x(od, centers, depths):
    choice = random.choices([0, 1, 2, 3], weights=[30, 30, 30, 10], k=1)[0] 
    options = ["Left", "Right", "Front", "Back"]
    xuhao = ["A", "B", "C", "D"]
    random.shuffle(options)
    view = [0, 1]
    random.shuffle(view) #od[view[0]]: camera, od[view[1]]: target object
    option_str = ""
    for i in range(4):
        option_str += f" {xuhao[i]}. {options[i]}"
    prompt = ""
    gt = "Left"
    if choice==0:
        prompt = f"<image>I've taken an image and there are two regions marked as 1, and 2 on the image. Assume that there is a camera at position '{str(od[view[0]])}' and it's facing to the left of the image. According to the camera, where is the region marker '{str(od[view[1]])}'?{option_str}. Consider cues such as depth, orientation, and 3D spatial relationship and ignore the height difference between the two regions. The final answer should be chosen from 'A', 'B', 'C', and 'D'."
        delta_depth_2d = depths[view[1]] - depths[view[0]]
        if delta_depth_2d>=0:
            gt = "Right"
        else:
            gt = "Left"

    elif choice==1:
        prompt = f"<image>I've taken an image and there are two regions marked as 1, and 2 on the image. Assume that there is a camera at position '{str(od[view[0]])}' and it's facing to the right of the image. According to the camera, where is the region marker '{str(od[view[1]])}'?{option_str}. Consider cues such as depth, orientation, and 3D spatial relationship and ignore the height difference between the two regions. The final answer should be chosen from 'A', 'B', 'C', and 'D'."
        delta_depth_2d = depths[view[1]] - depths[view[0]]
        if delta_depth_2d>=0:
            gt = "Left"
        else:
            gt = "Right"

    elif choice==2:
        prompt = f"<image>I've taken an image and there are two regions marked as 1, and 2 on the image. Assume that there is a camera at position '{str(od[view[0]])}' and it's facing to me. According to the camera, where is the region marker '{str(od[view[1]])}'?{option_str}. Consider cues such as depth, orientation, and 3D spatial relationship and ignore the height difference between the two regions. The final answer should be chosen from 'A', 'B', 'C', and 'D'."
        delta_depth_2d = depths[view[1]] - depths[view[0]]
        if delta_depth_2d>=0:
            gt = "Back"
        else:
            gt = "Front"

    elif choice==3:
        prompt = f"<image>I've taken an image and there are two regions marked as 1, and 2 on the image. Assume that there is a camera at position '{str(od[view[0]])}' and it's facing to away from me. According to the camera, where is the region marker '{str(od[view[1]])}'?{option_str}. Consider cues such as depth, orientation, and 3D spatial relationship and ignore the height difference between the two regions. The final answer should be chosen from 'A', 'B', 'C', and 'D'."
        delta_depth_2d = depths[view[1]] - depths[view[0]]
        if delta_depth_2d>=0:
            gt = "Front"
        else:
            gt = "Back"

    correct_option = xuhao[options.index(gt)]
    return prompt, correct_option

def get_far_prompt_diff_x(od, centers, depths):
    choice = random.choices([0, 1, 2, 3], weights=[30, 30, 30, 10], k=1)[0]
    options = ["Left-Front", "Right-Front", "Right-Back", "Left-Back"]
    xuhao = ["A", "B", "C", "D"]
    random.shuffle(options)
    view = [0, 1]
    random.shuffle(view) #od[view[0]]: camera, od[view[1]]: target object
    option_str = ""
    for i in range(4):
        option_str += f" {xuhao[i]}. {options[i]}"
    prompt = ""
    gt = "Left"
    if choice==0:
        prompt = f"<image>I've taken an image and there are two regions marked as 1, and 2 on the image. Assume that there is a camera at position '{str(od[view[0]])}' and it's facing to the left of the image. According to the camera, where is the region marker '{str(od[view[1]])}'?{option_str}. Consider cues such as depth, orientation, and 3D spatial relationship and ignore the height difference between the two regions. The final answer should be chosen from 'A', 'B', 'C', and 'D'."
        delta_x_2d = centers[view[1]][1] - centers[view[0]][1]
        delta_depth_2d = depths[view[1]] - depths[view[0]]
        if delta_x_2d>=0 and delta_depth_2d>=0:
            gt = "Right-Back"
        elif delta_x_2d>=0 and delta_depth_2d<0:
            gt = "Left-Back"
        elif delta_x_2d<0 and delta_depth_2d>=0:
            gt = "Right-Front"
        else:
            gt = "Left-Front"

    elif choice==1:
        prompt = f"<image>I've taken an image and there are two regions marked as 1, and 2 on the image. Assume that there is a camera at position '{str(od[view[0]])}' and it's facing to the right of the image. According to the camera, where is the region marker '{str(od[view[1]])}'?{option_str}. Consider cues such as depth, orientation, and 3D spatial relationship and ignore the height difference between the two regions. The final answer should be chosen from 'A', 'B', 'C', and 'D'."
        delta_x_2d = centers[view[1]][1] - centers[view[0]][1]
        delta_depth_2d = depths[view[1]] - depths[view[0]]
        if delta_x_2d>=0 and delta_depth_2d>=0:
            gt = "Left-Front"
        elif delta_x_2d>=0 and delta_depth_2d<0:
            gt = "Right-Front"
        elif delta_x_2d<0 and delta_depth_2d>=0:
            gt = "Left-Back"
        else:
            gt = "Right-Back"

    elif choice==2:
        prompt = f"<image>I've taken an image and there are two regions marked as 1, and 2 on the image. Assume that there is a camera at position '{str(od[view[0]])}' and it's facing to me. According to the camera, where is the region marker '{str(od[view[1]])}'?{option_str}. Consider cues such as depth, orientation, and 3D spatial relationship and ignore the height difference between the two regions. The final answer should be chosen from 'A', 'B', 'C', and 'D'."
        delta_x_2d = centers[view[1]][1] - centers[view[0]][1]
        delta_depth_2d = depths[view[1]] - depths[view[0]]
        if delta_x_2d>=0 and delta_depth_2d>=0:
            gt = "Left-Back"
        elif delta_x_2d>=0 and delta_depth_2d<0:
            gt = "Left-Front"
        elif delta_x_2d<0 and delta_depth_2d>=0:
            gt = "Right-Back"
        else:
            gt = "Right-Front"

    elif choice==3:
        prompt = f"<image>I've taken an image and there are two regions marked as 1, and 2 on the image. Assume that there is a camera at position '{str(od[view[0]])}' and it's facing to away from me. According to the camera, where is the region marker '{str(od[view[1]])}'?{option_str}. Consider cues such as depth, orientation, and 3D spatial relationship and ignore the height difference between the two regions. The final answer should be chosen from 'A', 'B', 'C', and 'D'."
        delta_x_2d = centers[view[1]][1] - centers[view[0]][1]
        delta_depth_2d = depths[view[1]] - depths[view[0]]
        if delta_x_2d>=0 and delta_depth_2d>=0:
            gt = "Right-Front"
        elif delta_x_2d>=0 and delta_depth_2d<0:
            gt = "Right-Back"
        elif delta_x_2d<0 and delta_depth_2d>=0:
            gt = "Left-Front"
        else:
            gt = "Left-Back"

    correct_option = xuhao[options.index(gt)]
    return prompt, correct_option

def get_far_prompt(od, centers, depths):
    if abs(centers[0][1] - centers[1][1])>x_diff:
        return get_far_prompt_diff_x(od, centers, depths)
    else:
        return get_far_prompt_same_x(od, centers, depths)

# Select two blocks whose depth are far and formulate the problem
# Saved as Position1.jpg
img_path = "source_img/RGB-D_img1.jpg"
depth_path = "source_img/RGB-D_depth1.h5"
f, od, centers, depths = one_sample_far(depth_path, img_path)
if f==1:
    question, gt = get_far_prompt(od, centers, depths)
    print("Question:", question)
    print("Ground-truth:", gt)

# Select two blocks whose depth are close and formulate the problem
# Saved as Position2.jpg
img_path = "source_img/RGB-D_img2.jpg"
depth_path = "source_img/RGB-D_depth2.h5"
f, od, centers = one_sample_near(depth_path, img_path)
if f==1:
    question, gt = get_near_prompt(od, centers)
    print("Question:", question)
    print("Ground-truth:", gt)

