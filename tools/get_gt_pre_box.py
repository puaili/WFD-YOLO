import numpy as np
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# 锚点的数量
num_points =5000
# 生成目标箱
# 目标箱面积
target_area = 1/32
# 目标箱的 7 种宽高比
aspect_ratios = [1 / 4, 1 / 3, 1 / 2, 1, 2, 3, 4]

# 圆形区域的半径
radius = 0.5
c_xy = 1
target_boxes = []
for ratio in aspect_ratios:
    # 根据面积和宽高比计算目标箱的高度
    height = np.sqrt(target_area / ratio)
    # 根据高度和宽高比计算目标箱的宽度
    width = ratio * height
    # 目标箱的中心点坐标
    # 确保 target_box 是 xywh 格式，x,y 为中心点坐标
    target_box = [c_xy, c_xy, width, height]
    target_boxes.append(target_box)

# 圆形区域的中心坐标
center_x, center_y = c_xy, c_xy
# 生成在 [0, radius^2) 均匀分布的随机数并开方，乘以半径
r = radius * np.sqrt(np.random.rand(num_points))
# 生成在 [0, 2*pi) 均匀分布的随机角度
theta = 2 * np.pi * np.random.rand(num_points)
# 根据极坐标转换为笛卡尔坐标得到锚点的 x 坐标
xx = center_x + r * np.cos(theta)
# 根据极坐标转换为笛卡尔坐标得到锚点的 y 坐标
yy = center_y + r * np.sin(theta)


# 7 scales of anchor boxes
scales = [1 / 32, 1 / 24, 3 / 64, 1 / 16, 1 / 12, 3 / 32, 1 / 8]
# scales = [s*16 for s in [1 / 32, 1 / 24, 3 / 64, 1 / 16, 1 / 12, 3 / 32, 1 / 8]]
# 7 Aspect Ratios for Anchor Boxes
aspect_ratios_anchors = [1 / 4, 1 / 3, 1 / 2, 1, 2, 3, 4]

data_to_save = []
# Generate prediction boxes
for i in tqdm(range(num_points), desc="Processing points"):
    pred_x = xx[i]
    pred_y = yy[i]
    for target_box in target_boxes:
        gt_x, gt_y, gt_w, gt_h = target_box
        for scale in scales:
            for ratio in aspect_ratios_anchors:
                S = scale * gt_w * gt_h
                pred_w = np.sqrt(S * ratio)
                pred_h = np.sqrt(S / ratio)
                pred = [pred_x, pred_y, pred_w, pred_h]
                data_to_save.append({
                    'gt': target_box,
                    'pred': pred
                })

# saving gt_pred_data.json file
with open('gt_pred_data.json', 'w') as f:
    json.dump(data_to_save, f)

# 计算回归案例数
num_regression_cases = len(scales) * len(aspect_ratios_anchors) * len(target_boxes) * num_points
print(f"总共有 {num_regression_cases} 个回归案例")

# 可选：可视化部分
# 设置图片清晰度
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['figure.dpi'] = 500
plt.rcParams['font.size'] = 16

# 创建画布
fig, ax = plt.subplots(figsize=(6, 6))

# 绘制锚点
ax.scatter(xx, yy, s=1, color='#1781B5', label='Points')

# 绘制目标箱
for box in target_boxes:
    x, y, w, h = box
    rect = Rectangle((x - w / 2, y - h / 2), w, h, facecolor='#652C97', alpha=0.3)
    ax.add_patch(rect)

# 可以选择绘制部分锚箱用于可视化展示
# 这里只绘制前 100 个锚箱作为示例
# for i, data in enumerate(data_to_save[:49]):
#     x, y, w, h = data['pred']
#     rect = Rectangle((x - w / 2, y - h / 2), w, h, facecolor='r', alpha=0.3)
#     ax.add_patch(rect)

# 设置坐标轴范围
ax.set_xlim(0.5, 1.5)
ax.set_ylim(0.5, 1.5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.savefig('anchor_p.jpg')
# 显示图形
plt.show()
