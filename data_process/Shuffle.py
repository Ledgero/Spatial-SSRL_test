# Processing data for the task Shuffled Patch Reordering

from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont
import IPython
from IPython.display import Image as display_img
import os

def draw_text_with_outline(draw, position, text, font, fill, outline_fill='black', outline_width=2):
    x, y = position
    for dx in [-outline_width, 0, outline_width]:
        for dy in [-outline_width, 0, outline_width]:
            if dx == 0 and dy == 0:
                continue
            draw.text((x + dx, y + dy), text, font=font, fill=outline_fill)
    draw.text(position, text, font=font, fill=fill)

# num_vertical is the number of slices on the vertical direction (h).
# num_horizontal is the number of slices on the horizontal direction (h).
# mosaic: True if a mask is added to one patch
def split_shuffle_and_renumber(image_path, num_vertical=2, num_horizontal=2, mosaic=False, border_color=(255, 0, 0), border_thickness=2, ):
    image = cv2.imread(image_path)
    h, w, _ = image.shape
    patch_h, patch_w = h // num_vertical, w // num_horizontal

    # Split into patches
    patches = []
    for i in range(num_vertical):
        for j in range(num_horizontal):
            patch = image[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w].copy()
            patches.append(patch)

    # Shuffle patches
    original_indices = list(range(num_vertical * num_horizontal))
    shuffled_indices = original_indices.copy()
    random.shuffle(shuffled_indices)

    shuffled_patches = [patches[i] for i in shuffled_indices]

    if mosaic:
        white_patch_index = random.randint(0, num_horizontal*num_vertical - 1)
        shuffled_patches[white_patch_index] = np.full_like(shuffled_patches[white_patch_index], 255)

    # Add patch id
    try:
        font = ImageFont.truetype("arial.ttf", 48)
    except:
        font = ImageFont.load_default()

    numbered_patches = []
    for new_idx, patch in enumerate(shuffled_patches):
        patch = cv2.rectangle(patch, (0, 0), (patch_w-1, patch_h-1), border_color, thickness=border_thickness)
        pil_patch = Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_patch)
        draw_text_with_outline(draw, (10, 10), str(new_idx), font=font, fill='white')
        patch = cv2.cvtColor(np.array(pil_patch), cv2.COLOR_RGB2BGR)
        numbered_patches.append(patch)

    # Concat into new image
    rows = []
    for i in range(num_vertical):
        row = np.hstack(numbered_patches[i*num_horizontal:(i+1)*num_horizontal])
        rows.append(row)
    final_image = np.vstack(rows)
    return final_image, np.argsort(shuffled_indices)





new_img, ids = split_shuffle_and_renumber('source_imgs/RGB_img.png', num_vertical=2, num_horizontal=2, mosaic=True)
cv2.imwrite(os.path.join('shuffle.jpg'), new_img)
print("Ground-truth:", ids)