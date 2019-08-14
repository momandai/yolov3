import os
import numpy as np
import shutil
import tqdm

gen_type = os.environ.get("gen_type")

in_file = "../coco/trainvalno5k.txt" if gen_type == "train" else "../coco/5k.txt"


output_label_dir = "../coco/person_labels" if gen_type == "train" else "../coco/person_labels_valid"
output_image_dir = "../coco/person_images" if gen_type == "train" else "../coco/person_images_valid"
output_abstract_file = "./data/coco_person.txt" if gen_type == "train" else "./data/coco_person_valid.txt"
output_abstract_file_np = None
if os.path.exists(output_label_dir):
    shutil.rmtree(output_label_dir)
os.makedirs(output_label_dir)

if os.path.exists(output_image_dir):
    shutil.rmtree(output_image_dir)
os.makedirs(output_image_dir)

img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif']
with open(in_file, 'r') as f:
    img_files = f.read().splitlines()
    img_files = [x for x in img_files if os.path.splitext(x)[-1].lower() in img_formats]

    # label_files = [x.replace('images', 'labels').replace(os.path.splitext(x)[-1], '.txt')
    #                     for x in img_files]

    for img_file in tqdm.tqdm(img_files[:2000]):
        label_file = img_file.replace('images', 'labels').replace(os.path.splitext(img_file)[-1], '.txt')
        if os.path.isfile(label_file):
            with open(label_file, 'r') as f_label_file:
                x = np.array([x.split() for x in f_label_file.read().splitlines()], dtype=np.float32)
                y = None
                has_person = False
                for i in range(x.shape[0]):
                    if x[i, 0] == 0:
                        has_person = True
                        y = np.vstack((y, [x[i, :]])) if y is not None else np.array([x[i, :]])
                if has_person:
                    output_label_file = os.path.join(output_label_dir, os.path.basename(label_file))
                    output_image_file = os.path.join(output_image_dir, os.path.basename(img_file))
                    shutil.copy(img_file, output_image_file)
                    np.savetxt(output_label_file, y, fmt='%.6f')
                    output_abstract_file_np = np.vstack((output_abstract_file_np, [output_image_file])) \
                        if output_abstract_file_np is not None else np.array([output_image_file])

    np.savetxt(output_abstract_file, output_abstract_file_np, fmt='%s')
