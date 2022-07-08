from pycocotools.coco import COCO
import os
from tools.annotate_images import annotate
import tools.eval_preds as eval_new
import tools.process_results as process_results

# Possibilities
annotations = False             # use annotation images
save_anns = False               # save every annotated person
evaluate_models = False         # evaluate models
get_results = False              # process results & put them in tables + graphs

path = "".join([os.getcwd()[:-6], '\\coco\\data'])
coco_val2017 = "".join([path, '\\person_keypoints_val2017.json'])
# coco_train2017 = "".join([path, '\\person_keypoints_train2017.json'])
image_path = "".join([path, '\\val2017'])
# image_path = "".join([path, '\\train2017'])
coco_kps = COCO(coco_val2017)

dataset_name_str = "COCO dataset"
categoryIds = coco_kps.getCatIds(catNms=['person']) # this will only filter the person class in COCO

if annotations:
    annotate(coco_kps, categoryIds, annotations, image_path, save_anns)

if evaluate_models:
    eval_new.test_all_models(eval_new.run_tests_models)

if get_results:
    process_results.latex_tables()
    process_results.make_ap_graphs(0)

