from pycocotools.coco import COCO
import cocoAPI
from annotate_images import annotate
import eval_new
import process_results

# Possibilities
annotations = False          # use annotation images
save_anns = False           # save every annotated person
evaluate_models = False     # evaluate models
get_results = False         # process results & put them in tables + graphs


# COCO change paths to 'data//coco//person_keypoints_val2017.json', 'data//coco//val2017'
coco_val2017 = 'data//coco//person_keypoints_val2017.json'
# coco_train2017 = 'data//coco//person_keypoints_train2017.json'
image_path = 'data//coco//val2017'
# image_path = 'data//coco//train2017'
coco_kps = COCO(coco_val2017)

dataset_name_str = "COCO dataset"
categoryIds = coco_kps.getCatIds(catNms=['person']) # this will only filter the person class in COCO
analyse_coco = cocoAPI.COCO_analysis

if annotations:
    annotate(coco_kps, categoryIds, annotations, image_path, save_anns)

if evaluate_models:
    eval_new.test_all_models(eval_new.run_tests_models)

if get_results:
    process_results.latex_tables()
    process_results.make_ap_graphs(0)

