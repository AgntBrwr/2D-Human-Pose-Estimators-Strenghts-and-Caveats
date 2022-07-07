import numpy as np
import os
import json
from pycocotools.coco import COCO

coco_val2017 = 'C://Users//hugob//Downloads//Master_thesis//coco_test//annotations//person_keypoints_val2017_test.json'
coco_val20172 = 'C://Users//hugob//Downloads//Master_thesis//coco_test//annotations//person_keypoints_val2017.json'


def read_json(filename="data//annotations.json"):
    if os.stat(filename).st_size == 0:
        return []
    else:
        with open(filename, "r") as f:
            json_file = json.load(f)
            return json_file



counter2 = 0
counter3 = 0


def filter_eval(filter, filter_variable, coco_val, annotations):
    # filter -1: regular settings
    # filter 0: occlusion & filter variable:
        # here 0 is unlabelled
        # 1 is occluded
        # 2 is visible
    # filter 1: occlusion type & filter variable:
        # 0 is unlabelled
        # 1 is self occlusion
        # 2 is other person occlusion
        # 3 is environmental occlusion
    # filter 2: truncation & filter variable
        # 0 is no truncation
        # 1 is 1 <= x < 5
        # 2 is lower body
    # filter 3: image size & filter variable
        # 1 is x < 32^2
        # 2 is 32^2 < x < 96^2
        # 3 is x > 96 ^2
    # filter_variable: which type of the above filters
    counter = 0
    occlusion_bool, type_occlusion_bool, truncation, image_size, grouping, r_s = False, False, False, False, False, False
    grouping_occl, full_size, w_a, w_a2 = False, False, False, False
    all_pers_keyps = []
    coco_kps = COCO(coco_val)
    coco_json = read_json(coco_val)
    tc1, tc2, tc3, tc4 = 0,0,0,0
    if filter == -1:
        r_s = True
    if filter == 0:
        occlusion_bool = True
    elif filter == 1:
        type_occlusion_bool = True
    elif filter == 2:
        truncation = True
    elif filter == 3:
        image_size = True
    elif filter == 4:
        grouping = True
    elif filter == 5:
        grouping_occl = True
    elif filter == 6:
        full_size = True
    elif filter == 7:
        full_size = True
    elif filter == 8:
        w_a = True
    elif filter == 9:
        w_a2 = True
    count_image_sizes = 0
    count_full_sizes = 0
    filtered = [1721312, 230376, 1312007, 1740513, 1746654, 1717452, 2166479, 1319021, 1323404, 1202788, 1210187,
                1227465, 1732135, 2167728, 534445, 1227106, 1319611, 2027292, 2005831, 1735123, 2160524, 1707631,
                1713926, 1254701, 2151116, 220757, 231186, 1299663, 1297197, 1706145, 545183, 1287881, 451357, 2030411,
                1714512, 1258253, 184243, 540436, 206340, 1274126, 2165945, 2011904, 1697531, 1705993, 1721168, 1751967,
                1720513, 1215532, 1732260, 1729952, 1712719, 2203727, 1711585, 1300396, 212268, 208131, 2000181,
                2151518, 533283, 1706978, 2162630, 506952, 208369, 216026, 1329144, 2017429, 1297384, 1741542, 1317699,
                227149, 1277114, 1712645, 1757929, 1212898, 2161403, 447531, 1274631, 1283246, 537310, 501094, 1689689,
                558017, 1681073, 541323, 1711191, 1712045, 1712864, 509543, 1297290, 1694331, 2015248, 1702263, 1676487,
                2003516, 2163607, 437772, 428948, 1698148, 1725446, 2026121, 2004368, 1742984, 513475, 200385, 517787,
                2024163, 217591]

    non_filterable = [1740513, 1319021, 1323404, 1227106, 1707631, 1746654, 1210187, 2167728, 1317699, 1717452, 2005831,
                      2151116, 184243, 1751967,
                      1729952, 506952, 1712645, 537310, 558017, 541323, 200385, 1297384, 1741542, 1698148, 1676487]
    cr_co, cr_co1, cr_co2,cr_co3, cr_co4 = 0, 0, 0, 0, 0
    for ind, id_pic in enumerate(coco_json["annotations"]):
        img_ids = coco_kps.getImgIds(imgIds=[id_pic["image_id"]])
        img = coco_kps.loadImgs(img_ids[np.random.randint(0, len(img_ids))])[0]
        ann_ids = coco_kps.getAnnIds(imgIds=img['id'])  # , catIds=1) #, iscrowd=0)
        anns = coco_kps.loadAnns(ann_ids)
        #f len(anns) == 1:
            #person["keypoints"] = person_keyps
            #person["num_keypoints"] = num_keyp
            # person["iscrowd"] = 1
            #all_pers_keyps.append(person)
            #continue
        for ind_person, person in enumerate(anns):
            counter_bool = True
            if id_pic["id"] == person["id"]:
                person_annotation_true = []
                if person["num_keypoints"] > 0:
                    for i in annotations:
                        if i["person_id"] == id_pic["id"]:
                            person_annotation_true = i
                else:
                    all_pers_keyps.append(person)
                    continue
                kp = np.array(person['keypoints'])
                person_keyps = []
                x_cor = kp[0::3]
                y_cor = kp[1::3]
                vis = person_annotation_true["occlusion"]
                vis_coco = kp[2::3]
                num_keyp = 0
                is_crowd = person["iscrowd"]
                if w_a or w_a2:
                    vis = vis_coco
                    if person_annotation_true["occlusion"]:
                        person["keypoints"] = person_keyps
                        person["num_keypoints"] = num_keyp
                        # person["iscrowd"] = 1
                        all_pers_keyps.append(person)
                        continue
                    elif w_a2 and person['id'] in non_filterable:
                        person["keypoints"] = person_keyps
                        person["num_keypoints"] = num_keyp
                        person["iscrowd"] = 1
                        all_pers_keyps.append(person)
                        continue
                elif r_s:
                    vis = vis_coco
                else:
                    checker = False
                    for j in vis:
                        if j != 0:
                            checker = True
                    if not checker:
                        continue
                    if not vis:
                        continue
                occl_type = person_annotation_true["type_occlusion"]
                one, two, three = True, True, True
                check = 0
                for keyp in range(0, 17):
                    if r_s or w_a or w_a2:
                        person_keyps.append(int(x_cor[keyp]))
                        person_keyps.append(int(y_cor[keyp]))
                        person_keyps.append(int(vis_coco[keyp]))
                        if vis_coco[keyp] != 0:
                            num_keyp += 1
                    if occlusion_bool:
                        if vis[keyp] == filter_variable: #and filter_variable != 0:
                            if counter_bool and vis[keyp] == filter_variable:
                                counter += 1
                                counter_bool = False
                            person_keyps.append(int(x_cor[keyp]))
                            person_keyps.append(int(y_cor[keyp]))
                            person_keyps.append(int(vis[keyp]))
                            num_keyp += 1
                        elif filter_variable == 0 and vis[keyp] != filter_variable:
                            counter += 1
                            person_keyps.append(int(x_cor[keyp]))
                            person_keyps.append(int(y_cor[keyp]))
                            person_keyps.append(int(vis[keyp]))
                            num_keyp += 1
                        else:
                            for i in range(0, 3):
                                person_keyps.append(0)
                    elif type_occlusion_bool:
                        if vis[keyp] == 1:
                            if occl_type[keyp] == filter_variable:
                                if counter_bool and occl_type[keyp] == 3:
                                    counter += 1
                                    counter_bool = False

                                # counter += 1
                                person_keyps.append(int(x_cor[keyp]))
                                person_keyps.append(int(y_cor[keyp]))
                                person_keyps.append(int(vis[keyp]))
                                num_keyp += 1
                            else:
                                for i in range(0, 3):
                                    person_keyps.append(0)
                        else:
                            for i in range(0, 3):
                                person_keyps.append(0)
                    elif truncation:
                        truncation_amount = person_annotation_true["truncation"][0]
                        if filter_variable == 0:
                            if truncation_amount == 0 and vis[keyp] > 0:
                                person_keyps.append(int(x_cor[keyp]))
                                person_keyps.append(int(y_cor[keyp]))
                                person_keyps.append(int(vis[keyp]))
                                num_keyp += 1
                                count_image_sizes += 1
                            else:
                                for i in range(0, 3):
                                    person_keyps.append(0)

                        if filter_variable == 1:
                            if 0 < truncation_amount < 5 and vis[keyp] > 0:
                                person_keyps.append(int(x_cor[keyp]))
                                person_keyps.append(int(y_cor[keyp]))
                                person_keyps.append(int(vis[keyp]))
                                tc1 += 1
                                num_keyp += 1
                                count_image_sizes += 1
                            else:
                                for i in range(0, 3):
                                    person_keyps.append(0)
                        if filter_variable == 2:
                            if 5 <= truncation_amount < 9 and vis[keyp] > 0:
                                person_keyps.append(int(x_cor[keyp]))
                                person_keyps.append(int(y_cor[keyp]))
                                person_keyps.append(int(vis[keyp]))
                                tc2 += 1
                                num_keyp += 1
                                count_image_sizes += 1
                            else:
                                for i in range(0, 3):
                                    person_keyps.append(0)
                        if filter_variable == 3:
                            if 9 <= truncation_amount < 13 and vis[keyp] > 0:
                                person_keyps.append(int(x_cor[keyp]))
                                person_keyps.append(int(y_cor[keyp]))
                                person_keyps.append(int(vis[keyp]))
                                tc3 += 1
                                num_keyp += 1
                                count_image_sizes += 1
                            else:
                                for i in range(0, 3):
                                    person_keyps.append(0)
                        if filter_variable == 4:
                            if 13 <= truncation_amount and vis[keyp] > 0:
                                person_keyps.append(int(x_cor[keyp]))
                                person_keyps.append(int(y_cor[keyp]))
                                person_keyps.append(int(vis[keyp]))
                                tc4 += 1
                                num_keyp += 1
                                count_image_sizes += 1
                            else:
                                for i in range(0, 3):
                                    person_keyps.append(0)
                        if filter_variable == 5:
                            if 0 < truncation_amount and vis[keyp] > 0:
                                person_keyps.append(int(x_cor[keyp]))
                                person_keyps.append(int(y_cor[keyp]))
                                person_keyps.append(int(vis[keyp]))
                                tc4 += 1
                                num_keyp += 1
                                count_image_sizes += 1
                            else:
                                for i in range(0, 3):
                                    person_keyps.append(0)
                    elif image_size:
                        try:
                            person_area = person_annotation_true["image_quality person"][1] * person_annotation_true["image_quality person"][0]
                        except:
                            person_area = person_annotation_true["image_res_person"][1] * person_annotation_true["image_res_person"][0]
                        if vis[keyp] == 0:
                            for i in range(0, 3):
                                person_keyps.append(0)
                            continue
                        if filter_variable == 1:
                            if person_area < 32 ** 2:
                                person_keyps.append(int(x_cor[keyp]))
                                person_keyps.append(int(y_cor[keyp]))
                                person_keyps.append(int(vis[keyp]))
                                num_keyp += 1
                                count_image_sizes += 1
                            else:
                                for i in range(0, 3):
                                    person_keyps.append(0)
                        if filter_variable == 2:
                            if 32 ** 2 < person_area < 96 ** 2:
                                person_keyps.append(int(x_cor[keyp]))
                                person_keyps.append(int(y_cor[keyp]))
                                person_keyps.append(int(vis[keyp]))
                                num_keyp += 1
                                count_image_sizes += 1
                            else:
                                for i in range(0, 3):
                                    person_keyps.append(0)
                        if filter_variable == 3:
                            if person_area > 96 ** 2:
                                person_keyps.append(int(x_cor[keyp]))
                                person_keyps.append(int(y_cor[keyp]))
                                person_keyps.append(int(vis[keyp]))
                                num_keyp += 1
                                count_image_sizes += 1
                            else:
                                for i in range(0, 3):
                                    person_keyps.append(0)
                    elif grouping:
                        if filter_variable == 0:
                            if keyp in range(0,5) and vis[keyp] > 0:
                                person_keyps.append(int(x_cor[keyp]))
                                person_keyps.append(int(y_cor[keyp]))
                                person_keyps.append(int(vis[keyp]))
                                num_keyp += 1
                            else:
                                for i in range(0, 3):
                                    person_keyps.append(0)
                        if filter_variable == 1:
                            if (keyp in range(5, 7) or keyp in range(11, 13)) and vis[keyp] > 0:
                                person_keyps.append(int(x_cor[keyp]))
                                person_keyps.append(int(y_cor[keyp]))
                                person_keyps.append(int(vis[keyp]))
                                num_keyp += 1
                            else:
                                for i in range(0, 3):
                                    person_keyps.append(0)
                        if filter_variable == 2:
                            if keyp in range(7, 11) and vis[keyp] > 0:
                                person_keyps.append(int(x_cor[keyp]))
                                person_keyps.append(int(y_cor[keyp]))
                                person_keyps.append(int(vis[keyp]))
                                num_keyp += 1
                            else:
                                for i in range(0, 3):
                                    person_keyps.append(0)
                        if filter_variable == 3:
                            if keyp in range(13, 17) and vis[keyp] > 0:
                                person_keyps.append(int(x_cor[keyp]))
                                person_keyps.append(int(y_cor[keyp]))
                                person_keyps.append(int(vis[keyp]))
                                num_keyp += 1
                            else:
                                for i in range(0, 3):
                                    person_keyps.append(0)
                    elif grouping_occl:

                        vis_occl = 1
                        if filter_variable < 4:
                            vis_occl = 2
                        if filter_variable == 0 or filter_variable == 4:
                            if (0 <= keyp < 5) and vis[keyp] == vis_occl:
                                print(vis[keyp])
                                person_keyps.append(int(x_cor[keyp]))
                                person_keyps.append(int(y_cor[keyp]))
                                person_keyps.append(int(vis[keyp]))
                                num_keyp += 1
                            else:
                                for i in range(0, 3):
                                    person_keyps.append(0)
                        if filter_variable == 1 or filter_variable == 5:
                            if (5 <= keyp < 7 or 11 <= keyp < 13) and vis[keyp] == vis_occl:
                                person_keyps.append(int(x_cor[keyp]))
                                person_keyps.append(int(y_cor[keyp]))
                                person_keyps.append(int(vis[keyp]))
                                num_keyp += 1
                            else:
                                for i in range(0, 3):
                                    person_keyps.append(0)
                        if filter_variable == 2 or filter_variable == 6:
                            if 7 <= keyp < 11 and vis[keyp] == vis_occl:
                                person_keyps.append(int(x_cor[keyp]))
                                person_keyps.append(int(y_cor[keyp]))
                                person_keyps.append(int(vis[keyp]))
                                num_keyp += 1
                            else:
                                for i in range(0, 3):
                                    person_keyps.append(0)
                        if filter_variable == 3 or filter_variable == 7:
                            if 13 <= keyp and vis[keyp] == vis_occl:
                                person_keyps.append(int(x_cor[keyp]))
                                person_keyps.append(int(y_cor[keyp]))
                                person_keyps.append(int(vis[keyp]))
                                num_keyp += 1
                            else:
                                for i in range(0, 3):
                                    person_keyps.append(0)

                person["keypoints"] = person_keyps
                person["num_keypoints"] = num_keyp
                all_pers_keyps.append(person)

    coco_json["annotations"] = all_pers_keyps
    subset_eval_file = open("coco_val2017.json", "w")
    json.dump(coco_json, subset_eval_file)
    subset_eval_file.close()



