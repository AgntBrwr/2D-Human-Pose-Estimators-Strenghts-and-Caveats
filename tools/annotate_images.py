import numpy as np
import os
import json
import matplotlib.pyplot as plt
import pylab
import logging
from PIL import Image
from matplotlib.collections import PatchCollection
import keyboard
from matplotlib.patches import Rectangle, Polygon

# Create logger
fmt = "%(levelname)s %(asctime)s - %(message)s"
logging.basicConfig(filename="logger", level=logging.DEBUG, format=fmt, filemode='w')
logger = logging.getLogger()


# read json file, return empty list in the case of no json content
def read_json(filename="annotations.json"):
    if os.stat(filename).st_size == 0:
        return []
    else:
        with open(filename, "r") as f:
            json_file = json.load(f)
            return json_file


# create a json file, used for annotation data
def write_json(data, filename="annotations//annotations.json"):
    with open(filename, "w") as f:
        json_data.append(data)
        json.dump({"annotations": json_data}, f, indent=4)  # indent = 4


# try to import previous annotations
json_data = read_json()

# if there are annotations to import, set json data to the annotations in the json object
if json_data:
    json_data = json_data["annotations"]

# Display of structure
"image_id"                      # id of the image
"person_id"                     # id of the person
"wrongly_annotated"             # wrong keypoint annotations: 0 is not wrongly annotated/ 1 is wrongly annotated
"occlusion"                     # 0 is not labelled/ 1 is occluded/ 2 is visible
"type_of_occlusion"             # 0 is not occluded, 1 is self, 2 is person, 3 is environment
"pixels"                        # amount of pixels
"truncation by image border"    # amount of keypoints truncated by image border


# loop through images
def annotate(kps_data, categoryids, annotations, image_path, save_annotations):
    dataset_name_str = "COCO dataset"
    infotextstr = "Added keyboard input: \n" \
                  "i: for this info screen \n" \
                  "c: switch between cropped and full images \n" \
                  "j: switch between showing the lines between keypoints and not \n" \
                  "r: go back to previous keypoint \n" \
                  "k: turn on/off segmentation for current person \n" \
                  "h: turn bbox on/off \n" \
                  "d: turn textbox on/off \n" \
                  "y: normal colours/ or open-pose-esque \n" \
                  "g: can be used to select all people for commands: j and w \n \n" \
                  "Press any key to continue"

    start = True
    no_skeleton = kps_data.loadCats(1)[0]['skeleton']

    last_annotated_bool = False
    last_annotated_id = ""  # id of last annotated image
    all_annotated_people_of_current_image = []  # list of all previously annotated people of the current image in the
    # case of continuing with the annotation process after shutting down the program

    # add previously annotated people of the current image to the list
    if json_data:
        last_annotated_id = list(list(json_data)[-1].values())[0]
        for person in reversed(json_data):
            if last_annotated_id != person["image_id"]:
                break
            all_annotated_people_of_current_image.append(person["person_id"])

    # the dictionary mapping keypoint number to a bodily keypoint (same as COCO)
    keypoint_dict = {1: "nose", 2: "left eye", 3: "right eye", 4: "left ear", 5: "right ear", 6: "left shoulder",
                    7: "right shoulder", 8: "left elbow", 9: "right elbow", 10: "left wrist", 11: "right wrist",
                    12: "left hip", 13: "right hip", 14: "left knee", 15: "right knee", 16: "left ankle",
                    17: "right ankle"}

    # loop over images
    logger.debug(f"Loop over images of {dataset_name_str}")
    for ind, id_pic in enumerate(kps_data.getImgIds(catIds=categoryids)):
        # make sure to continue the annotation process at the right person of the right image (where you left off)
        '''if json_data:
            if id_pic == last_annotated_id:
                last_annotated_bool = True
                if not all_annotated_people_of_current_image:
                    continue
            if not last_annotated_bool:
                continue
        else:
            last_annotated_bool = True'''

        # load the data of the image + image itself by means of the corresponding image id
        imgids = kps_data.getImgIds(imgIds=[id_pic])
        img = kps_data.loadImgs(imgids[np.random.randint(0, len(imgids))])[0]
        logger.debug(f"Load Image {id_pic}")

        # Make sure the .jpg file name corresponds to the actual name in the dataset
        id_pic2 = f"{str(0) * (12 - len(str(id_pic)))}{id_pic}"

        image = f"{image_path}\\{str(id_pic2)}.jpg"
        I = Image.open(image).convert('RGB')
        amount_of_pixels_image = len(list(I.getdata())) # get amount of pixels in the image

        # choose either keypoint annotations on or off, to just show images without any keypoint info or with
        if annotations:
            if ind is 0:
                logger.debug("Annotations is true")

            # get annotations corresponding to the current image
            annids = kps_data.getAnnIds(imgIds=img['id'], catIds=categoryids, iscrowd=0)
            anns = kps_data.loadAnns(annids)
            # kps_data.showAnns(anns)

            pers_counter = 0  # to keep track of amount of people in the image
            bool_all = -1
            colours = []
            bb_colour = "b"

            for pers in range(0,len(anns)):
                c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]  # create random colours to annotate the person
                colours.append(c)

            # use while loop to keep the possibility to go back in the loop
            while pers_counter < len(anns):
                segm_bool = 1
                lines = []
                show_lines_kp = 1
                person = anns[pers_counter]  # get annotations corresponding the current person in current image
                trunc_dd = False
                skeleton = []
                kp = np.array(person['keypoints'])
                x_cor = kp[0::3]
                y_cor = kp[1::3]
                vis = kp[2::3]

                wrongly_annotated = []
                occlusion = []
                type_occlusion = []
                truncation = []

                # check if any labelled coordinates
                if np.all((np.array([x_cor]) == 0)):
                    pers_counter += 1
                    continue

                amount_annotations = 4
                fig, ax = plt.subplots(nrows=1, ncols=1, frameon=False)
                fig = pylab.gcf()
                fig.canvas.manager.set_window_title(str(id_pic))
                plt.axis('off')
                plt.imshow(I)

                # turn this on for normal
                plt.plot(x_cor[vis > 0], y_cor[vis > 0], 'o', markersize=4, markerfacecolor=colours[pers_counter],
                            markeredgecolor='k', markeredgewidth=2)
                plt.plot(x_cor[vis > 1], y_cor[vis > 1], 'o', markersize=4, markerfacecolor=colours[pers_counter],
                         markeredgecolor=colours[pers_counter], markeredgewidth=2)

                for annotation_type in range(0, amount_annotations):
                    # 0 is wrong annotations
                    # 1 is whether occlusion is present
                    # 2 is what type of occlusion for occluded joints
                    # 3 is truncation by the image border

                    ax.set_autoscale_on(False)
                    show_text = 1

                    # crop the image
                    x_pad = (max(x_cor[x_cor > 0]) - min(x_cor[x_cor > 0])) * 0.5
                    y_pad = (min(y_cor[y_cor > 0]) - max(y_cor[y_cor > 0])) * 0.5

                    # minimal padding for cases such as where only one or another small amount of keypoints are present
                    if x_pad < 100:
                        x_pad = 100
                    if y_pad < 100:
                        y_pad = 100
                    crop_coordinates = [cor if cor > 0 else 0 for cor in [min(x_cor[x_cor > 0])-x_pad,
                                                                          max(x_cor[x_cor > 0])+x_pad,
                                                                          max(y_cor[y_cor > 0])+y_pad,
                                                                          min(y_cor[y_cor > 0])-y_pad]]
                    bounding_box_coordinates = [cor if cor > 0 else 0 for cor in [min(x_cor[x_cor > 0]),
                                                                                  max(x_cor[x_cor > 0]),
                                                                          max(y_cor[y_cor > 0]), min(y_cor[y_cor > 0])]]
                    image_quality_bb = [int(max(x_cor[x_cor > 0]) - min(x_cor[x_cor > 0])), int(max(y_cor[y_cor > 0]) -
                                     min(y_cor[y_cor > 0]))]
                    if image_quality_bb[0] * image_quality_bb[1] == 0:
                        bb_x = round(np.sqrt(i["area"]))
                        bb_y = round(np.sqrt(i["area"]))
                        if j["image_quality person"][0] != 0:
                            bb_x = image_quality_bb[0]
                        if j["image_quality person"][1] != 0:
                            bb_y = image_quality_bb[1]
                        image_quality_bb = [bb_x, bb_y]

                    image_quality_full = [int(I.size[0]), int(I.size[1])]

                    original_coordinates = [0, I.size[0], I.size[1], 0]
                    plt.axis(crop_coordinates)

                    keyp_ind = 0
                    go_back = False  # check whether the user wants to go back to previous image
                    cropped = 1  # check whether image is cropped
                    max_ind = 17
                    if annotation_type == 0:
                        max_ind = 1  # since with annotation type 0 (wrong annotations) look at whole instead of per kp

                    if annotation_type == 3:
                        max_ind = 2
                    while keyp_ind < max_ind:
                        keypoint = vis[keyp_ind]
                        if wrongly_annotated:
                            if wrongly_annotated[0] == 1:
                                keyp_ind += 1
                                continue
                        if not keypoint and not go_back and annotation_type != 0 and annotation_type != 3:
                            if annotation_type == 1:
                                occlusion.append(0)
                            elif annotation_type == 2:
                                type_occlusion.append(0)
                            keyp_ind += 1
                            continue
                        if annotation_type == 2 and occlusion[keyp_ind] != 1:
                            keyp_ind += 1
                            type_occlusion.append(0)
                            continue
                        else:
                            go_back = False
                            if annotation_type == 0:
                                rect = Rectangle((bounding_box_coordinates[0], bounding_box_coordinates[3]),
                                                 bounding_box_coordinates[1] - bounding_box_coordinates[0],
                                                 bounding_box_coordinates[2] - bounding_box_coordinates[3],
                                                 edgecolor=bb_colour, facecolor='none', linewidth=3)
                            else:
                                bb_size = 5
                                rect = Rectangle((x_cor[keyp_ind]-(bb_size/2), y_cor[keyp_ind]-bb_size/2), bb_size,
                                                 bb_size, edgecolor=bb_colour, facecolor="none", linewidth=3)

                            draw_rect = ax.add_patch(rect)
                            textstr = "Is this person wrongly annotated? 'v' is yes, wrongly annotated, 'x' is no"
                            if start:
                                textstr = infotextstr
                                fig.canvas.draw()
                                start = False
                            if annotation_type == 1:
                                textstr = f"Is the {keypoint_dict[keyp_ind + 1]} occluded?: 'v' is occluded and 'x' " \
                                                  f"not"
                            elif annotation_type == 2:
                                textstr = f"How is {keypoint_dict[keyp_ind + 1]} occluded?: 'b' is by " \
                                                         f"self, 'n' is by other person, 'm' is by environment"
                            elif annotation_type == 3:
                                if keyp_ind == 0:
                                    textstr = f"Are more than 9 keypoints truncated by the image border? 'v' is yes," \
                                              f"'x' is no"
                                if keyp_ind == 1:
                                    textstr = f"How many keypoints are truncated by the image border?"
                            props = dict(boxstyle='round', facecolor='wheat')

                            if show_text == 1:
                                text = ax.text(0.01, 0.99, textstr, transform=ax.transAxes, fontsize=10,
                                            verticalalignment='top', bbox=props, weight="bold")

                            plt.show(block=False)
                            # plt.clf()
                            fig.canvas.draw()
                            fig.canvas.flush_events()

                            # plt.show(block=False)
                            plt.waitforbuttonpress()

                            draw_rect.remove()
                            if show_text == 1:
                                text.remove()

                            # go back to previous keypoint
                            if keyboard.is_pressed("r"):
                                if keyp_ind == 0:
                                    continue
                                if annotation_type == 1:
                                    while not vis[keyp_ind - 1] and keyp_ind > 1:
                                        keyp_ind -= 1
                                        occlusion.pop()
                                if annotation_type == 2:
                                    occ_keyp = type_occlusion[keyp_ind - 1]
                                    while not occ_keyp and keyp_ind > 1:
                                        keyp_ind -= 1
                                        type_occlusion.pop()
                                        occ_keyp = type_occlusion[keyp_ind - 1]
                                if keyp_ind > 0:
                                    keyp_ind = keyp_ind - 1
                                    if occlusion and annotation_type == 1:
                                        occlusion.pop()
                                    if type_occlusion and annotation_type == 2:
                                        type_occlusion.pop()
                                continue

                            if keyboard.is_pressed("w"):
                                segm = person["segmentation"]
                                color = []
                                polygons = []
                                if segm_bool == 1:
                                    if bool_all == 1:
                                        for pers_ind in range(0, len(anns)):
                                            pers = anns[pers_ind]
                                            segm = pers["segmentation"]
                                            for seg in segm:
                                                poly = np.array(seg).reshape((int(len(seg) / 2), 2))
                                                polygons.append(Polygon(poly))
                                                color.append(colours[pers_ind])
                                    else:
                                        for seg in segm:
                                            poly = np.array(seg).reshape((int(len(seg) / 2), 2))
                                            polygons.append(Polygon(poly))
                                            color.append(colours[pers_counter])
                                    p1 = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.4)
                                    ax.add_collection(p1)
                                    p2 = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=2)
                                    ax.add_collection(p2)
                                    segm_bool *= -1
                                elif segm_bool == -1:
                                    p1.remove()
                                    p2.remove()
                                    segm_bool *= -1
                                continue

                            if keyboard.is_pressed("d"):
                                show_text *= -1
                                continue

                            if keyboard.is_pressed("g"):
                                bool_all *= -1
                                continue

                            if keyboard.is_pressed("h"):
                                if bb_colour == "b":
                                    bb_colour = "none"
                                else:
                                    bb_colour = "b"
                                continue
                            if keyboard.is_pressed("j"):
                                if show_lines_kp == 1:
                                    lines = []
                                    kps_data.loadCats(person['category_id'])[0]['skeleton'] = no_skeleton
                                if show_lines_kp == -1:
                                    kps_data.loadCats(person['category_id'])[0]['skeleton'] = skeleton
                                    for line in lines:
                                        line2 = line.pop(0)
                                        line2.remove()
                                sks = np.array(kps_data.loadCats(person['category_id'])[0]['skeleton']) - 1
                                if sks.any():
                                    sk_ind = 0
                                    bound_colour = colours
                                    for sk in sks:
                                        sk_ind += 1
                                        if sk_ind == 15:
                                            sk_ind = 14
                                        if bool_all == 1:
                                            for pers_ind in range(0, len(anns)):
                                                c_ind = pers_ind
                                                pers = anns[pers_ind]
                                                kp2 = np.array(pers['keypoints'])
                                                x_cor2 = kp2[0::3]
                                                y_cor2 = kp2[1::3]
                                                vis2 = kp2[2::3]
                                                if np.all(vis2[sk] > 0):
                                                    line = plt.plot(x_cor2[sk], y_cor2[sk],
                                                                    color=bound_colour[c_ind], linewidth=2)
                                                    lines.append(line)
                                        else:
                                            if np.all(vis[sk] > 0):
                                                line = plt.plot(x_cor[sk], y_cor[sk],
                                                                color=bound_colour[pers_counter], linewidth=2)
                                                lines.append(line)

                                plt.show(block=False)
                                fig.canvas.draw()
                                show_lines_kp *= -1

                            # quit
                            if keyboard.is_pressed("esc"):
                                break

                            # correct
                            if keyboard.is_pressed("v") and annotation_type != 2:
                                keyp_ind += 1
                                if annotation_type == 0:
                                    wrongly_annotated.append(1)
                                if annotation_type == 1:
                                    occlusion.append(1)
                                if annotation_type == 3:
                                    if keyp_ind == 1:
                                        trunc_dd = True
                                    elif keyp_ind == 2:
                                        keyp_ind -= 1
                                continue

                            # incorrect
                            if keyboard.is_pressed("x") and annotation_type != 2:
                                keyp_ind += 1
                                if annotation_type == 0:
                                    wrongly_annotated.append(0)

                                if annotation_type == 1:
                                    occlusion.append(2)
                                if annotation_type == 3:
                                    if keyp_ind == 1:
                                        trunc_dd = False
                                    elif keyp_ind == 2:
                                        keyp_ind -= 1
                                continue

                            if keyboard.is_pressed("i"):

                                infotext = ax.text(0.01, 0.99, infotextstr, transform=ax.transAxes, fontsize=10,
                                               verticalalignment='top', bbox=props, weight="bold")
                                plt.show(block=False)
                                fig.canvas.draw()
                                fig.canvas.flush_events()
                                infotext.remove()
                                plt.waitforbuttonpress()


                            if keyboard.is_pressed("b") and annotation_type == 2:
                                keyp_ind += 1
                                type_occlusion.append(1)
                                continue

                            if keyboard.is_pressed("n") and annotation_type == 2:
                                keyp_ind += 1
                                type_occlusion.append(2)
                                continue

                            if keyboard.is_pressed("m") and annotation_type == 2:
                                keyp_ind += 1
                                type_occlusion.append(3)
                                continue

                            if keyboard.is_pressed("c"):
                                cropped *= -1
                                if cropped == 1:
                                    plt.axis(crop_coordinates)
                                elif cropped == -1:
                                    plt.axis(original_coordinates)
                                continue

                            if keyboard.is_pressed("0") and annotation_type == 3:
                                if trunc_dd:
                                    truncation.append(10)
                                else:
                                    truncation.append(0)
                                keyp_ind += 1
                                continue

                            if keyboard.is_pressed("1") and annotation_type == 3:
                                if trunc_dd:
                                    truncation.append(11)
                                else:
                                    truncation.append(1)
                                keyp_ind += 1
                                continue
                            if keyboard.is_pressed("2") and annotation_type == 3:
                                if trunc_dd:
                                    truncation.append(12)
                                else:
                                    truncation.append(2)
                                keyp_ind += 1
                                continue
                            if keyboard.is_pressed("3") and annotation_type == 3:
                                if trunc_dd:
                                    truncation.append(13)
                                else:
                                    truncation.append(3)
                                keyp_ind += 1
                                continue
                            if keyboard.is_pressed("4") and annotation_type == 3:
                                if trunc_dd:
                                    truncation.append(14)
                                else:
                                    truncation.append(4)
                                keyp_ind += 1
                                continue
                            if keyboard.is_pressed("5") and annotation_type == 3:
                                if trunc_dd:
                                    truncation.append(15)
                                else:
                                    truncation.append(5)
                                keyp_ind += 1
                                continue
                            if keyboard.is_pressed("6") and annotation_type == 3:
                                if trunc_dd:
                                    truncation.append(16)
                                else:
                                    truncation.append(6)
                                keyp_ind += 1
                                continue
                            if keyboard.is_pressed("7") and not trunc_dd and annotation_type == 3:
                                truncation.append(7)
                                keyp_ind += 1
                                continue
                            if keyboard.is_pressed("8") and not trunc_dd and annotation_type == 3:
                                truncation.append(8)
                                keyp_ind += 1
                                continue
                            if keyboard.is_pressed("9") and not trunc_dd and annotation_type == 3:
                                truncation.append(9)
                                keyp_ind += 1
                                continue
                            else:
                                continue
                        keyp_ind += 1

                if save_annotations:
                    write_json(data={"image_id": id_pic,
                                     "person_id": person["id"],
                                     "occlusion": occlusion,
                                     "type_occlusion": type_occlusion,
                                     "truncation": truncation,
                                     "wrongly_annotated": wrongly_annotated,
                                     "image_res_person": image_quality_pers,
                                     "image_res_full": image_quality_full},
                               filename="annotations//annotations.json")
                pers_counter += 1
                plt.close()

        elif not annotations:
            logger.debug("Show images without annotations")
            fig = pylab.gcf()
            fig.canvas.manager.set_window_title(str(id_pic))
            plt.axis('off')
            plt.imshow(I)
            plt.show()

