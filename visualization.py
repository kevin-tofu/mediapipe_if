


import os
import numpy as np
import cv2
from pycocotools.coco import COCO
import pylab as plt
from skimage.draw import line


def minmax(value, _min, _max):
    value = min(value, _max)
    value = max(value, _min)
    return value


def draw_keypoint2img(img, labels, pairs, color = [255, 0, 0], th=0.5):

    ret = np.copy(img)
    for label in labels:
        keypoints = np.array(label['keypoints'])
        keypoints = keypoints.reshape((keypoints.shape[0]//3, 3))
        scores = np.array(label['keyscore'])
        for pair in pairs:
            score1 = scores[pair[0]]    
            score2 = scores[pair[1]]
            if score1 < th or score2 < th:
                continue
            x1 = int(minmax(keypoints[pair[0]][0], 0, ret.shape[1] - 1))
            y1 = int(minmax(keypoints[pair[0]][1], 0, ret.shape[0] - 1))
            x2 = int(minmax(keypoints[pair[1]][0], 0, ret.shape[1] - 1))
            y2 = int(minmax(keypoints[pair[1]][1], 0, ret.shape[0] - 1))

            # print(y1, x1, y2, x2)
            rr, cc = line(y1, x1, y2, x2)
            _color = np.array(color).astype(np.uint8)
            ret[rr, cc, :] = _color
        
    return ret


def draw_keypoint2img_colors(img, labels, pairsList, colorList, th=0.5):

    ret = np.copy(img)

    for label in labels:
        keypoints = np.array(label['keypoints'])
        # print("keypoints:", keypoints.shape)
        keypoints = keypoints.reshape((keypoints.shape[0] // 3, 3)) # (22, 3) ?
        scores = keypoints[:, 2] 

        # print("keypoints:", keypoints.shape)
        # scores = np.array(label['keyscore'])
        # print(keypoints.shape)
        for pairs, color in zip(pairsList, colorList):
            for pair in pairs:
                # print(pair)
                # score1 = scores[pair[0]]    
                # score2 = scores[pair[1]]
                # if score1 < th or score2 < th:
                #     continue
                
                x1 = int(minmax(keypoints[pair[0]][0], 0, ret.shape[1] - 1))
                y1 = int(minmax(keypoints[pair[0]][1], 0, ret.shape[0] - 1))
                x2 = int(minmax(keypoints[pair[1]][0], 0, ret.shape[1] - 1))
                y2 = int(minmax(keypoints[pair[1]][1], 0, ret.shape[0] - 1))
                rr, cc = line(y1, x1, y2, x2)
                _color = np.array(color).astype(np.uint8)
                ret[rr, cc, :] = _color
        
    return ret


def draw_keypoint2video_colors(path_video, \
                               path_video_dst, \
                               labels, pairsList, colorList, th=0.5):

    from mediapipe_if.parse import set_audio

    cap = cv2.VideoCapture(path_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fname_ext = os.path.splitext(path_video_dst)[-1]
    if fname_ext == ".mp4" or fname_ext == ".MP4":
        fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    else:
        raise ValueError(" mp4 video only ")

    writer = cv2.VideoWriter(path_video_dst, fmt, fps, (width, height))
    img_id = -1
    
    while True:
    # for labels_images in labels["images"]:

        img_id += 1
        success, image = cap.read()
        # _id = labels_images["id"]
        ann_image = [loop for loop in labels["images"] if loop["id"] == img_id]
        # print(ann_image)

        if success:
            # image_height, image_width, _ = image.shape
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.asarray(image)

            if len(ann_image) == 1:
                l_point_ignore = [loop for loop in labels["annotations"] \
                              if loop["image_id"] == ann_image[0]["id"]]
                image_keypoint = draw_keypoint2img_colors(image, 
                                                          l_point_ignore, 
                                                          pairsList, 
                                                          colorList, 
                                                          th=th)
            else:
                image_keypoint = image    
            
            image_keypoint = cv2.cvtColor(image_keypoint, cv2.COLOR_RGB2BGR)
            writer.write(image_keypoint)

        else:
            break

    cap.release()
    writer.release()

    set_audio(path_video, path_video_dst)


