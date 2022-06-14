
import os
import numpy as np
import cv2
import mediapipe as mp
from format_annotation import fmt_coco

def set_audio(path_src, path_dst):
    """
    https://kp-ft.com/684
    https://stackoverflow.com/questions/46864915/python-add-audio-to-video-opencv
    """

    import os, shutil
    import moviepy.editor as mp
    import time

    root_ext_pair = os.path.splitext(path_src)
    path_dst_copy = f"{root_ext_pair[0]}-copy{root_ext_pair[1]}"
    shutil.copyfile(path_dst, path_dst_copy)
    time.sleep(0.5)
    # print("path_dst_copy: ", path_dst_copy)

    # Extract audio from input video.                                                                     
    clip_input = mp.VideoFileClip(path_src)
    # clip_input.audio.write_audiofile(path_audio)
    # Add audio to output video.                                                                          
    clip = mp.VideoFileClip(path_dst_copy)
    clip.audio = clip_input.audio

    time.sleep(0.5)
    clip.write_videofile(path_dst)

    time.sleep(0.5)
    os.remove(path_dst_copy)


def checkerComplexity(_model_complexity):
    if int(_model_complexity) == 1 or int(_model_complexity) == 2:
        _model_complexity = int(_model_complexity)
    else:
        _model_complexity = 1
    return int(_model_complexity)

def checkerThreshold(param):
    if param < 0.0:
        param = 0.
    if param >= 1.0:
        param = 0.99
    return param

def mp2coco_keypoint(landmark, posenames, imheight, imwidth):
    
    ret_keypoints = list()
    ret_score_list = list()
    ret_z_list = list()
    for pname in posenames:
        #print(pname.name)
        k_loop = landmark[pname]
        #print(k_loop)
        z = k_loop.z
        score = k_loop.visibility
        v = 0
        if score > 0.3 and score <= 0.7:
            v = 1
        elif score > 0.7:
            v = 2

        temp = [k_loop.x * imwidth, k_loop.y * imheight, v]
        ret_keypoints.append(temp)
        ret_score_list.append(round(score, 2))
        ret_z_list.append(round(k_loop.z, 4))

    ret_keypoints = np.array(ret_keypoints).astype(np.int32)
    return ret_keypoints.ravel().tolist(), ret_score_list, ret_z_list


def draw_keypoint2image(fpath_src, fpath_ex, mp_pose, mp_drawing, mp_drawing_styles,\
                        _model_complexity=1,\
                        _min_detection_confidence=0.3):
    
    _model_complexity = checkerComplexity(_model_complexity)
    _min_detection_confidence = checkerThreshold(_min_detection_confidence)

    image_cv = cv2.imread(fpath_src)
    # print(_model_complexity, _min_detection_confidence)
    # image_cv = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with mp_pose.Pose(
                static_image_mode=True,
                model_complexity=_model_complexity,
                enable_segmentation=False,
                min_detection_confidence=_min_detection_confidence) as pose:

        image_cv.flags.writeable = False
        #results = pose.process(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
        #results = pose.process(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
        results = pose.process(image_cv)

        # Draw the pose annotation on the image.
        image_cv.flags.writeable = True
        image_keypoint = image_cv.copy()
        image_keypoint.flags.writeable = True
        #image_keypoint = cv2.cvtColor(image_keypoint, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image_keypoint,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        cv2.imwrite(fpath_ex, image_keypoint)
        # Plot pose world landmarks.
        mp_drawing.plot_landmarks(
            results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
             


def draw_keypoint2video(path_video, path_export, mp_pose, mp_drawing, mp_drawing_styles,\
                        _model_complexity=1,\
                        _min_detection_confidence=0.5,\
                        _min_tracking_confidence=0.5):
    
    _model_complexity = checkerComplexity(_model_complexity)
    _min_detection_confidence = checkerThreshold(_min_detection_confidence)
    _min_tracking_confidence = checkerThreshold(_min_tracking_confidence)
    
    cap = cv2.VideoCapture(path_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fname_ext = os.path.splitext(path_export)[-1]
    # print(fname_ext)
    if fname_ext == ".mp4" or fname_ext == ".MP4":
        fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # elif fname_ext == ".avi" or fname_ext == ".AVI":
    #     fmt = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    # elif fname_ext == ".mov" or fname_ext == ".MOV":
    #     fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # elif fname_ext == ".wmv" or fname_ext == ".WMV":
    #     fmt = cv2.VideoWriter_fourcc('H', '2', '6', '3')
    else:
        raise ValueError("file extention error")
    
    writer = cv2.VideoWriter(path_export, fmt, fps, (width, height))

    with mp_pose.Pose(
        model_complexity = _model_complexity,\
        min_detection_confidence=_min_detection_confidence,\
        min_tracking_confidence=_min_tracking_confidence) as pose:
        while True:
            success, image = cap.read()
            if success:
                image.flags.writeable = False
                results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                # Draw the pose annotation on the image.
                image.flags.writeable = True
                image_keypoint = image.copy()
                image_keypoint = cv2.cvtColor(image_keypoint, cv2.COLOR_RGB2BGR)
                mp_drawing.draw_landmarks(
                    image_keypoint,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                image_keypoint = cv2.cvtColor(image_keypoint, cv2.COLOR_BGR2RGB)
                writer.write(image_keypoint)
                
            else:
                break
        cap.release()
        writer.release()

        set_audio(path_video, path_export)


def get_keypoint_from_video(path_video, mp_pose, 
                            _model_complexity=1,\
                            _min_detection_confidence=0.5,\
                            _min_tracking_confidence=0.5):

    _model_complexity = checkerComplexity(_model_complexity)
    _min_detection_confidence = checkerThreshold(_min_detection_confidence)
    _min_tracking_confidence = checkerThreshold(_min_tracking_confidence)

    cap = cv2.VideoCapture(path_video)
    pose_list = list()
    imgsize_list = list()
    with mp_pose.Pose(
        model_complexity=_model_complexity,\
        min_detection_confidence=_min_detection_confidence,\
        min_tracking_confidence=_min_tracking_confidence) as pose:

        num_images = 0
        while True:
            success, image = cap.read()
            if success:
                image_height, image_width, _ = image.shape
                image.flags.writeable = False
                results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                pose_list.append(results.pose_landmarks)
                imgsize_list.append([image_height, image_width])
            else:
                break

        cap.release()
    
    return pose_list, imgsize_list


def get_cocokeypoint_from_video(path_video, mp_pose, \
                                _model_complexity=1,\
                                _min_detection_confidence=0.5,\
                                _min_tracking_confidence=0.5, \
                                fname_org = None):

    if fname_org is None:
        fname_org = os.path.basename(path_video)
    _model_complexity = checkerComplexity(_model_complexity)
    _min_detection_confidence = checkerThreshold(_min_detection_confidence)
    _min_tracking_confidence = checkerThreshold(_min_tracking_confidence)
    cap = cv2.VideoCapture(path_video)
    
    coco_image = list()
    coco_annotation = list()

    # keypoint_list = [keypoint.name for keypoint in mp_pose.PoseLandmark]
    # skeleton = [[loop[0], loop[1]] for loop in list(mp_pose.POSE_CONNECTIONS)]
    #frozenset({(15, 21), (16, 20), (18, 20), (3, 7), (14, 16), (23, 25), (28, 30), (11, 23), (27, 31), (6, 8), (15, 17), (24, 26), (16, 22), (4, 5), (5, 6), (29, 31), (12, 24), (23, 24), (0, 1), (9, 10), (1, 2), (0, 4), (11, 13), (30, 32), (28, 32), (15, 19), (16, 18), (25, 27), (26, 28), (12, 14), (17, 19), (2, 3), (11, 12), (27, 29), (13, 15)})
    # coco_category = fmt_coco.make_coco_category('person', 1, 'person', keypoint=keypoint_list, skeleton=skeleton)
    coco_category = get_coco_categories(mp_pose)

    with mp_pose.Pose(
        model_complexity=_model_complexity,\
        min_detection_confidence=_min_detection_confidence,\
        min_tracking_confidence=_min_tracking_confidence) as pose:

        num_images = 0
        annid = 0
        while True:
            success, image = cap.read()
            #print(num_images)
            if success:
                image_height, image_width, _ = image.shape
                #print(image_height, image_width)
                image.flags.writeable = False
                results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                num_images += 1

                if results.pose_landmarks is not None:
                    keypoint_coco, score, z_list = mp2coco_keypoint(results.pose_landmarks.landmark, mp_pose.PoseLandmark,\
                                                            image_height, image_width)

                    d = dict(id=annid, image_id=num_images, bbox=[],\
                             keypoints=keypoint_coco, category_id=1, iscrowd=0,\
                             z=z_list) # num_keypoints=33, keyscore=score, 
                    coco_image += fmt_coco.make_coco_image(num_images, fname_org, image_height, image_width)
                    coco_annotation.append(d)
                    annid += 1
            
            else:
                break

        cap.release()
    
    return coco_image, coco_annotation, coco_category

def get_coco_categories(mp_pose):
    keypoint_list = [keypoint.name for keypoint in mp_pose.PoseLandmark]
    skeleton = [[loop[0]+1, loop[1]+1] for loop in list(mp_pose.POSE_CONNECTIONS)]
    # skeleton = [[loop[0], loop[1]] for loop in list(mp_pose.POSE_CONNECTIONS)]
    #frozenset({(15, 21), (16, 20), (18, 20), (3, 7), (14, 16), (23, 25), (28, 30), (11, 23), (27, 31), (6, 8), (15, 17), (24, 26), (16, 22), (4, 5), (5, 6), (29, 31), (12, 24), (23, 24), (0, 1), (9, 10), (1, 2), (0, 4), (11, 13), (30, 32), (28, 32), (15, 19), (16, 18), (25, 27), (26, 28), (12, 14), (17, 19), (2, 3), (11, 12), (27, 29), (13, 15)})
    coco_category = fmt_coco.make_coco_category('person', 1, 'person', keypoint=keypoint_list, skeleton=skeleton)
    return coco_category

# def get_cocokeypoint_from_image(image, mp_pose):
def get_cocokeypoint_from_image(path_image, mp_pose, \
                                _model_complexity=1,\
                                _min_detection_confidence=0.3, 
                                fname_org = None):

    if fname_org is None:
        fname_org = os.path.basename(path_image)

    _model_complexity = checkerComplexity(_model_complexity)
    _min_detection_confidence = checkerThreshold(_min_detection_confidence)
    # from skimage import io
    coco_image = list()
    coco_annotation = list()

    # keypoint_list = [keypoint.name for keypoint in mp_pose.PoseLandmark]
    # skeleton = [[loop[0], loop[1]] for loop in list(mp_pose.POSE_CONNECTIONS)]
    #frozenset({(15, 21), (16, 20), (18, 20), (3, 7), (14, 16), (23, 25), (28, 30), (11, 23), (27, 31), (6, 8), (15, 17), (24, 26), (16, 22), (4, 5), (5, 6), (29, 31), (12, 24), (23, 24), (0, 1), (9, 10), (1, 2), (0, 4), (11, 13), (30, 32), (28, 32), (15, 19), (16, 18), (25, 27), (26, 28), (12, 14), (17, 19), (2, 3), (11, 12), (27, 29), (13, 15)})
    # coco_category = fmt_coco.make_coco_category('person', 1, 'person', keypoint=keypoint_list, skeleton=skeleton)
    coco_category = get_coco_categories(mp_pose)

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=_model_complexity,
        enable_segmentation=True,
        min_detection_confidence=_min_detection_confidence) as pose:

        num_images = 0
        annid = 0
        # image = io.imread(path_image)
        image = cv2.imread(path_image)

        image_height, image_width, _ = image.shape
        #print(image_height, image_width)
        image.flags.writeable = False
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if results.pose_landmarks is not None:
            keypoint_coco, score, z_list = mp2coco_keypoint(results.pose_landmarks.landmark, mp_pose.PoseLandmark,\
                                                    image_height, image_width)

            d = dict(id=annid, image_id=num_images, bbox=[],\
                        keypoints=keypoint_coco, category_id=1, iscrowd=0,\
                        z=z_list) # num_keypoints=33, keyscore=score, 
            coco_image += fmt_coco.make_coco_image(num_images, fname_org, image_height, image_width)
            coco_annotation.append(d)
            annid += 1

    return coco_image, coco_annotation, coco_category


def get_coco_image_from_video(path_video):

    cap = cv2.VideoCapture(path_video)

    coco_image = list()
    num_images = 0
    while True:
        success, image = cap.read()
        if success:
            image_height, image_width, _ = image.shape
            coco_image += fmt_coco.make_coco_image(num_images, path_video, image_height, image_width)
            num_images += 1
        else:
            break

    cap.release()
    
    return coco_image