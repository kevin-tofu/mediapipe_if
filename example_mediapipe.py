
import numpy as np
import cv2
import mediapipe as mp


def mp2coco_keypoint(landmark, posenames, imheight, imwidth):
    
    ret_keypoints = list()
    ret_score_list = list()
    for pname in posenames:

        k_loop = landmark[pname]
        print(k_loop)
        z = k_loop.z
        score = k_loop.visibility
        v = 0
        if score > 0.3 and score <= 0.7:
            v = 1
        elif score > 0.7:
            v = 2

        temp = [k_loop.x * imwidth, k_loop.y * imheight, v]
        ret_keypoints.append(temp)
        ret_score_list.append(score)
    return np.array(ret_keypoints).astype(np.int32), np.array(ret_score_list)


def mp_images(images, mp_drawing, mp_drawing_styles, mp_pose):
    # For static images:
    IMAGE_FILES = images
    BG_COLOR = (192, 192, 192) # gray
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.1) as pose:
        for idx, file in enumerate(IMAGE_FILES):
            print(file)
            image = cv2.imread(file)
            image_height, image_width, _ = image.shape
            # Convert the BGR image to RGB before processing.
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if not results.pose_landmarks:
                continue

            print(
                f'Nose coordinates: ('
                f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width}, '
                f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height})'
            )
            for _ in mp_pose.PoseLandmark:
                print(_)
            #print(len(results.pose_landmarks.landmark))
            keypoint_coco, score = mp2coco_keypoint(results.pose_landmarks.landmark, mp_pose.PoseLandmark,\
                                                    image_height, image_width)
            print(keypoint_coco)

            annotated_image = image.copy()
            # Draw segmentation on the image.
            # To improve segmentation around boundaries, consider applying a joint
            # bilateral filter to "results.segmentation_mask" with "image".
            condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
            bg_image = np.zeros(image.shape, dtype=np.uint8)
            bg_image[:] = BG_COLOR
            annotated_image = np.where(condition, annotated_image, bg_image)
            #print(results.pose_landmarks)

            # Draw pose landmarks on the image.
            mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            cv2.imwrite('./imgs/export/annotated_image' + str(idx) + '.png', annotated_image)
            # Plot pose world landmarks.
            mp_drawing.plot_landmarks(
                results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)


def mp_webcam(mp_drawing, mp_drawing_styles, mp_pose):
    # For webcam input:
    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
                continue

            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            results = pose.process(image)

            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            cv2.imshow('MediaPipe Pose', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
        cap.release()


def test():

    import os
    import pathlib
    dirname = './imgs/'
    if os.path.exists(dirname) == False:
        os.makedirs(dirname)
    if os.path.exists(dirname+'export/') == False:
        os.makedirs(dirname+'export/')

    images = list()
    p_temp = pathlib.Path(dirname).glob('*.jpg')
    for p in p_temp:
        print(p.name)
        images.append(dirname + p.name)

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    mp_images(images, mp_drawing, mp_drawing_styles, mp_pose)

    #mp_webcam(mp_drawing, mp_drawing_styles, mp_pose)

if __name__ == '__main__':

    test()