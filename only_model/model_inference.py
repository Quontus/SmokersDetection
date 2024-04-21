import os

from ultralytics import YOLO
from main import xywh_to_ymin_ymax_xmin_xmax
from skimage.io import imread
import cv2
import xgboost
import tensorflow as tf
import numpy as np
from tqdm import tqdm


def get_conv(type: str):
    IMG_SIZE = (56, 56)
    IMG_SHAPE = IMG_SIZE + (3,)

    if type == 'face':
        base_model = tf.keras.applications.VGG16(input_shape=IMG_SHAPE,
                                                 include_top=False,
                                                 weights='imagenet')
        base_model.trainable = False

        return base_model

    elif type == 'wrist':
        base_model = tf.keras.applications.DenseNet121(include_top=False,
                                                       weights="imagenet",
                                                       input_shape=IMG_SHAPE)
        base_model.trainable = False

        return base_model


def decision_for_img(pred_dict, img, wrist_model, face_model):
    # IMG_SIZE = (56, 56)
    # print('Загрузка моделей классификации...')
    # face_model = xgboost.XGBClassifier()
    # face_model.load_model(r'C:\Users\79777\PycharmProjects\SmokersDetectionHack\face_model_vgg16.json')
    # wrist_model = xgboost.XGBClassifier()
    # wrist_model.load_model(r'C:\Users\79777\PycharmProjects\SmokersDetectionHack\wrist_model_densenet121.json')
    # print('Модели загружены')
    check_list = []
    if pred_dict:
        for obj in pred_dict:
            images = []
            bbox = obj["prediction_bbox"]
            cropped_img = img[bbox[0]:bbox[1], bbox[2]:bbox[3]]
            # print(cropped_img.shape)
            cropped_img = cv2.resize(cropped_img, IMG_SIZE)
            images.append(cropped_img)
            images = np.array(images)
            # cropped_img = cropped_img.reshape(1, 56, 56, 3)
            # cropped_img = np.array(cropped_img)
            # print(images.shape)
            label = obj["label"]
            if label == 'wrist':
                base_model = get_conv(label)
                # print('Загрузили свертку')
                features = base_model.predict(images)
                features = features.reshape(features.shape[0], -1)
                prediction = wrist_model.predict(features)
                # print(f'prediction = {prediction}')
                if prediction == 1:
                    check_list.append(1)
                else:
                    check_list.append(0)

            elif label == 'face':
                base_model = get_conv(label)
                # print('Загрузили свертку')
                features = base_model.predict(images)
                features = features.reshape(features.shape[0], -1)
                prediction = face_model.predict(features)
                if prediction == 1:
                    check_list.append(1)
                else:
                    check_list.append(0)

        if check_list.count(1) > 0:
            verdict = 'Курящие!'
            # print(verdict)
            return 1
        else:
            verdict = 'Не курящие!'
            # print(verdict)
            return 0
    else:
        return 1


def predict_by_image(img_name: str, model):

    results = model.predict(img_name)
    pred_dict = []
    img = imread(img_name)
    # model.predict(img_path, imgsz=512, save=True, save_txt=True, save_conf=True)
    # results = model(img_path)  # Make predictions on the input image
    for r in results:
        boxes = r.boxes
        # print(boxes)
        i = 0
        for box in boxes:
            b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
            c = box.cls
            coord = list(map(int, b.tolist()))
            cls = model.names[int(c)]
            pred_dict.append({})
            pred_dict[i]["prediction_bbox"] = [coord[1], coord[3], coord[0], coord[2]]
            pred_dict[i]["label"] = cls
            i = i + 1

    # print('Получили словарь предиктов')
    return pred_dict, img


if __name__ == '__main__':
    IMG_SIZE = (56, 56)
    print('Загрузка моделей классификации...')
    face_model = xgboost.XGBClassifier()
    face_model.load_model(r'C:\Users\79777\PycharmProjects\SmokersDetectionHack\face_model_vgg16.json')
    wrist_model = xgboost.XGBClassifier()
    wrist_model.load_model(r'C:\Users\gapan\Desktop\Py Projects\GITHUB HACKATHON\SmokersDetection\wrist_model_densenet121.json')

    model = YOLO(r'C:\Users\79777\PycharmProjects\SmokersDetectionHack\detection_model.pt')
    print('Модели загружены')
    total_dict = {}
    folder_with_img = r'C:\Users\gapan\Desktop\Py Projects\GITHUB HACKATHON\SmokersDetection\dataset_upd\images\val'
    for img_name in tqdm(os.listdir(folder_with_img)):
        img_path = os.path.join(folder_with_img, img_name)
        pred_dict = predict_by_image(img_path, model)[0]
        img = predict_by_image(img_path, model)[1]
        print(img_name)
        verdict = decision_for_img(pred_dict, img, wrist_model, face_model)
        total_dict[img_name] = verdict

    print(total_dict)
