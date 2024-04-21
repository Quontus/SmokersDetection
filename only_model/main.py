import os
import random
import shutil
from tqdm import tqdm
from ultralytics import YOLO
from skimage.io import imsave, imread
import cv2

def xywh_to_ymin_ymax_xmin_xmax(x, y, w, h, initial_shape: tuple) -> list:
    width = initial_shape[1]*w
    height = initial_shape[0]*h
    xmin = initial_shape[1]*x-width/2
    xmax = xmin+width
    ymin = initial_shape[0]*y-height/2
    ymax = ymin+height
    return list(map(int, [ymin, ymax, xmin, xmax]))


def make_folders(root_folder):
    os.makedirs(root_folder, exist_ok=True)
    os.makedirs(os.path.join(root_folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(root_folder, 'labels'), exist_ok=True)
    os.makedirs(os.path.join(root_folder, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(root_folder, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(root_folder, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(root_folder, 'labels', 'val'), exist_ok=True)


def split_samples_from_dir(folder:str, root_folder):
    # for file in os.listdir(folder):
    list_of_files = list(set([file.split('.')[0] for file in os.listdir(folder) if file != 'classes.txt']))
    print(list_of_files)
    # print(len(list_of_files))
    train = random.sample(list_of_files, k=int(0.8*len(list_of_files)))
    print(len(train))
    for file in tqdm(list_of_files):
        if file in train:
            src = os.path.join(folder, file+'.jpg')
            dst = os.path.join(root_folder, 'images', 'train')
            shutil.copy2(src, dst)
            src = os.path.join(folder, file+'.txt')
            dst = os.path.join(root_folder, 'labels', 'train')
            shutil.copy2(src, dst)
        elif file not in train:
            src = os.path.join(folder, file+'.jpg')
            dst = os.path.join(root_folder, 'images', 'val')
            shutil.copy2(src, dst)
            src = os.path.join(folder, file+'.txt')
            dst = os.path.join(root_folder, 'labels', 'val')
            shutil.copy2(src, dst)


def update_files(folder_from, folder_to, folder_init):
    make_folders(folder_to)
    files_full = list(set([file.split('.')[0] for file in os.listdir(folder_from) if file != 'classes.txt']))
    files_init_train = list(set([file.split('.')[0] for file in os.listdir(folder_init) if file != 'classes.txt']))
    for f in tqdm(files_full):
        if f in files_init_train:
            src = os.path.join(folder_from, f+'.jpg')
            dst = os.path.join(folder_to, 'images', 'train')
            shutil.copy2(src, dst)
            src = os.path.join(folder_from, f+'.txt')
            dst = os.path.join(folder_to, 'labels', 'train')
            shutil.copy2(src, dst)

        elif f not in files_init_train:
            src = os.path.join(folder_from, f+'.jpg')
            dst = os.path.join(folder_to, 'images', 'val')
            shutil.copy2(src, dst)
            src = os.path.join(folder_from, f+'.txt')
            dst = os.path.join(folder_to, 'labels', 'val')
            shutil.copy2(src, dst)


def save_small_pictures_from_annotations(folder, save_folder):
    list_of_files = list(set([file.split('.')[0] for file in os.listdir(folder) if file != 'classes.txt']))
    os.makedirs(save_folder, exist_ok=True)
    os.makedirs(os.path.join(save_folder, 'face'), exist_ok=True)
    os.makedirs(os.path.join(save_folder, 'face_with_cigarette'), exist_ok=True)
    # os.makedirs(os.path.join(save_folder, 'wrist'), exist_ok=True)
    # os.makedirs(os.path.join(save_folder, 'wrist_with_cigarette'), exist_ok=True)
    for file in tqdm(list_of_files):
        img = imread(os.path.join(folder, file+'.jpg'))
        with open(os.path.join(folder, file+'.txt')) as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
            for line in lines:
                cl, x, y, w, h = line.split(' ')
                cl = int(cl)
                x, y, w, h = float(x), float(y), float(w), float(h),
                ymin,ymax,xmin,xmax = xywh_to_ymin_ymax_xmin_xmax(x, y, w, h, img.shape[:2])
                if cl == 3:
                    imsave(os.path.join(save_folder, 'face', file+'.jpg'), img[ymin:ymax, xmin:xmax])
                elif cl == 4:
                    imsave(os.path.join(save_folder, 'face_with_cigarette', file+'.jpg'), img[ymin:ymax, xmin:xmax])
                # if cl == 5:
                #     imsave(os.path.join(save_folder, 'wrist', file+'.jpg'), img[ymin:ymax, xmin:xmax])
                # elif cl == 6:
                #     imsave(os.path.join(save_folder, 'wrist_with_cigarette', file + '.jpg'), img[ymin:ymax, xmin:xmax])


def merge_classes(folder, save_folder):
    os.makedirs(save_folder, exist_ok=True)
    list_of_files = list(set([file.split('.')[0] for file in os.listdir(folder) if file != 'classes.txt']))
    for file in tqdm(list_of_files):
        src = os.path.join(folder, file+'.jpg')
        dst = os.path.join(save_folder, file+'.jpg')
        shutil.copy2(src, dst)
        with open(os.path.join(folder, file+'.txt'), 'r') as f:
            with open(os.path.join(save_folder, file+'.txt'), 'w') as t:
                lines = f.readlines()
                lines = [line.strip() for line in lines]
                for line in lines:
                    cl, x, y, w, h = line.split(' ')
                    cl = int(cl)
                    x, y, w, h = float(x), float(y), float(w), float(h)
                    if cl == 6:
                        cl = 0
                        t.write(f'{cl} {x} {y} {w} {h}\n')
                    elif cl == 5:
                        cl = 0
                        t.write(f'{cl} {x} {y} {w} {h}\n')
                    elif cl == 3:
                        cl = 1
                        t.write(f'{cl} {x} {y} {w} {h}\n')
                    elif cl == 4:
                        cl = 1
                        t.write(f'{cl} {x} {y} {w} {h}\n')


def read_yolo_labels_and_draw_rectangles(yolo_txt_folder, images_folder=None, result_folder=None, names=None):
    thickness = 3
    color = [0, 255, 0]
    for txt_file in tqdm(os.listdir(yolo_txt_folder)):
        if txt_file.endswith('txt'):
            path_to_txt = os.path.join(yolo_txt_folder, txt_file)
            path_to_img = os.path.join(images_folder, txt_file.replace('txt', 'jpg'))
            # print(path_to_img)
            with open(path_to_txt, 'r', encoding='utf8') as t:
                lines = [line.strip() for line in t.readlines()]
                # img = cv2.cvtColor(cv2.imread(path_to_img), cv2.COLOR_BGR2RGB)
                img = imread(path_to_img)
                # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                # print(img.shape)
                for line in lines:
                    cl, x, y, w, h = line.split(' ')
                    x, y, w, h = float(x), float(y), float(w), float(h),
                    ymin,ymax,xmin,xmax = xywh_to_ymin_ymax_xmin_xmax(x, y, w, h, img.shape[:2])
                    startpoint = xmin, ymin
                    endpoint = xmax, ymax
                    img = cv2.rectangle(img, startpoint, endpoint, color, thickness)
                    cv2.putText(img, names[int(cl)], (xmin, ymin), cv2.FONT_HERSHEY_COMPLEX, 1,
                                      (255, 0, 0), thickness)
                    # print(os.path.join(result_folder, txt_file.replace('txt', 'jpg')))
            imsave(os.path.join(result_folder, txt_file.replace('txt', 'png')), img)

def train_yolo():
    model = YOLO(r'yolov8x.pt')
    model.train(data=r'smokers.yaml', cfg=r'cfg.yaml')


if __name__ == '__main__':
    names = {
        1: "face",
        0: "wrist"
    }
    folder_from = r'C:\Users\79777\PycharmProjects\SmokersDetectionHack\merged_classes_init_dataset'
    folder_to = r'C:\Users\79777\PycharmProjects\SmokersDetectionHack\merged_classes_init_dataset_splited'
    folder_init = r'C:\Users\79777\PycharmProjects\SmokersDetectionHack\dataset\images\train'
    folder_save = r'C:\Users\79777\PycharmProjects\SmokersDetectionHack\Faces_new'
    os.makedirs(folder_save, exist_ok=True)
    folder = r'C:\Users\79777\PycharmProjects\SmokersDetectionHack\PhotoFromCameras'
    save_folder = r'C:\Users\79777\PycharmProjects\SmokersDetectionHack\merged_classes_init_dataset'
    root = r'C:\Users\79777\PycharmProjects\SmokersDetectionHack\dataset_detection'
    # make_folders(root)
    # split_samples_from_dir(save_folder, root)
    # read_yolo_labels_and_draw_rectangles(save_folder, save_folder, folder_save, names)
    # update_files(folder_from, folder_to, folder_init)
    save_small_pictures_from_annotations(folder, folder_save)
    # merge_classes(folder, save_folder)
    # train_yolo()
    # make_folders(root)