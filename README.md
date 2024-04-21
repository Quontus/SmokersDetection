# SmokersDetection
Training+inference for smokers detection
### Installation
  ```sh
  pip install -r requirements.txt
  ```
### Запуск обучения (модель детекции yolov8s) (внести изменения в cfg.yaml и smokers.yaml)
В smokers.yaml указать пути к датасету
  ```sh
  python train.py
  ```
### Запуск обучения классификатора (инструкция в блокноте - XGBoost_vgg16_plus_dense121.ipynb)

### Запуск инференса каскада моделей (model_inference.py)
Указать путь в файлу с изображениями - вход, выход - Словарь: {название файла: 1|0, 0 -не курят, 1 - курят}
  ```sh
  folder_with_img = r'C:\Users\79777\PycharmProjects\SmokersDetectionHack\dataset_upd\images\val'
  ```
