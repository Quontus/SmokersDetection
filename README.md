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
В smokers.yaml указать пути к датасету
  ```sh
  python train.py
  ```
