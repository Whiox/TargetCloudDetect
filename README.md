# TargetCloudDetect

Python-библиотека для автоматического определения подходящей точки на
изображении и размещения графического элемента относительно найденной области.

Проект создан для команды моего бота:
[discord-wlf](https://github.com/Whiox/discord-wlf)

## Как это работает

В основе используется модель на базе EfficientNet.

Изображение приводится к размеру `224 × 224`, после чего модель формирует
сетку признаков `7 × 7`. Для каждой ячейки предсказываются:

- смещение точки по оси X;
- смещение точки по оси Y;
- уверенность в наличии целевого объекта.

Ячейка с максимальной уверенностью используется для восстановления координат
на исходном изображении.

Модель обучена на вручную собранном и размеченном датасете.

## Возможности

- определение координат целевой области;
- получение оценки уверенности модели;
- обработка изображений из файла или массива байтов;
- автоматическое размещение облака ответа;
- готовые веса входят в состав пакета;

## Установка

```bash
pip install git+https://github.com/Whiox/TargetCloudDetect.git
```

## Использование

### Получение координат
```python
from PIL import Image
from target_cloud_detect import Model

model = Model()
image = Image.open("image.png")

x, y, confidence = model.predict_x_y_conf(image)

print(f"Координаты: {x}, {y}")
print(f"Уверенность: {confidence:.2%}")
```

### Обработка файла
```python
from target_cloud_detect import Model

model = Model()
result = model.predict_from_file("image.png")
result.save("result.png")
```

### Обработка массива байтов
```python
from target_cloud_detect import Model

model = Model()

with open("image.png", "rb") as file:
    result = model.predict_from_bytes(file.read())

result.save("result.png")
```

### Собственные веса можно передать при создании модели:
```python
model = Model("path/to/model.keras")
```

## Обучение
Блокнот с архитектурой и процессом обучения модели доступен в
[Google Colab](https://colab.research.google.com/drive/1xF9R73FT-Msx6JSpzuJC86G_XA7ianWZ?usp=sharing)


## Стек
- Python
- Keras
- TensorFlow
- EfficientNet
- NumPy
- Pillow
