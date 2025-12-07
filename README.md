

## 👋 hello

**We write your reusable computer vision tools.** Whether you need to load your dataset from your hard drive, draw detections on an image or video, or count how many detections are in a zone. You can count on us! 🤝

## 💻 install

Pip install the supervision package in a
[**Python>=3.8**](https://www.python.org/) environment.

```bash
pip install supervision
```

Read more about conda, mamba, and installing from source in our [guide](https://roboflow.github.io/supervision/).

## 🔥 quickstart

### models

Supervision was designed to be model agnostic. Just plug in any classification, detection, or segmentation model. For your convenience, we have created [connectors](https://supervision.roboflow.com/latest/detection/core/#detections) for the most popular libraries like Ultralytics, Transformers, or MMDetection.

```python
import cv2
import supervision as sv
from ultralytics import YOLO

image = cv2.imread(...)
model = YOLO("yolov8s.pt")
result = model(image)[0]
detections = sv.Detections.from_ultralytics(result)

len(detections)
# 5
```

<details>
<summary>👉 more model connectors</summary>

- inference

  Running with [Inference](https://github.com/roboflow/inference) requires a [Roboflow API KEY](https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key).

  ```python
  import cv2
  import supervision as sv
  from inference import get_model

  image = cv2.imread(...)
  model = get_model(model_id="yolov8s-640", api_key=<ROBOFLOW API KEY>)
  result = model.infer(image)[0]
  detections = sv.Detections.from_inference(result)

  len(detections)
  # 5
  ```

</details>

### annotators

Supervision offers a wide range of highly customizable [annotators](https://supervision.roboflow.com/latest/detection/annotators/), allowing you to compose the perfect visualization for your use case.

```python
import cv2
import supervision as sv

image = cv2.imread(...)
detections = sv.Detections(...)

box_annotator = sv.BoxAnnotator()
annotated_frame = box_annotator.annotate(
  scene=image.copy(),
  detections=detections)
```

https://github.com/roboflow/supervision/assets/26109316/691e219c-0565-4403-9218-ab5644f39bce

### datasets

Supervision provides a set of [utils](https://supervision.roboflow.com/latest/datasets/core/) that allow you to load, split, merge, and save datasets in one of the supported formats.

```python
import supervision as sv
from roboflow import Roboflow

project = Roboflow().workspace(<WORKSPACE_ID>).project(<PROJECT_ID>)
dataset = project.version(<PROJECT_VERSION>).download("coco")

ds = sv.DetectionDataset.from_coco(
    images_directory_path=f"{dataset.location}/train",
    annotations_path=f"{dataset.location}/train/_annotations.coco.json",
)

path, image, annotation = ds[0]
    # loads image on demand

for path, image, annotation in ds:
    # loads image on demand
```

<details close>
<summary>👉 more dataset utils</summary>

- load

  ```python
  dataset = sv.DetectionDataset.from_yolo(
      images_directory_path=...,
      annotations_directory_path=...,
      data_yaml_path=...
  )

  dataset = sv.DetectionDataset.from_pascal_voc(
      images_directory_path=...,
      annotations_directory_path=...
  )

  dataset = sv.DetectionDataset.from_coco(
      images_directory_path=...,
      annotations_path=...
  )
  ```

- split

  ```python
  train_dataset, test_dataset = dataset.split(split_ratio=0.7)
  test_dataset, valid_dataset = test_dataset.split(split_ratio=0.5)

  len(train_dataset), len(test_dataset), len(valid_dataset)
  # (700, 150, 150)
  ```

- merge

  ```python
  ds_1 = sv.DetectionDataset(...)
  len(ds_1)
  # 100
  ds_1.classes
  # ['dog', 'person']

  ds_2 = sv.DetectionDataset(...)
  len(ds_2)
  # 200
  ds_2.classes
  # ['cat']

  ds_merged = sv.DetectionDataset.merge([ds_1, ds_2])
  len(ds_merged)
  # 300
  ds_merged.classes
  # ['cat', 'dog', 'person']
  ```

- save

  ```python
  dataset.as_yolo(
      images_directory_path=...,
      annotations_directory_path=...,
      data_yaml_path=...
  )

  dataset.as_pascal_voc(
      images_directory_path=...,
      annotations_directory_path=...
  )

  dataset.as_coco(
      images_directory_path=...,
      annotations_path=...
  )
  ```

- convert

  ```python
  sv.DetectionDataset.from_yolo(
      images_directory_path=...,
      annotations_directory_path=...,
      data_yaml_path=...
  ).as_pascal_voc(
      images_directory_path=...,
      annotations_directory_path=...
  )
  ```

</details>

