import os
import yaml
import time
import shutil
import requests

from label_studio_tools.core.utils.io import get_local_path
from label_studio_ml.utils import get_env
from mmdet.utils import register_all_modules

ROOT = os.path.join(os.path.dirname(__file__))
print('=> ROOT = ', ROOT)
os.environ['HOSTNAME'] = 'http://localhost:2524'
os.environ['API_KEY'] = '39f2c408c24e5e4a7f2350eef99bf4647e157101'
HOSTNAME = get_env('HOSTNAME')
API_KEY = get_env('API_KEY')

print('=> LABEL STUDIO HOSTNAME = ', HOSTNAME)
if not API_KEY:
    print('=> WARNING! API_KEY is not set')

with open(os.path.join(ROOT, "projects/LabelStudio/backend_template/label_studio_conf.yaml"), errors='ignore') as f:
    conf = yaml.safe_load(f)

register_all_modules()

import os
import logging
import boto3
import io
import json

from mmdet.apis import init_detector, inference_detector

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import (
    get_image_size,
    get_single_tag_keys,
    DATA_UNDEFINED_NAME,
)
from label_studio_tools.core.utils.io import get_data_dir
from botocore.exceptions import ClientError
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class MMDetection(LabelStudioMLBase):
    """Object detector based on https://github.com/open-mmlab/mmdetection"""

    def __init__(
            self,
            config_file=conf['config_file'],
            checkpoint_file=conf['checkpoint_file'],
            image_dir=None,
            labels_file=None,
            score_threshold=0.3,
            device=conf['device'],
            **kwargs,
    ):
        """
        Load MMDetection model from config and checkpoint into memory.
        (Check https://mmdetection.readthedocs.io/en/v1.2.0/GETTING_STARTED.html#high-level-apis-for-testing-images)

        Optionally set mappings from COCO classes to target labels
        :param config_file: Absolute path to MMDetection config file (e.g. /home/user/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x.py)
        :param checkpoint_file: Absolute path MMDetection checkpoint file (e.g. /home/user/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth)
        :param image_dir: Directory where images are stored (should be used only in case you use direct file upload into Label Studio instead of URLs)
        :param labels_file: file with mappings from COCO labels to custom labels {"airplane": "Boeing"}
        :param score_threshold: score threshold to wipe out noisy results
        :param device: device (cpu, cuda:0, cuda:1, ...)
        :param kwargs:
        """
        super(MMDetection, self).__init__(**kwargs)
        config_file = config_file or os.environ['config_file']
        checkpoint_file = checkpoint_file or os.environ['checkpoint_file']
        self.config_file = config_file
        self.checkpoint_file = checkpoint_file
        self.labels_file = labels_file

        # default Label Studio image upload folder
        upload_dir = os.path.join(get_data_dir(), 'media', 'upload')
        self.image_dir = image_dir or upload_dir
        logger.debug(f'{self.__class__.__name__} reads images from {self.image_dir}')

        if self.labels_file and os.path.exists(self.labels_file):
            self.label_map = json_load(self.labels_file)
        else:
            self.label_map = {}

        (
            self.from_name,
            self.to_name,
            self.value,
            self.labels_in_config,
        ) = get_single_tag_keys(self.parsed_label_config, 'RectangleLabels', 'Image')
        schema = list(self.parsed_label_config.values())[0]
        self.labels_in_config = set(self.labels_in_config)

        print('Load new model from: ', config_file, checkpoint_file)
        self.model = init_detector(config_file, checkpoint_file, device=device)
        self.score_thresh = score_threshold

        self.labels_attrs = schema.get('labels_attrs')
        self.build_labels_from_labeling_config(schema)

    def build_labels_from_labeling_config(self, schema):
        """
        Collect label maps from `predicted_values="airplane,car"` attribute in <Label> tags,
        e.g. "airplane", "car" are label names from COCO dataset
        """
        mmdet_labels = self.model.dataset_meta.get('classes', [])
        mmdet_labels_lower = [label.lower() for label in mmdet_labels]
        print('COCO dataset labels supported by this mmdet model:', self.model.dataset_meta)

        # if labeling config has Label tags
        if self.labels_attrs:
            # try to find something like <Label value="Vehicle" predicted_values="airplane,car">
            for ls_label, label_attrs in self.labels_attrs.items():
                predicted_values = label_attrs.get('predicted_values', '').split(',')
                for predicted_value in predicted_values:
                    if predicted_value:  # it shouldn't be empty (like '')
                        if predicted_value not in mmdet_labels:
                            print(f'Predicted value "{predicted_value}" is not in mmdet labels')
                        self.label_map[predicted_value] = ls_label

        if not self.label_map:
            # try to find thin <Label value="Vehicle" predicted_values="airplane,car">
            for ls_label, _ in self.labels_attrs.items():
                try:
                    index = mmdet_labels_lower.index(ls_label.lower())
                except ValueError:
                    continue  # no label found in mmdet labels
                else:
                    self.label_map[mmdet_labels[index]] = ls_label

        print('MMDetection => Label Studio mapping of labels :\n', self.label_map)

    def _get_image_url(self, task):
        image_url = task['data'].get(self.value) or task['data'].get(
            DATA_UNDEFINED_NAME
        )
        if image_url.startswith('s3://'):
            # pre-sign s3 url
            r = urlparse(image_url, allow_fragments=False)
            bucket_name = r.netloc
            key = r.path.lstrip('/')
            client = boto3.client('s3')
            try:
                image_url = client.generate_presigned_url(
                    ClientMethod='get_object',
                    Params={'Bucket': bucket_name, 'Key': key},
                )
            except ClientError as exc:
                logger.warning(
                    f'Can\'t generate presigned URL for {image_url}. Reason: {exc}'
                )
        return image_url

    def predict(self, tasks, **kwargs):
        if len(tasks) > 1:
            print("==> Only the first task will be processed to avoid ML backend overloading")
            tasks = [tasks[0]]

        predictions = []
        for task in tasks:
            prediction = self.predict_one_task(task)
            predictions.append(prediction)
        return predictions

    def predict_one_task(self, task):
        image_url = self._get_image_url(task)
        image_path = self.get_local_path(image_url)
        model_results = inference_detector(self.model, image_path).pred_instances
        results = []
        all_scores = []
        img_width, img_height = get_image_size(image_path)
        classes = self.model.dataset_meta.get('classes')
        print(f">>> model_results: {model_results}")
        print(f">>> label_map {self.label_map}")
        print(f">>> classes {classes}")
        print(f">>> self.model.dataset_meta: {self.model.dataset_meta}")

        for item in model_results:
            bboxes, label, scores = item['bboxes'], item['labels'][0], item['scores']
            mmdet_label = classes[label]
            score = float(scores[-1])
            print('----------------------')
            print(f"task id > {task.get('id')}")
            print(f"bboxes > {bboxes}")
            print(f"label > {mmdet_label}")
            print(f"score > {score}")

            # bbox score is too low
            if score < self.score_thresh:
                continue

            output_label = mmdet_label

            for bbox in bboxes:
                bbox = list(bbox)
                if not bbox:
                    continue

                x, y, xmax, ymax = bbox[:4]
                results.append(
                    {
                        'from_name': self.from_name,
                        'to_name': self.to_name,
                        'type': 'rectanglelabels',
                        'value': {
                            'rectanglelabels': [output_label],
                            'x': float(x) / img_width * 100,
                            'y': float(y) / img_height * 100,
                            'width': (float(xmax) - float(x)) / img_width * 100,
                            'height': (float(ymax) - float(y)) / img_height * 100,
                        },
                        'score': score,
                    }
                )
                all_scores.append(score)
        avg_score = sum(all_scores) / max(len(all_scores), 1)
        print(f">>> RESULTS: {results}")
        return {'result': results, 'score': avg_score, 'model_version': 'mmdet'}

    # def fit(self, completions, num_epochs=5, **kwargs):

    # 需要设计并实现更强大的迭代训练策略。

    def gen_train_data(self, project_id):
        import zipfile
        import glob
        download_url = f'{HOSTNAME.rstrip("/")}/api/projects/{project_id}/export?export_type=COCO&download_all_tasks=false&download_resources=true'
        response = requests.get(download_url, headers={'Authorization': f'Token {API_KEY}'})
        zip_path = os.path.join(conf['workdir'], "train.zip")
        train_path = os.path.join(conf['workdir'], "train")

        with open(zip_path, 'wb') as file:
            file.write(response.content)  # 通过二进制写文件的方式保存获取的内容
            file.flush()
        f = zipfile.ZipFile(zip_path)  # 创建压缩包对象
        f.extractall(train_path)  # 压缩包解压缩
        f.close()
        os.remove(zip_path)
        if not os.path.exists(os.path.join(train_path, "images", str(project_id))):
            os.makedirs(os.path.join(train_path, "images", str(project_id)))
        for img in glob.glob(os.path.join(train_path, "images", "*.jpg")):
            basename = os.path.basename(img)
            shutil.move(img, os.path.join(train_path, "images", str(project_id), basename))
        return True


def json_load(file, int_keys=False):
    with io.open(file, encoding='utf8') as f:
        data = json.load(f)
        if int_keys:
            return {int(k): v for k, v in data.items()}
        else:
            return data
