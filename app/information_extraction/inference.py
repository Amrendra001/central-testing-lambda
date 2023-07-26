import json
import os
import time
import boto3
import itertools
import datasets
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from transformers import AutoModelForTokenClassification, AutoProcessor
import evaluate
import numpy as np
from functools import wraps
import warnings
from tqdm import tqdm
from utils import invoke_localisation_lambda

warnings.filterwarnings('ignore')

BASE_DIR = 'data'
CHECKPOINTS_DIR = 'checkpoints'
MODEL_PATH = 'data/models/checkpoint-2702'
DATA_DIR = 'data'
LABEL_TO_COLOUR_MAP = {
    'Item_Desc': 'green',
    'Qty': 'red',
    'Free_Qty': 'blue',
    'Pack': 'yellow',
    'Qty_Plus_Free': 'purple',
    'Other': 'black'
}


#
# processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
# model = AutoModelForTokenClassification.from_pretrained(f"{BASE_DIR}/{CHECKPOINTS_DIR}/{MODEL_PATH}/")


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__} took {total_time:.4f} seconds')
        return result

    return timeit_wrapper


def get_file_name_from_path(path):
    return path.rsplit('/')[-1].rsplit('.')[0]


def iob_to_label(label):
    label = label[2:]
    if not label:
        return 'other'
    return label


def get_layoutlm_bboxes_from_ocr_data(page_ocr_data):
    page_ocr_data['minx'] = (page_ocr_data['minx'] * 1000).astype(int)
    page_ocr_data['miny'] = (page_ocr_data['miny'] * 1000).astype(int)
    page_ocr_data['maxx'] = (page_ocr_data['maxx'] * 1000).astype(int)
    page_ocr_data['maxy'] = (page_ocr_data['maxy'] * 1000).astype(int)
    page_ocr_data['bboxes'] = [[row['minx'], row['miny'], row['maxx'], row['maxy']]
                               for _, row in page_ocr_data.iterrows()]

    return page_ocr_data['bboxes'].tolist()


def restructure_batched_encodings(encoding):
    # change the shape of input_ids
    x = []
    for i in range(0, len(encoding['input_ids'])):
        x.append(encoding['input_ids'][i])
    x = torch.stack(x)
    encoding['input_ids'] = x

    # change the shape of pixel values
    x = []
    for i in range(0, len(encoding['pixel_values'])):
        x.append(encoding['pixel_values'][i])
    x = torch.stack(x)
    encoding['pixel_values'] = x

    # change the shape of attention_mask
    x = []
    for i in range(0, len(encoding['attention_mask'])):
        x.append(torch.tensor(encoding['attention_mask'][i]))
    x = torch.stack(x)
    encoding['attention_mask'] = x

    # change the shape of bbox
    x = []
    for i in range(0, len(encoding['bbox'])):
        x.append(torch.tensor(encoding['bbox'][i]))
    x = torch.stack(x)
    encoding['bbox'] = x

    return encoding


def get_encoding(image, words, bboxes):
    encoding = processor(image, words, boxes=bboxes, max_length=512, stride=128,
                         truncation=True, padding="max_length", return_overflowing_tokens=True,
                         return_offsets_mapping=True, return_tensors="pt")
    encoding.pop('offset_mapping')
    encoding.pop('overflow_to_sample_mapping')

    encoding = restructure_batched_encodings(encoding)

    return encoding


def do_prediction(encoding):
    # Do prediction
    with torch.no_grad():
        outputs = model(**encoding)

    logits = outputs.logits
    predictions = logits.argmax(-1).squeeze().tolist()

    return predictions


def process_predictions_and_bboxes(predictions, token_bboxes):
    if (len(token_bboxes) == 512):
        predictions = [predictions]
        token_bboxes = [token_bboxes]

    predictions = list(itertools.chain(*predictions))
    token_bboxes = list(itertools.chain(*token_bboxes))
    predictions = [iob_to_label(model.config.id2label[sub_pred]) for idx, sub_pred in enumerate(predictions)]

    return predictions, token_bboxes


@timeit
def inference(example, words, bboxes):
    encoding = get_encoding(Image.open(example["image"]).convert("RGB"), words, bboxes)
    token_bboxes = encoding.bbox.squeeze().tolist()

    predictions = do_prediction(encoding)

    predictions, token_bboxes = process_predictions_and_bboxes(predictions, token_bboxes)

    return predictions, token_bboxes


def get_label_list(labels):
    unique_labels = set()
    for label in labels:
        unique_labels = unique_labels | set(label)
    label_list = list(unique_labels)
    label_list.sort()
    return label_list


def download_from_s3(filename, bucket_name, s3_filename, s3_client=None):
    if not s3_client:
        s3_client = boto3.client('s3')
    s3_client.download_file(bucket_name, s3_filename, filename)
    print(f'Downloaded file {s3_filename} from S3 bucket {bucket_name}')


def get_ocr_data_for_doc(document_s3_path):
    doc_id_without_extension = get_file_name_from_path(document_s3_path)
    ocr_output_filename = f'/tmp/{doc_id_without_extension}.parquet'
    ocr_output_filename_s3 = f'ocr_output/{doc_id_without_extension}.parquet'
    download_from_s3(ocr_output_filename, 'javis-ai-parser-dev', ocr_output_filename_s3)
    word_df = pd.read_parquet(ocr_output_filename)

    return word_df


def get_layoutlm_bboxes_from_ocr_data(page_ocr_data):
    page_ocr_data['minx'] = (page_ocr_data['minx'] * 1000).astype(int)
    page_ocr_data['miny'] = (page_ocr_data['miny'] * 1000).astype(int)
    page_ocr_data['maxx'] = (page_ocr_data['maxx'] * 1000).astype(int)
    page_ocr_data['maxy'] = (page_ocr_data['maxy'] * 1000).astype(int)
    page_ocr_data['bboxes'] = [[row['minx'], row['miny'], row['maxx'], row['maxy']]
                               for _, row in page_ocr_data.iterrows()]

    return page_ocr_data['bboxes'].tolist()


def filter_misc_labels(predictions, token_bboxes):
    token_bboxes = [bbox for idx, bbox in enumerate(token_bboxes) if predictions[idx].lower() != 'other']
    predictions = [pred for pred in predictions if pred.lower() != 'other']

    return predictions, token_bboxes


def convert_predictions_to_table(predictions, token_bboxes, page_ocr_data):
    predictions, token_bboxes = filter_misc_labels(predictions, token_bboxes)

    page_ocr_data['bbox_key_string'] = page_ocr_data['bboxes'].apply(lambda bbox: ','.join(list(map(str, bbox))))
    token_bbox_key_strings = [','.join(list(map(str, bbox))) for bbox in token_bboxes]

    for predicted_label, token_bbox_key_string in zip(predictions, token_bbox_key_strings):
        page_ocr_data.loc[page_ocr_data['bbox_key_string'] == token_bbox_key_string, 'label'] = predicted_label

    page_ocr_data = page_ocr_data[page_ocr_data['label'].notnull()]
    level_label_grouped_df = page_ocr_data.groupby(['level_id', 'label'])
    agg_table = level_label_grouped_df.agg({'word': ' '.join}).reset_index()
    page_table = agg_table.pivot(index='level_id', columns='label', values='word')

    return page_table


def unnormalize_layoutlm_box(bbox, width, height):
    return [
        width * (bbox[0] / 1000),
        height * (bbox[1] / 1000),
        width * (bbox[2] / 1000),
        height * (bbox[3] / 1000),
    ]


@timeit
def visualise_predictions(image, predictions, token_bboxes, output_file_path=None):
    image = Image.open(image).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    img_width, img_height = image.size

    predictions, token_bboxes = filter_misc_labels(predictions, token_bboxes)
    token_bboxes = [unnormalize_layoutlm_box(sub_box, img_width, img_height) for idx, sub_box in
                    enumerate(token_bboxes)]

    for predicted_label, box in zip(predictions, token_bboxes):
        if box[2] < box[0] or box[3] < box[1]:
            if predicted_label != 'other':
                print('Problematic box:', predicted_label, box)
            box[2] = box[0] + abs(box[2] - box[0])
            box[3] = box[1] + abs(box[3] - box[1])
        draw.rectangle(box, outline=LABEL_TO_COLOUR_MAP.get(predicted_label, 'orange'))
        draw.text((box[0] + 10, box[1] - 10), text=predicted_label,
                  fill=LABEL_TO_COLOUR_MAP.get(predicted_label, 'orange'), font=font)

    if not output_file_path:
        output_file_path = f'{BASE_DIR}/predictions_{str(int(time.time()))}.jpg'

    output_dir = output_file_path[:output_file_path.rfind('/')]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image.save(output_file_path)


@timeit
def driver(example):
    ocr_data = get_ocr_data_for_doc(example['doc_id'])
    page_ocr_data = ocr_data[ocr_data['page'] == example['page'][0]]

    words = page_ocr_data['word'].tolist()
    bboxes = get_layoutlm_bboxes_from_ocr_data(page_ocr_data)

    labels_true = [iob_to_label(sub_pred) for idx, sub_pred in enumerate(example['ner_tags'])]
    boxes_true = [list(map(lambda x: int(x), bbox)) for bbox in example['bboxes']]
    page_table_true = convert_predictions_to_table(labels_true, boxes_true, page_ocr_data)

    return page_table_true


def s3_sync(source, destination):
    sync_command = f'aws s3 sync "{source}" "{destination}"'
    os.system(sync_command)


def get_E2E_doc_parser_pred(doc_id):
    event = {
        'ocr_s3_path': f'ocr_output/{doc_id}.parquet',
        'images_s3_prefix': f'images_of_pdf_pages/{doc_id}/',
        'images_s3_bucket': 'javis-ai-parser-dev',
        'model_type': 'LayoutLMv3'
    }

    return invoke_localisation_lambda(event, 'E2E_Document_Parser')

def get_OCR(doc_id):
    event = {
        "service": "default",
        "s3_pdf_path": f"Attachments/{doc_id}.PDF",
        "s3_bucket": "email-attachment-dev",
        "s3_region": "ap-south-1",
        "aggregation_level": "word",
        "to_standardize": False,
        "write_to_s3": True,
        "s3_file_type": "parquet",
        "ocr_s3_bucket": "javis-ai-parser-dev",
        "ocr_s3_key": f"ocr_output/{doc_id}.parquet",
        "send_output": False
    }
    return invoke_localisation_lambda(event, 'OCR')


def mapping_fn(entity):
    return {k:v for k,v in entity.items() if k in ['Item_Desc', 'Pack', 'Qty_Plus_Free']}

def get_true_labels(dataset):
    os.makedirs(f'data/ground_truth/', exist_ok=True)
    for i, example in tqdm(enumerate(dataset['test'])):
        doc_id = example['doc_id'][:-4]

        if doc_id in ['20228228282112986']:
            continue

        page_table_true = driver(example)
        page_table_true.reset_index(drop=True, inplace=True)

        if os.path.isfile(f'data/ground_truth/{doc_id}.csv'):
            df = pd.read_csv(f'data/ground_truth/{doc_id}.csv')
            df = pd.concat([df, page_table_true], ignore_index=True)
            df.to_csv(f'data/ground_truth/{doc_id}.csv', index=False)
        else:
            page_table_true.to_csv(f'data/ground_truth/{doc_id}.csv', index=False)

def get_pred_labels(dataset):
    os.makedirs(f'data/prediction/', exist_ok=True)
    for i, example in tqdm(enumerate(dataset['test'])):
        doc_id = example['doc_id'][:-4]

        if os.path.isfile(f'data/prediction/{doc_id}.csv'):
            continue

        if doc_id in ['20228228282112986']:
            continue

        get_OCR(doc_id)
        page_pred = get_E2E_doc_parser_pred(doc_id)
        page_table = pd.DataFrame(page_pred['sku_table']).loc[:, [x['Header'] for x in page_pred['header_mapping_data']['0']]]
        page_table.to_csv(f'data/prediction/{doc_id}.csv', index=False)

def get_score(dataset):
    fully_correct_pages = 0
    correct_docs = []
    incorrect_docs = []
    start = 0

    for i, example in tqdm(enumerate(dataset['test'])):
        doc_id = example['doc_id'][:-4]
        if doc_id in ['20228228282112986']:
            continue

        page_table_true = pd.read_csv(f'data/ground_truth/{doc_id}.csv')
        page_table_pred = pd.read_csv(f'data/prediction/{doc_id}.csv')

        if page_table_true.equals(page_table_pred):
            fully_correct_pages += 1
        #     correct_docs.append(get_file_name_from_path(example['image']))
        #     print(f'Doc #{(i + 1)} Num fully correct:', f"{fully_correct_pages} / {len(dataset['test'])}")
        # else:
        #     incorrect_docs.append(get_file_name_from_path(example['image']))

    return fully_correct_pages

def inference_2():

    # s3_sync('s3://document-ai-training-data/training_data/information_extraction/LayoutLMv3/processed_hf_dataset3/', 'data/')
    # s3_sync('s3://document-ai-training-data/information_extraction_models/LayoutLMv3/true_label_images/', f'{BASE_DIR}/{CHECKPOINTS_DIR}/information_extraction_models/LayoutLMv3/true_label_images/')
    dataset = datasets.load_from_disk(f'{DATA_DIR}/', keep_in_memory=False)

    # get_true_labels(dataset)
    # get_pred_labels(dataset)

    fully_correct_pages = get_score(dataset)

    return None