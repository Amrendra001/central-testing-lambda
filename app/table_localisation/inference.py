import pandas as pd
import json
import boto3
import os
from table_localisation.metrics_util import precision_recall, metrics_table, metrics_col, metrics_row, check_table, check_column, check_row
from global_variables import DATA_DIR_NAME, OCR_S3_PATH, TEST_S3_BUCKET, TEST_S3_PATH, LABELS_S3_PATH


def s3_cp(source, destination):
    sync_command = f'aws s3 cp "{source}" "{destination}"'
    os.system(sync_command)

def download_ocr(doc_id):
    source = f'{OCR_S3_PATH}/{doc_id}.parquet'
    destination = f'{DATA_DIR_NAME}/ocr/{doc_id}.parquet'
    os.makedirs(f'{DATA_DIR_NAME}/ocr/', exist_ok=True)
    s3_cp(source, destination)

def add(cum_TP, cum_FP, cum_FN, TP, FP, FN):
    return cum_TP + TP, cum_FP + FP, cum_FN + FN


def score(df, real_path, pred_path, thresholds):
    result = ""
    for thresh_iou in thresholds:
        cum_TP_col, cum_FP_col, cum_FN_col, cum_TP_row, cum_FP_row, cum_FN_row = 0, 0, 0, 0, 0, 0
        table_score_ls = []
        start = 0
        for filename, doc_id, page_no in zip(df['file_name'], df['doc_id'], df['page_number_index']):
            page_no -= 1
            doc_id = doc_id[:-4]
            filename = filename[:filename.rfind('.')] + '.json'
            if not os.path.isfile(f'{DATA_DIR_NAME}/ocr/{doc_id}.parquet'):
                download_ocr(doc_id)

            if check_table(real_path, pred_path, filename):
                table_score = metrics_table(real_path, pred_path, filename)
                table_score_ls.append(table_score)

            if check_column(real_path, pred_path, filename):
                TP, FP, FN = metrics_col(real_path, pred_path, filename, doc_id, thresh_iou)
                cum_TP_col, cum_FP_col, cum_FN_col = add(cum_TP_col, cum_FP_col, cum_FN_col, TP, FP, FN)

            if check_row(real_path, pred_path, filename):
                TP, FP, FN = metrics_row(real_path, pred_path, filename, doc_id, thresh_iou, page_no)
                cum_TP_row, cum_FP_row, cum_FN_row = add(cum_TP_row, cum_FP_row, cum_FN_row, TP, FP, FN)

        result += f'For Thresh IOU = {thresh_iou}\n'

        if check_table(real_path, pred_path, filename):
            avg_table_score = sum(table_score_ls) / len(table_score_ls)
            result += f'Average Table score = {avg_table_score}\n'

        if check_column(real_path, pred_path, filename):
            precision_col, recall_col = precision_recall(cum_TP_col, cum_FP_col, cum_FN_col)
            result += f'Column Seprators Precision = {precision_col}\n'
            result += f'Column Seprators Recall = {recall_col}\n'

        if check_row(real_path, pred_path, filename):
            precision_row, recall_row = precision_recall(cum_TP_row, cum_FP_row, cum_FN_row)
            result += f'Row Seprators Precision for = {precision_row}\n'
            result += f'Row Seprators Recall for = {recall_row}\n'
        result += '\n'
    return result


def invoke_localisation_lambda(input_data):
    input_data = json.dumps(input_data)

    table_data = {}

    try:
        lambda_client = boto3.client('lambda', region_name='ap-south-1')
        response = lambda_client.invoke(FunctionName='table-localisation-ml', Payload=input_data)
        if response['StatusCode'] in range(200, 300):
            response = response['Payload'].read()
            table_data = json.loads(response)
            # print('Table Localisation Lambda response:', table_data)
    except Exception as e:
        raise Exception(
            'Table Localisation Lambda Invocation Failed: ' + str(e))

    return table_data


def get_yolov5_pred(s3_path, s3_bucket):
    event = {'key': s3_path,
             'bucket': s3_bucket,
             'region': 'ap-south-1'}

    return invoke_localisation_lambda(event)


def get_score():
    df = pd.read_csv(f'{DATA_DIR_NAME}/test_set_v1.csv')
    os.makedirs(f'{DATA_DIR_NAME}/labels/', exist_ok=True)
    os.makedirs(f'{DATA_DIR_NAME}/model_outputs/', exist_ok=True)

    for file_name in df['file_name']:
        s3_bucket = TEST_S3_BUCKET
        s3_path = f'{TEST_S3_PATH}/images/{file_name}'
        data = get_yolov5_pred(s3_path, s3_bucket)
        json_file_name = file_name[:file_name.rfind('.')] + '.json'
        with open(f'{DATA_DIR_NAME}/model_outputs/' + json_file_name, 'w') as f:
            json.dump(data, f)

        s3_cp(f'{LABELS_S3_PATH}/{file_name[:-4]}.json', f'{DATA_DIR_NAME}/labels/{file_name[:-4]}.json')

    real_path = f'{DATA_DIR_NAME}/labels/'
    pred_path = f'{DATA_DIR_NAME}/model_outputs/'
    thresh_iou = [0.5, 0.9]
    return score(df, real_path, pred_path, thresh_iou)