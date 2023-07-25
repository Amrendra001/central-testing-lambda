import pandas as pd
import json
import boto3
import os
from table_localisation.metrics_util import precision_recall, metrics_table, metrics_col, metrics_row, check_table, check_column, check_row
from global_variables import LOCAL_DATA_DIR, OCR_S3_PATH, TEST_S3_BUCKET, TEST_S3_PATH, LABELS_S3_PATH, BEST_RESULT_S3_PATH
from lambda_utils import call_email_lambda


def s3_cp(source, destination):
    sync_command = f'aws s3 cp "{source}" "{destination}"'
    os.system(sync_command)

def download_ocr(doc_id):
    source = f'{OCR_S3_PATH}/{doc_id}.parquet'
    destination = f'{LOCAL_DATA_DIR}/ocr/{doc_id}.parquet'
    os.makedirs(f'{LOCAL_DATA_DIR}/ocr/', exist_ok=True)
    s3_cp(source, destination)

def add(cum_TP, cum_FP, cum_FN, TP, FP, FN):
    return cum_TP + TP, cum_FP + FP, cum_FN + FN


def get_score(df, real_path, pred_path, thresholds):
    result = dict()
    for thresh_iou in thresholds:
        cum_TP_col, cum_FP_col, cum_FN_col, cum_TP_row, cum_FP_row, cum_FN_row = 0, 0, 0, 0, 0, 0
        table_score_ls = []
        for filename, doc_id, page_no in zip(df['file_name'], df['doc_id'], df['page_number_index']):
            page_no -= 1
            doc_id = doc_id[:-4]
            filename = filename[:filename.rfind('.')] + '.json'
            if not os.path.isfile(f'{LOCAL_DATA_DIR}/ocr/{doc_id}.parquet'):
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

        thresh_key = str(thresh_iou)
        result[thresh_key] = dict()

        if check_table(real_path, pred_path, filename):
            avg_table_score = sum(table_score_ls) / len(table_score_ls)
            result[thresh_key]['Average Table Score'] = avg_table_score

        if check_column(real_path, pred_path, filename):
            precision_col, recall_col = precision_recall(cum_TP_col, cum_FP_col, cum_FN_col)
            result[thresh_key]['Column Seprators Precision'] = precision_col
            result[thresh_key]['Column Seprators Recall'] = recall_col

        if check_row(real_path, pred_path, filename):
            precision_row, recall_row = precision_recall(cum_TP_row, cum_FP_row, cum_FN_row)
            result[thresh_key]['Row Seprators Precision'] = precision_row
            result[thresh_key]['Row Seprators Recall'] = recall_row

    return result


def get_bucket_analysis(df_org, real_path, pred_path, thresholds):
    result = dict()
    output = 'Bucket Analysis: <br>'
    col_ls = ['format', 'structuring', 'row_levels', 'table_size', 'divisions_presence', 'partial_structure']
    for col in col_ls:
        result[col] = dict()
        for bucket_type in df_org[col].unique():
            if pd.isna(bucket_type): continue
            result[col][bucket_type] = dict()
            df = df_org[df_org[col] == bucket_type]
            output += '******************************************************************************************<br>'
            output += f'Column = {col} <br>'
            output += f'Bucket Type = {bucket_type} <br>'
            for thresh_iou in thresholds:
                result[col][bucket_type][thresh_iou] = dict()
                cum_TP_col, cum_FP_col, cum_FN_col, cum_TP_row, cum_FP_row, cum_FN_row = 0, 0, 0, 0, 0, 0
                table_score_ls = []
                for filename, doc_id, page_no in zip(df['file_name'], df['doc_id'], df['page_number_index']):
                    page_no -= 1
                    doc_id = doc_id[:-4]
                    filename = filename[:filename.rfind('.')] + '.json'
                    if check_table(real_path, pred_path, filename):
                        table_score = metrics_table(real_path, pred_path, filename)
                        table_score_ls.append(table_score)

                    if check_column(real_path, pred_path, filename):
                        TP, FP, FN = metrics_col(real_path, pred_path, filename, doc_id, thresh_iou)
                        cum_TP_col += TP
                        cum_FP_col += FP
                        cum_FN_col += FN

                    if check_row(real_path, pred_path, filename):
                        TP, FP, FN = metrics_row(real_path, pred_path, filename, doc_id, thresh_iou)
                        cum_TP_row += TP
                        cum_FP_row += FP
                        cum_FN_row += FN

                output += f'For Thresh IOU = {thresh_iou} <br>'

                if check_table(real_path, pred_path, filename):
                    avg_table_score = sum(table_score_ls) / len(table_score_ls)
                    output += f'Average Table Score = {avg_table_score} <br>'
                    result[col][bucket_type][thresh_iou]['Average Table Score'] = avg_table_score

                if check_column(real_path, pred_path, filename):
                    precision_col, recall_col = precision_recall(cum_TP_col, cum_FP_col, cum_FN_col)
                    output += f'For Column Seprators <br>'
                    result[col][bucket_type][thresh_iou]['Column Seprators'] = dict()
                    output += f'TP = {cum_TP_col} <br>'
                    output += f'FP = {cum_FP_col} <br>'
                    output += f'FN = {cum_FN_col} <br>'
                    output += f'Precision = {precision_col} <br>'
                    output += f'Recall = {recall_col} <br>'
                    result[col][bucket_type][thresh_iou]['Column Seprators']['TP'] = cum_TP_col
                    result[col][bucket_type][thresh_iou]['Column Seprators']['FP'] = cum_FP_col
                    result[col][bucket_type][thresh_iou]['Column Seprators']['FN'] = cum_FN_col
                    result[col][bucket_type][thresh_iou]['Column Seprators']['Precision'] = precision_col
                    result[col][bucket_type][thresh_iou]['Column Seprators']['Recall'] = recall_col

                if check_row(real_path, pred_path, filename):
                    precision_row, recall_row = precision_recall(cum_TP_row, cum_FP_row, cum_FN_row)
                    output += f'For Row Seprators <br>'
                    result[col][bucket_type][thresh_iou]['Row Seprators'] = dict()
                    output += f'TP = {cum_TP_row} <br>'
                    output += f'FP = {cum_FP_row} <br>'
                    output += f'FN = {cum_FN_row} <br>'
                    output += f'Precision = {precision_row} <br>'
                    output += f'Recall = {recall_row} <br>'
                    result[col][bucket_type][thresh_iou]['Row Seprators']['TP'] = cum_TP_row
                    result[col][bucket_type][thresh_iou]['Row Seprators']['FP'] = cum_FP_row
                    result[col][bucket_type][thresh_iou]['Row Seprators']['FN'] = cum_FN_row
                    result[col][bucket_type][thresh_iou]['Row Seprators']['Precision'] = precision_row
                    result[col][bucket_type][thresh_iou]['Row Seprators']['Recall'] = recall_row
                output += '<br>'

    return result, output


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

def get_model_output(df):
    for file_name in df['file_name']:
        s3_bucket = TEST_S3_BUCKET
        s3_path = f'{TEST_S3_PATH}/images/{file_name}'
        data = get_yolov5_pred(s3_path, s3_bucket)
        json_file_name = file_name[:file_name.rfind('.')] + '.json'
        with open(f'{LOCAL_DATA_DIR}/model_outputs/' + json_file_name, 'w') as f:
            json.dump(data, f)

        s3_cp(f'{LABELS_S3_PATH}/{file_name[:-4]}.json', f'{LOCAL_DATA_DIR}/labels/{file_name[:-4]}.json')


def get_best_result():
    s3_cp(BEST_RESULT_S3_PATH, f'{LOCAL_DATA_DIR}/best_result.json')
    with open(f'{LOCAL_DATA_DIR}/best_result.json', 'r') as f:
        best_result = json.loads(f.read())
    return best_result


def get_comparision(best_result, model_result):
    output = ''
    for thresh_iou in best_result.keys():
        output += f'For Thresh IOU = {thresh_iou} <br>'
        for score in best_result[thresh_iou].keys():
            output += f'{score} for best_result={best_result[thresh_iou][score]} \t current_model={model_result[thresh_iou][score]} <br>'
        output += '<br>'
    return output


def inference():
    df = pd.read_csv(f'{LOCAL_DATA_DIR}/test_set_v1.csv')
    os.makedirs(f'{LOCAL_DATA_DIR}/labels/', exist_ok=True)
    os.makedirs(f'{LOCAL_DATA_DIR}/model_outputs/', exist_ok=True)

    # get_model_output(df)

    real_path = f'{LOCAL_DATA_DIR}/labels/'
    pred_path = f'{LOCAL_DATA_DIR}/model_outputs/'
    thresh_iou = [0.5, 0.9]
    model_result = get_score(df, real_path, pred_path, thresh_iou)
    best_result = get_best_result()
    compare_result = get_comparision(best_result, model_result)

    bucket_result, bucket_output = get_bucket_analysis(df, real_path, pred_path, thresh_iou)
    final_output = compare_result + '<br>' + bucket_output
    call_email_lambda(final_output, 'Table Localisation')

    return None