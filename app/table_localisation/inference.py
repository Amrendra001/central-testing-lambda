import pandas as pd
import json
import os
from table_localisation.metrics_util import precision_recall, metrics_table, metrics_col, metrics_row, check_table, \
    check_column, check_row
from global_variables import LOCAL_DATA_DIR, OCR_S3_PATH, TEST_S3_BUCKET, TEST_S3_PATH, LABELS_S3_PATH, \
    BEST_RESULT_S3_PATH
from lambda_utils import call_email_lambda
from utils import invoke_localisation_lambda


def s3_cp(source, destination):
    sync_command = f'aws s3 cp "{source}" "{destination}"'
    os.system(sync_command)


def download_ocr(doc_id):
    """
        Getting ocr output file from s3 for a particular doc_id.
    :param doc_id:
    :return:
    """
    source = f'{OCR_S3_PATH}/{doc_id}.parquet'
    destination = f'{LOCAL_DATA_DIR}/ocr/{doc_id}.parquet'
    os.makedirs(f'{LOCAL_DATA_DIR}/ocr/', exist_ok=True)
    s3_cp(source, destination)


def add(cum_TP, cum_FP, cum_FN, TP, FP, FN):
    return cum_TP + TP, cum_FP + FP, cum_FN + FN


def f1_score(precision, recall):
    return (2 * precision * recall) / (precision + recall)


def get_score(df, real_path, pred_path, thresholds):
    """
        Getting table, column and row score for prediction. Returns dictionary of results with scores.
    :param df: CSV with test set details.
    :param real_path: Path to real labels.
    :param pred_path: Path to predicted labels.
    :param thresholds: IOU threshold
    :return: Dictionary of results with scores
    """
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
            result[thresh_key]['Column Seprators F1 Score'] = f1_score(precision_col, recall_col)

        if check_row(real_path, pred_path, filename):
            precision_row, recall_row = precision_recall(cum_TP_row, cum_FP_row, cum_FN_row)
            result[thresh_key]['Row Seprators Precision'] = precision_row
            result[thresh_key]['Row Seprators Recall'] = recall_row
            result[thresh_key]['Row Seprators F1 Score'] = f1_score(precision_row, recall_row)

    return result


def get_bucket_analysis(df_org, real_path, pred_path, thresholds):
    """
        Function for bucket analysis i.e. get score on each bucket of test data.
    :param df_org: CSV with test set details.
    :param real_path: Path to real labels.
    :param pred_path: Path to predicted labels.
    :param thresholds: IOU threshold
    :return: Dictionary of results with scores and email output.
    """
    result = dict()
    output = 'Bucket Analysis: <br>'
    col_ls = ['format']
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
                    output += f'Average Table Score = {avg_table_score:.4f} <br>'
                    result[col][bucket_type][thresh_iou]['Average Table Score'] = avg_table_score

                if check_column(real_path, pred_path, filename):
                    precision_col, recall_col = precision_recall(cum_TP_col, cum_FP_col, cum_FN_col)
                    output += f'For Column Seprators <br>'
                    result[col][bucket_type][thresh_iou]['Column Seprators'] = dict()
                    output += f'Precision = {precision_col:.4f} <br>'
                    output += f'Recall = {recall_col:.4f} <br>'
                    output += f'F1 Score = {f1_score(precision_col, recall_col):.4f} <br>'
                    result[col][bucket_type][thresh_iou]['Column Seprators']['TP'] = cum_TP_col
                    result[col][bucket_type][thresh_iou]['Column Seprators']['FP'] = cum_FP_col
                    result[col][bucket_type][thresh_iou]['Column Seprators']['FN'] = cum_FN_col
                    result[col][bucket_type][thresh_iou]['Column Seprators']['Precision'] = precision_col
                    result[col][bucket_type][thresh_iou]['Column Seprators']['Recall'] = recall_col
                    result[col][bucket_type][thresh_iou]['Column Seprators']['F1 Score'] = f1_score(precision_col,
                                                                                                    recall_col)

                if check_row(real_path, pred_path, filename):
                    precision_row, recall_row = precision_recall(cum_TP_row, cum_FP_row, cum_FN_row)
                    output += f'For Row Seprators <br>'
                    result[col][bucket_type][thresh_iou]['Row Seprators'] = dict()
                    output += f'Precision = {precision_row:.4f} <br>'
                    output += f'Recall = {recall_row:.4f} <br>'
                    output += f'F1 Score = {f1_score(precision_row, recall_row):.4f} <br>'
                    result[col][bucket_type][thresh_iou]['Row Seprators']['TP'] = cum_TP_row
                    result[col][bucket_type][thresh_iou]['Row Seprators']['FP'] = cum_FP_row
                    result[col][bucket_type][thresh_iou]['Row Seprators']['FN'] = cum_FN_row
                    result[col][bucket_type][thresh_iou]['Row Seprators']['Precision'] = precision_row
                    result[col][bucket_type][thresh_iou]['Row Seprators']['Recall'] = recall_row
                    result[col][bucket_type][thresh_iou]['Row Seprators']['F1 Score'] = f1_score(precision_row,
                                                                                                 recall_row)
                output += '<br>'

    return result, output


def get_yolov5_pred(s3_path, s3_bucket):
    """
        Function to get prediction from table-localisation-ml lambda.
    :param s3_path: S3 path for image on which the prediction to get.
    :param s3_bucket: S3 bucket for image on which the prediction to get.
    :return: Output of table-localisation-ml lambda i.e. localisation prediction on the provided image.
    """
    event = {'key': s3_path,
             'bucket': s3_bucket,
             'region': 'ap-south-1'}

    return invoke_localisation_lambda(event, 'table-localisation-ml')


def get_model_output(df):
    """
        Function to get prediction on whole dataset.
    :param df: CSV with test set details.
    :return: None
    """
    for file_name in df['file_name']:
        s3_bucket = TEST_S3_BUCKET
        s3_path = f'{TEST_S3_PATH}/images/{file_name}'
        data = get_yolov5_pred(s3_path, s3_bucket)
        json_file_name = file_name[:file_name.rfind('.')] + '.json'
        with open(f'{LOCAL_DATA_DIR}/model_outputs/' + json_file_name, 'w') as f:
            json.dump(data, f)

        s3_cp(f'{LABELS_S3_PATH}/{file_name[:-4]}.json', f'{LOCAL_DATA_DIR}/labels/{file_name[:-4]}.json')


def get_best_result():
    """
        Download best result from s3.
    :return: best run result.
    """
    s3_cp(BEST_RESULT_S3_PATH, f'{LOCAL_DATA_DIR}/best_result.json')
    with open(f'{LOCAL_DATA_DIR}/best_result.json', 'r') as f:
        best_result = json.loads(f.read())
    return best_result


def get_comparison(best_result, model_result):
    """
        Convert score result to email output format.
    :param best_result: Best run result.
    :param model_result: Current model result.
    :return: Email output.
    """
    output = ''
    for thresh_iou in model_result.keys():
        output += f'For Thresh IOU = {thresh_iou} <br>'
        for score in model_result[thresh_iou].keys():
            output += f'{score} for best_model={best_result[thresh_iou][score]:.4f} \t current_model={model_result[thresh_iou][score]:.4f} <br>'
        output += '<br>'
    return output


def is_new_model_better(best_result, model_result):
    """
        Check if current model is better than best model.
    :param best_result: Best run result.
    :param model_result: Current model result.
    :return:
    """
    cnt = 0
    score = 0
    for thresh_iou in model_result.keys():
        cnt += 1
        if model_result[thresh_iou]['Column Seprators F1 Score'] > best_result[thresh_iou]['Column Seprators F1 Score']:
            score += 1
    if cnt > score // 2: return True
    return False


def inference():
    df = pd.read_csv(f'{LOCAL_DATA_DIR}/test_set_v1.csv')  # Read test data csv
    os.makedirs(f'{LOCAL_DATA_DIR}/labels/', exist_ok=True)
    os.makedirs(f'{LOCAL_DATA_DIR}/model_outputs/', exist_ok=True)

    get_model_output(df)  # Get test set predictions for current model.

    real_path = f'{LOCAL_DATA_DIR}/labels/'
    pred_path = f'{LOCAL_DATA_DIR}/model_outputs/'
    thresh_iou = [0.5]
    model_result = get_score(df, real_path, pred_path, thresh_iou)  # Get score for current model's prediction
    with open(f'{LOCAL_DATA_DIR}/model_result.json', 'w') as f:  # Save current model's score
        json.dump(model_result, f)
    best_result = get_best_result()  # Get best model score
    compare_result = get_comparison(best_result,
                                     model_result)  # Check if current model score is better than best model score
    if is_new_model_better(best_result,
                           model_result):  # If current model score is better than best model score then replace best model score with current model score
        compare_result += "<br><b> Since new model's F1 score is better. Updated best_model metrics with new_model metrics.</b><br>"
        s3_cp(f'{LOCAL_DATA_DIR}/model_result.json', BEST_RESULT_S3_PATH)

    bucket_result, bucket_output = get_bucket_analysis(df, real_path, pred_path,
                                                       thresh_iou)  # Get bucket analysis for current model
    final_output = compare_result + '<br>' + bucket_output  # Combine all result to be sent on mail
    call_email_lambda(final_output, 'Table Localisation')  # Send all the results by mail

    return None
