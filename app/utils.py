import json
import boto3
import os

def invoke_localisation_lambda(input_data, name):
    input_data = json.dumps(input_data)

    data = {}

    try:
        lambda_client = boto3.client('lambda', region_name='ap-south-1')
        response = lambda_client.invoke(FunctionName=name, Payload=input_data)
        if response['StatusCode'] in range(200, 300):
            response = response['Payload'].read()
            data = json.loads(response)
    except Exception as e:
        raise Exception(
            'Table Localisation Lambda Invocation Failed: ' + str(e))

    return data


def s3_cp(source, destination):
    sync_command = f'aws s3 cp "{source}" "{destination}"'
    os.system(sync_command)


def get_best_result(best_result_s3_path, local_data_dir):
    """
        Download best result from s3.
    :return: best run result.
    """
    s3_cp(best_result_s3_path, f'{local_data_dir}/best_result.json')
    with open(f'{local_data_dir}/best_result.json', 'r') as f:
        best_result = json.loads(f.read())
    return best_result