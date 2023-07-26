import json
import boto3

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