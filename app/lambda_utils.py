import json
import os
import boto3
import botocore.config


# Email utils
def call_email_lambda(status_string, task):
    lambda_input = {
        "toEmail": os.environ['EMAIL_RECIPIENT'],
        "ccEmail": "",
        "bccEmail": "",
        "fileName": [],
        "fromEmail": "noreply@javis.co",
        "subject": f"Central Testing Lambda - {task}",
        "messageBody": status_string
    }
    print('Sending project creation Email :', lambda_input)
    invoke_lambda(os.environ['EMAIL_LAMBDA'], lambda_input, invocation_type='async')


def invoke_lambda(lambda_name, input_data, aws_region='ap-south-1', invocation_type='sync', max_retries=None, timeout=None):
    input_data = json.dumps(input_data)
    print(f'Invoking {lambda_name} lambda')
    try:
        config = botocore.config.Config(
            read_timeout=timeout if timeout is not None else 900,
            retries={'max_attempts': max_retries if max_retries is not None else 0}
        )
        lambda_client = boto3.client('lambda', region_name=aws_region, config=config)

        invocation_type = 'Event' if invocation_type == 'async' else 'RequestResponse'
        response = lambda_client.invoke(FunctionName=lambda_name,
                                        InvocationType=invocation_type,
                                        Payload=input_data)
        if response['StatusCode'] in range(200, 300):
            response = response['Payload'].read()
            if invocation_type != 'Event':
                response = json.loads(response)
            else:
                response = ""
    except Exception as e:
        raise Exception(f'{lambda_name} lambda Invocation Failed: ' + str(e))

    return response
