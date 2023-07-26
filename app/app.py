from table_localisation.inference import inference
from information_extraction.inference import inference_2

def handler(event, context):

    if event['task'] == 'table_localisation':
        return inference()
    if event['task'] == 'info_extraction':
        return inference_2()

    return None