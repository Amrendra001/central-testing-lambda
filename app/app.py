from table_localisation.inference import inference

def handler(event, context):

    if event['task'] == 'table_localisation':
        return inference()

    return None