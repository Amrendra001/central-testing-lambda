from table_localisation.inference import get_score






def handler(event, context):

    if event['task'] == 'table_localisation':
        return get_score()

    return None