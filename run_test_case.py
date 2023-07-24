if __name__ == '__main__':
    import sys
    sys.path.append('app/')
    from app import handler

    event = {
        'task': 'table_localisation',
    }

    print(handler(event, 'context'))