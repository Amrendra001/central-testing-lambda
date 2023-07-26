from dotenv import load_dotenv
load_dotenv()
import sys
sys.path.append('app/')
from app import handler

if __name__ == '__main__':
    event = {
        # 'task': 'table_localisation',
        'task': 'info_extraction'
    }
    print(handler(event, 'context'))