from pathlib import Path
import pickle

from src.exception import CustomException

def save_object(obj, path: Path|str):
    try:
        path = Path(path)
        path.parent.mkdir(exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
            f.close()
            
    except Exception as e:
        raise CustomException(e)