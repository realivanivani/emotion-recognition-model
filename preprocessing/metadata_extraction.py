import pandas as pd
import json

def load_annotations(annotation_path):
    """Load AffectNet annotations from file"""
    if annotation_path.endswith('.csv'):
        return pd.read_csv(annotation_path)
    elif annotation_path.endswith('.json'):
        with open(annotation_path) as f:
            return json.load(f)
    else:
        raise ValueError("Unsupported annotation format")

def parse_metadata(annotation_df):
    """Extract and clean metadata"""
    metadata = []
    
    for _, row in annotation_df.iterrows():
        item = {
            'image_path': row['image_path'],
            'expression': int(row['expression']),
            'valence': float(row['valence']),
            'arousal': float(row['arousal']),
            'landmarks': parse_landmarks(row['facial_landmarks'])
        }
        metadata.append(item)
    
    return metadata

def parse_landmarks(landmark_data):
    """Convert landmark data to list of coordinates"""
    # Implementation depends on how landmarks are stored
    pass
