import pandas as pd
import json
import os

# Load the train.csv file
train_file_path = 'datasets/HKTK/annotations/test.csv'
train_df = pd.read_csv(train_file_path, sep=' ', header=None, names=['file_path', 'verb_class'])

# Load the label_class_mapping_action.json file
label_mapping_file_path = 'datasets/HKTK/annotations/label_class_mapping_action.json'
with open(label_mapping_file_path, 'r') as f:
    label_mapping = json.load(f)

# Reverse the label mapping to get verb and noun from class ID
reverse_label_mapping = {v: k for k, v in label_mapping.items()}

# Extract video_id, verb, and noun from the train_df
train_df['video_id'] = train_df['file_path'].apply(lambda x: os.path.basename(x).replace('.mp4', ''))
train_df['verb'] = train_df['verb_class'].apply(lambda x: reverse_label_mapping[f'class{x}'])
train_df['noun'] = train_df['verb'].apply(lambda x: x.split('】')[0].replace('【', ''))
train_df['noun_class'] = train_df['verb_class'].apply(lambda x: 0 if 'Outdoor unit' in reverse_label_mapping[f'class{x}'] else 1)
train_df['verb'] = train_df['verb'].apply(lambda x: x.split('】')[1])

# Generate additional columns
train_df['narration_id'] = ''
train_df['participant_id'] = ''
train_df['narration_timestamp'] = ''
train_df['start_timestamp'] = '0:00:00'
train_df['stop_timestamp'] = '0:00:06'
train_df['start_frame'] = 0
train_df['stop_frame'] = 180  # 6 seconds * 30 fps

# 修正: ナレーションを [{noun}] {verb} の形式に変更
train_df['narration'] = train_df.apply(lambda row: f"[{row['noun']}] {row['verb']}.", axis=1)

train_df['all_nouns'] = [[] for _ in range(len(train_df))]
train_df['all_noun_classes'] = [[] for _ in range(len(train_df))]

# Reorder the columns according to the specification
final_columns = [
    'narration_id', 'participant_id', 'video_id', 'narration_timestamp',
    'start_timestamp', 'stop_timestamp', 'start_frame', 'stop_frame',
    'narration', 'verb', 'verb_class', 'noun', 'noun_class', 'all_nouns', 'all_noun_classes'
]
train_df = train_df[final_columns]

# Save the transformed dataframe to a new CSV file
output_file_path = 'datasets/HKTK/annotations/HKTK-action-test.csv'
train_df.to_csv(output_file_path, index=False)
