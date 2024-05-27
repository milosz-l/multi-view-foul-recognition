import os
from SoccerNet.Evaluation.MV_FoulRecognition import evaluate

# Define the path to the directory containing the prediction files
predictions_dir = '/net/tscratch/people/plgmiloszl/outputs'
annotations_path = '/net/tscratch/people/plgmiloszl/data/Test/annotations.json'

# Iterate over each file in the directory
for filename in os.listdir(predictions_dir):
    if filename.startswith('predicitions_test_'):
        file_path = os.path.join(predictions_dir, filename)
        
        # Evaluate the prediction file
        results = evaluate(annotations_path, file_path)
        
        # Print the results
        print(f"Results for {filename}:")
        print(results)