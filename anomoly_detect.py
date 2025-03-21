import boto3
import datetime
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings("ignore")

# AWS Credentials (replace with your own credentials)
AWS_ACCESS_KEY_ID = ''
AWS_SECRET_ACCESS_KEY = ''
AWS_REGION = 'us-east-1' 

# CloudWatch client 
cloudwatch = boto3.client(
    'cloudwatch',
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

# Instance IDs and metrics to fetch ( Replace with your when running)
instance_ids = ['i-015ac772714759150', 'i-0ff0fef257d31bf6e']
metrics = [
    'CPUUtilization',
    'DiskReadOps',
    'DiskWriteOps',
    'DiskReadBytes',
    'DiskWriteBytes',
    'NetworkIn',
    'NetworkOut',
    'NetworkPacketsIn',
    'NetworkPacketsOut'
]

# Set time window for past 1 hour (UTC)
end_time = datetime.datetime.utcnow()
start_time = end_time - datetime.timedelta(hours=1)
period = 300  # 5 minutes period

results = []

# Loop over each instance and metric, then compute average values
for instance_id in instance_ids:
    instance_data = {'InstanceId': instance_id}
    for metric in metrics:
        response = cloudwatch.get_metric_statistics(
            Namespace='AWS/EC2',
            MetricName=metric,
            Dimensions=[{'Name': 'InstanceId', 'Value': instance_id}],
            StartTime=start_time,
            EndTime=end_time,
            Period=period,
            Statistics=['Average']
        )
        datapoints = response.get('Datapoints', [])
        if datapoints:
            avg_value = sum(point['Average'] for point in datapoints) / len(datapoints)
        else:
            avg_value = None
        instance_data[metric] = avg_value
    results.append(instance_data)

# Create DataFrame from CloudWatch results
df = pd.DataFrame(results)
print("Fetched CloudWatch Metrics:")
print(df)

# ------------------------------
# Load your saved model and scaler
# ------------------------------
with open('anomoly_detection_Random_Forest.pkl', 'rb') as f:
    best_model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# ------------------------------
# Prepare feature data for prediction
# ------------------------------
feature_columns = [
    'CPUUtilization',
    'DiskReadOps',
    'DiskWriteOps',
    'DiskReadBytes',
    'DiskWriteBytes',
    'NetworkIn',
    'NetworkOut',
    'NetworkPacketsIn',
    'NetworkPacketsOut'
]

# Fill missing values (if any) with zeros or another strategy used during training
df_features = df[feature_columns].fillna(0)

# Scale the features (using the same scaler from training)
scaled_features = scaler.transform(df_features)

# ------------------------------
# Use the best model to predict anomalies
# ------------------------------
predictions = best_model.predict(scaled_features)

# ------------------------------
# Add predictions to the DataFrame and print messages
# ------------------------------
# Create a new column "Prediction" with human-friendly messages
df['Prediction'] = ['anomaly' if pred == 1 else 'normal' for pred in predictions]

for idx, pred in enumerate(predictions):
    instance_id = df.loc[idx, 'InstanceId']
    if pred == 1:
        print(f"Instance {instance_id} is an anomaly.")
    else:
        print(f"Instance {instance_id} is normal.")

# ------------------------------
# Save the DataFrame with predictions to a CSV file
# ------------------------------
csv_file = 'anomoly_prediction_ec2.csv'
df.to_csv(csv_file, index=False)
print(f"Metrics have been written to {csv_file}")
