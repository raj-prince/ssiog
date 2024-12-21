#!/usr/bin/env python3
# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd
import gcsfs

def analyze_metrics(bucket_path, timestamp_filter=True):
    """
    Analyzes metrics from CSV files in a Google Cloud Storage bucket.

    Args:
        bucket_path: The path to the bucket containing CSV files.  Should be a GCS path, e.g., "gs://my-bucket/path/to/files/*.csv"

    Returns:
        A pandas DataFrame containing the combined latency data, or None if no files are found. Also, timebased filtering which selects
        common entry among all the CSV files if timestamp_filter is set to True.
    """
    try:
        # Use gcsfs to access the bucket.  Install with: pip install gcsfs
        fs = gcsfs.GCSFileSystem()
        
        # Find all CSV files in the bucket path using glob-like pattern matching.
        csv_files = list(fs.glob(bucket_path))
        if not csv_files:
            return None
        
        start_timestamps = []
        end_timestamps = []        
        all_data = []
        for file in csv_files:
            with fs.open(file, 'r') as f:
                df = pd.read_csv(f)
                if not df.empty:
                    start_timestamps.append(df['timestamp'].iloc[0])
                    end_timestamps.append(df['timestamp'].iloc[-1])
                all_data.append(df)

        combined_df = pd.concat(all_data)
        
        if not start_timestamps or not end_timestamps:
            return None

        min_timestamp = max(start_timestamps)
        max_timestamp = min(end_timestamps)
        
        # Filter which is not recorded b/w min_timestamp and max_timestamp
        if timestamp_filter:
            combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'], unit='s')
            combined_df = combined_df[(combined_df['timestamp'] >= pd.to_datetime(min_timestamp, unit='s')) & (combined_df['timestamp'] <= pd.to_datetime(max_timestamp, unit='s'))]
        
        if combined_df.empty:
            return None
    
        return combined_df
    
    except Exception as e:
        return None


# Create a main executor which provides a hardcoded path to analyze the metrics create a main method instead
def main():
    bucket_path = "gs://princer-ssiog-metrics-bkt/test_0_1_0-1/*/*.csv"
    result_df = analyze_metrics(bucket_path, timestamp_filter=True)
    if result_df is not None:
        print(result_df['sample_lat'].describe(percentiles=[0.1, 0.25, 0.5, 0.9, 0.99, 0.999]))

if __name__ == "__main__":
    main()

