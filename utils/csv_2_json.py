import pandas as pd
import json
import random
import string

def generate_random_id(length=6):
    """Generate a random alphanumeric ID."""
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))

def convert_parquet_to_json(input_file, num_lines, output_file):
    # Read data from Parquet file into a DataFrame
    df = pd.read_parquet(input_file)

    # Take the first 'num_lines' rows from the DataFrame
    df_subset = df.head(num_lines)

    # Generate a random ID for the JSON data
    json_id = generate_random_id()

    # Create the dictionary for JSON
    json_data = {
        "id": json_id,
        "rows": df_subset.values.tolist(),
        "columns": df_subset.columns.tolist()
    }

    # Convert the dictionary to JSON string
    json_string = json.dumps(json_data, indent=2)

    # Save the JSON string to the output file
    with open(output_file, "w") as f:
        f.write(json_string)

if __name__ == "__main__":
    # Input Parquet file name
    input_parquet_file = "/mnt/d/MLops/Competition/mlops-mara-sample-public/data/raw_data/phase-2/prob-2/raw_train.parquet"

    # Number of lines to convert to JSON
    num_lines_to_convert = 50

    # Output JSON file name
    output_json_file = "/mnt/d/MLops/Competition/mlops-mara-sample-public/data/curl/phase-2/prob-2/payload-1.json"

    # Call the function to convert Parquet to JSON
    convert_parquet_to_json(input_parquet_file, num_lines_to_convert, output_json_file)
