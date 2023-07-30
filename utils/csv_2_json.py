import os
import pandas as pd
import json
import random
import string

def generate_random_id(length=6):
    """Generate a random alphanumeric ID."""
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))

def convert_parquet_to_json(input_file, num_lines, output_file, index):
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

def convert_all_parquet_files_to_json(input_dir, num_lines, output_dir):
    # Get the list of all files in the input directory
    files = os.listdir(input_dir)

    # Initialize the index variable
    index = 0

    # Loop through each file
    for file in files:
        # Check if the file is a Parquet file
        if file.endswith(".parquet"):
            # Create the input file path
            input_file_path = os.path.join(input_dir, file)

            # Generate the output JSON file name
            output_json_file = f"payload-{index}.json"
            output_json_file_path = os.path.join(output_dir, output_json_file)

            # Call the function to convert Parquet to JSON
            convert_parquet_to_json(input_file_path, num_lines, output_json_file_path, index)

            # Increment the index for the next file
            index += 1

if __name__ == "__main__":
    # Input directory containing Parquet files
    input_parquet_dir = "/home/ubuntu/MLOps-Marathon-2023/data/captured_data/phase-2/prob-1"

    # Number of lines to convert to JSON for each file
    num_lines_to_convert = 50

    # Output directory for JSON files
    output_json_dir = "/home/ubuntu/MLOps-Marathon-2023/data/curl/phase-2/prob-1"

    # Call the function to convert all Parquet files to JSON
    convert_all_parquet_files_to_json(input_parquet_dir, num_lines_to_convert, output_json_dir)