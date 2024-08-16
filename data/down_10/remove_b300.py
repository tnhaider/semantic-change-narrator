import pandas as pd

def filter_and_write(input_file, output_file, label_to_keep='b', num_instances=300):
    # Read the input file into a DataFrame
    df = pd.read_csv(input_file, sep='\t')
    
    # Filter to keep only instances where label is not 'b'
    df_not_b = df[df['label'] != label_to_keep]
    
    # Sample 300 instances where label is 'b'
    df_b = df[df['label'] == label_to_keep].sample(n=num_instances, random_state=1)
    
    # Concatenate the two DataFrames
    df_final = pd.concat([df_b, df_not_b])
    
    # Write the concatenated DataFrame to the output file
    df_final.to_csv(output_file, sep='\t', index=False)

if __name__ == "__main__":
    # Input and output file paths
    input_file = 'train.tsv'
    output_file = 'train_d.tsv'
    
    # Label to keep and number of instances to keep
    label_to_keep = 'b'
    num_instances = 300
    
    # Call the function to filter and write
    filter_and_write(input_file, output_file, label_to_keep, num_instances)
