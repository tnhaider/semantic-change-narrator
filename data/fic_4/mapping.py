import sys
import csv

# Define the label mapping
label_mapping = {
    'a': 'non-fic',
    'b': 'non-fic',
    'c': 'fic',
    'd': 'fic'
}

# Input and output file paths
input_file = sys.argv[1]  # Replace with your input TSV file path
output_file = sys.argv[2]  # Replace with your desired output TSV file path

def transform_labels(input_file, output_file):
    with open(input_file, mode='r', encoding='utf-8') as infile, open(output_file, mode='w', encoding='utf-8', newline='') as outfile:
        reader = csv.reader(infile, delimiter='\t')  # Reading TSV file
        writer = csv.writer(outfile, delimiter='\t')  # Writing TSV file

        # Write header
        header = next(reader)
        writer.writerow(header)

        for row in reader:
            if len(row) == 2:  # Ensure there are exactly two columns
                text, label = row
                new_label = label_mapping.get(label, label)  # Default to the original label if not found in mapping
                writer.writerow([text, new_label])

if __name__ == "__main__":
    transform_labels(input_file, output_file)
