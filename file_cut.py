import json
import sys
from transformers import BertTokenizer


def read_file(file_path, num_chars=0):
    """Read the specified number of characters from a file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        if num_chars > 0:
            return file.read(num_chars)
        else:
            return file.read()


def escape_special_characters(text):
    """Escape special characters for JSON."""
    return json.dumps(text)[1:-1]  # Remove the surrounding double quotes added by json.dumps


def write_file(file_path, content):
    """Write the content to a file."""
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)


def count_ml_tokens(text):
    """Count the number of machine learning tokens using a BERT tokenizer."""
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encoding = tokenizer.encode(text)
    return len(encoding)


def process_text_file(input_path, output_path_template, num_chars=0):
    """Process text file: read, escape characters, and write to new file."""
    # Read the content of the input file
    original_text = read_file(input_path, num_chars)

    # Escape special characters
    processed_text = escape_special_characters(original_text)

    # Get the size of the processed text
    text_size = len(processed_text)

    # Calculate the number of word tokens
    num_word_tokens = len(processed_text.split())

    # Calculate the number of machine learning tokens
    num_ml_tokens = count_ml_tokens(processed_text)

    # Create the output file path with the size, number of word tokens, and number of ML tokens in the name
    output_path = output_path_template.format(size=text_size, words=num_word_tokens, ml_tokens=num_ml_tokens)

    # Write the processed text to the new file
    write_file(output_path, processed_text)

    print(f"Processed text saved to: {output_path}")


# Example usage
if __name__ == "__main__":
    input_file_path = 'input.txt'  # Change to your input file path
    output_file_template = 'output_{size}_{words}_{ml_tokens}.txt'  # Change to your desired output file naming pattern

    # Check if num_chars is provided as a command line argument
    if len(sys.argv) > 1:
        num_chars = int(sys.argv[1])
    else:
        num_chars = 0

    process_text_file(input_file_path, output_file_template, num_chars)

    # Example usage: python file_cut.py 100