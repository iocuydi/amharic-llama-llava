import os
import csv
import torch
import sys 
from threading import Thread
from queue import Queue
from torch import Tensor
from torch.nn.functional import pad as pad_tensor

PATH_TO_SEAMLESS_COMMUNICATION = ''
PATH_TO_SEAMLESS_COMMUNICATION_SRC = '' + '/src'

sys.path.append(PATH_TO_SEAMLESS_COMMUNICATION)
sys.path.append(PATH_TO_SEAMLESS_COMMUNICATION_SRC)

from seamless_communication.models.unity import (
    load_unity_text_tokenizer,
)
from seamless_communication.models.inference import Translator
from typing import List, Any

device = torch.device("cuda:0")
device_cpu = torch.device('cpu')
dtype = torch.float16
text_tokenizer = load_unity_text_tokenizer("seamlessM4T_large")
token_encoder = text_tokenizer.create_encoder(
                task="translation", lang='eng', mode="source", device=device_cpu
            )
translator = Translator("seamlessM4T_large", "vocoder_36langs", device, dtype)

def contains_letter(s):
    """Check if the string contains any letter."""
    return any(c.isalpha() for c in s)

def batch_tensors(tensors: List[Tensor], pad_value: Any) -> Tensor:
    padding_size = max(tensor.shape[0] for tensor in tensors)
    dims = len(tensors[0].shape)
    padded_tensors = []
    for tensor in tensors:
        padding = [0] * 2 * dims
        padding[-1] = padding_size - tensor.shape[0]
        padded_tensors.append(pad_tensor(tensor, padding, "constant", pad_value))
    return torch.stack([tensor for tensor in padded_tensors], dim=0)

def reader_thread(filename, input_queue):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:

            encoded_row = [item for item in row[:-1]]

            input_queue.put({'translate_targ': encoded_row, 'original': row, 'end': False})  # Exclude the last column
    input_queue.put({'end': True})
def translator_thread(input_queue, output_queue):
    while True:
        batch_to_translate = input_queue.get()  # Dequeue the batch

        if batch_to_translate['end']:
            output_queue.put({'end': True})
            break

        translated_text, _, _ = translator.predict(
            batch_to_translate['translate_targ'],
            't2tt',
            'amh',
            src_lang='eng',
            ngram_filtering=False,
        )
        output_queue.put({'translate_out': translated_text, 'original': batch_to_translate['original'], 'end': False})  # Enqueue the translated text

def writer_thread(input_queue, output_queue, filename):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        while True:
            next_output = input_queue.get()  # Dequeue the translated text

            if next_output['end']:
                break

            translated_text = next_output['translate_out']
            original_row = next_output['original']
            #original_row = output_queue.get()  # Dequeue the original row
            #print('here is a tx')
            #print(translated_text)
            output_final = []
            for idx, item in enumerate(translated_text):
                if contains_letter(original_row[idx]):
                    output_final.append(item)
                else:
                    output_final.append(original_row[idx])
            #print(original_row)
            #print('what is the total')
            print(output_final + [original_row[-1]])
            writer.writerow(output_final + [original_row[-1]])  # Append the last column from the original row

def translate_csv_files(input_dir, output_dir):
    for filename in os.listdir(input_dir):
        input_queue = Queue()
        output_queue = Queue()

        input_file = os.path.join(input_dir, filename)
        output_file = os.path.join(output_dir, filename)
        print(input_file)
        if os.path.exists(output_file):
            print('existing skipping')
            continue

        reader = Thread(target=reader_thread, args=(input_file, input_queue))
        translator = Thread(target=translator_thread, args=(input_queue, output_queue))
        writer = Thread(target=writer_thread, args=(output_queue, input_queue, output_file))

        reader.start()
        translator.start()
        writer.start()

        reader.join()
        translator.join()
        writer.join()

if __name__ == "__main__":
    translate_csv_files('eng', 'translated')
