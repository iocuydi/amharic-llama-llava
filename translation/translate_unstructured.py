import os
import torch
import json 
import sys 

from threading import Thread

from fairseq2.data import Collater
from nltk.tokenize import sent_tokenize

from collections import defaultdict
from queue import Queue, PriorityQueue

PATH_TO_SEAMLESS_COMMUNICATION = ''
PATH_TO_SEAMLESS_COMMUNICATION_SRC = '' + '/src'

sys.path.append(PATH_TO_SEAMLESS_COMMUNICATION)
sys.path.append(PATH_TO_SEAMLESS_COMMUNICATION_SRC)

from seamless_communication.models.unity import (
    load_unity_text_tokenizer,
)
from seamless_communication.models.inference import Translator

device = torch.device("cuda:0")
device_cpu = torch.device('cpu')
dtype = torch.float16
text_tokenizer = load_unity_text_tokenizer("seamlessM4T_large")
token_encoder = text_tokenizer.create_encoder(
                task="translation", lang='eng', mode="source", device=device_cpu
            )

collate = Collater(
            pad_idx=text_tokenizer.vocab_info.pad_idx, pad_to_multiple=2
        )
        
translator = Translator("seamlessM4T_large", "vocoder_36langs", device, dtype)


# Helper function to find newlines in the text
def find_newlines_in_text(text, tokenized_sentences):
    newlines = set()
    position = 0
    for index, sentence in enumerate(tokenized_sentences):
        found_at = text.find(sentence, position)
        if found_at == -1:
            continue
        if '\n' in text[position:found_at]:
            newlines.add(index)
        position = found_at + len(sentence)
    return newlines

# Function to log byte index to a file
def log_byte_index(byte_index, log_file="byte_index.log"):
    with open(log_file, "w") as f:
        f.write(str(byte_index))

def handle_long_sentences(sentence, max_sentence_length):
    #print('attempt shortening a senetence')
    #print(sentence)
    new_encodings = []
    enc_lens = []

    # Split the sentence on newlines
    parts = sentence.split('\n')

    for part in parts:
        # Only include parts that are shorter than max_sentence_length and longer than 0
        encoded_part = token_encoder(part)
        if 0 < len(encoded_part) <= max_sentence_length:
            new_encodings.append(part)
            enc_lens.append(len(encoded_part))

        elif len(part) > max_sentence_length:
            #pass
            print('EXCLUDE: {}'.format(part))
    

    return new_encodings, enc_lens

# Thread 1: Reading and Tokenization
def reader_thread(filename, start_byte, input_queue, batch_size, max_sentence_length):
    with open(filename, "r") as f:
        f.seek(start_byte)
        f.readline()
        global_index = 0  # Initialize a global index
        token_buffer = []
        index_buffer = []  # To track the indices of the sentences
        newline_list = []  # A list of booleans to track newlines
        enc_len_buffer = [] # A list of encoded sentence lengths
        current_byte = start_byte

        while True:
            initial_byte = f.tell()  # Save the byte index before reading the line
            line = f.readline()
            final_byte = f.tell()  # Save the byte index after reading the line
            
            if not line:
                break

            current_byte += (final_byte - initial_byte)  # Update the byte index

            document = json.loads(line)
            #if document['meta']['language'] != 'en':
            #    continue
            
            sentences = sent_tokenize(document['text'])
            newlines_in_doc = find_newlines_in_text(document['text'], sentences)


            for idx, sentence in enumerate(sentences):
                
                encoded_sentence = token_encoder(sentence)

                if len(encoded_sentence) > max_sentence_length:
                    new_encodings, enc_lens = handle_long_sentences(sentence, max_sentence_length)
                    for new_idx, new_encoding in enumerate(new_encodings):

                        token_buffer.append(new_encoding)
                        # we always need a newline for these

                        newline_list.append(True)
                        index_buffer.append(global_index)
                        enc_len_buffer.append(enc_lens[new_idx])
                        global_index += 1

                        if len(token_buffer) >= batch_size:

                            input_queue.put({'batch': token_buffer.copy(), 'newlines': newline_list.copy(), 
                                'enclen': enc_len_buffer.copy(), 'indices': index_buffer.copy(), 'byte': current_byte})
                            token_buffer = []
                            index_buffer = []
                            newline_list = []
                            enc_len_buffer = []
                            


                    continue

                # normal case where max length not exceeded
                token_buffer.append(sentence)
                index_buffer.append(global_index)
                newline_list.append(idx in newlines_in_doc)
                enc_len_buffer.append(len(encoded_sentence))
                global_index += 1


                if len(token_buffer) >= batch_size:

                    input_queue.put({'batch': token_buffer.copy(), 'newlines': newline_list.copy(), 
                        'enclen': enc_len_buffer.copy(), 'indices': index_buffer.copy(), 'byte': current_byte})
                    token_buffer = []
                    index_buffer = []
                    newline_list = []
                    enc_len_buffer = []
                    
                    #newline_set = set()
            # always add newline indicator after the end of a document
            if len(token_buffer) > 0:
                newline_list[-1] = True


def bucket_thread(read_to_bucket_queue, input_queue, batch_size, fallback_batch_size):
    bucket_map = defaultdict(list)
    
    def get_bucket_key(length, bucket_size=50):
        rb = (length // bucket_size) * bucket_size
        return min(rb, 100)
    
    while True:
        data = read_to_bucket_queue.get()
        for encoded_sentence, index, newline, enc_len in zip(data['batch'], data['indices'], data['newlines'], data['enclen']):
            bucket_key = get_bucket_key(enc_len)#get_bucket_key(len(encoded_sentence))
            bucket_map[bucket_key].append({'sentence': encoded_sentence, 'index': index, 'newline': newline, 'byte': data['byte']})

            evaluation_batch_size = batch_size if bucket_key < 100 else fallback_batch_size

            if len(bucket_map[bucket_key]) >= evaluation_batch_size:
                batch = bucket_map[bucket_key]
                sentences = [item['sentence'] for item in batch]
                newlines = [item['newline'] for item in batch]
                indices = [item['index'] for item in batch]
                ref_bytes = [item['byte'] for item in batch]

                min_index_in_batch = min(item['index'] for item in bucket_map[bucket_key])

                input_queue.put((min_index_in_batch, {'batch': sentences, 'byte': ref_bytes, 'indices': indices, 'newlines': newlines}))

                bucket_map[bucket_key] = []
            
# Thread 2: Translation
def translator_thread(input_queue, output_queue):
    while True:
        idx, batch_to_translate = input_queue.get()  # Dequeue the batch
        print('min idx in translation batch {}'.format(idx))
        translated_text, wav, sr = translator.predict(
                batch_to_translate['batch'],#args.input,
                't2tt',
                'amh',
                src_lang='eng',
                ngram_filtering=False,
            )

        output_queue.put({'batch': translated_text, 'newlines': batch_to_translate['newlines'], 
                'indices': batch_to_translate['indices'],
                'byte': batch_to_translate['byte']})  # Enqueue the translated text


# Maintains sorted order of the output lines and sends them to writer thread as appropriate
def writer_prep_thread(output_queue, write_queue):
    heap = PriorityQueue()
    last_written_index = -1
    
    while True:
        if not heap.empty() and heap.queue[0][0] == last_written_index + 1:
            _, item = heap.get()
            write_queue.put(item)

            last_written_index = item['index']
        else:
            print('waiting on {}, size q = {}'.format(last_written_index + 1, heap.qsize()))
            out_batch = output_queue.get()
            for idx, sentence_item in enumerate(out_batch['batch']):
                text = out_batch['batch'][idx]
                index = out_batch['indices'][idx]
                newline = out_batch['newlines'][idx]
                byte = out_batch['byte'][idx]
                item = {
                    'batch': [text],
                    'index': index,
                    'newlines': [newline],
                    'byte': byte
                }
                heap.put((index, item))


# Thread 3: Writing to Disk
def writer_thread(output_queue, file_prefix):
    file_count = 0
    current_file = open(f"{file_prefix}_{file_count}.txt", "w")
    current_size = 0
    last_byte = None
    while True:
        out_batch = output_queue.get()  # Dequeue the translated text

        translated_base = out_batch['batch']
        newlines = out_batch['newlines']
        translated_text = " "

        for idx, text in enumerate(translated_base):
            translated_text += str(text)
            if newlines[idx]:
                translated_text += '\n'
            else:
                translated_text += '.'

        current_size += len(translated_text)
        if current_size > 20 * 1024 * 1024:  # 50 MB
            current_file.close()
            file_count += 1
            current_file = open(f"{file_prefix}_{file_count}.txt", "w")
            current_size = len(translated_text)

        current_file.write(translated_text)

        if out_batch['byte'] != last_byte:
            log_byte_index(out_batch['byte'])
            last_byte = out_batch['byte']


if __name__ == "__main__":
    # Initialize
    read_to_bucket_queue = Queue(maxsize=256)
    input_queue = PriorityQueue(maxsize=256)  # Limits the queue size
    output_queue = Queue()
    write_queue = Queue(maxsize=256)

    if not os.path.exists('outputs'):
        os.mkdir('outputs')

    filename = "source.jsonl"
    start_byte = 38940910049#35156513209#28037468357# original 27805515516  # Replace with the actual starting byte
    batch_size = 512  # Replace with the actual batch size
    fallback_batch_size = 256 # batch size to use for oversized batches
    file_prefix = "outputs/output_txt_file"
    max_sentence_length = 236
    print('starting at byte index {}'.format(start_byte))
    # Create threads
    t1 = Thread(target=reader_thread, args=(filename, start_byte, read_to_bucket_queue, batch_size, max_sentence_length))
    t_bucket = Thread(target=bucket_thread, args=(read_to_bucket_queue, input_queue, batch_size, fallback_batch_size))
    t2 = Thread(target=translator_thread, args=(input_queue, output_queue))
    t_prep = Thread(target=writer_prep_thread, args=(output_queue, write_queue))
    t3 = Thread(target=writer_thread, args=(write_queue, file_prefix))

    # Start threads
    t1.start()
    t_bucket.start() 
    t2.start()
    t_prep.start()
    t3.start()

    # Wait for threads to complete
    t1.join()
    t_bucket.join()
    t2.join()
    t_prep.join()
    t3.join()
