import re 
import torch
import json 
import time
import sys 
from tqdm import tqdm
from fairseq2.data import Collater



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
            print(sentence)
            WARNING = "WARNING @@@@@ WARNING @@@@ EXCLUSION @@@@ INCLUDED @@@@ 2394832597397549568"
            new_encodings.append(WARNING)
            enc_lens.append(len(token_encoder(WARNING)))
            #abf = input('@@@@@@')
    
    return new_encodings, enc_lens

def do_nonbatch_translation(translator, text, max_sentence_length):
    
    if len(text) == 0:
        return text

    token_encoded = token_encoder(text)
    if len(token_encoded) > max_sentence_length:
        new_texts, new_len = handle_long_sentences(text, max_sentence_length)
        sublists = [do_nonbatch_translation(translator, item, max_sentence_length) for item in new_texts]

        sublists = "\n".join([str(item) for item in sublists])

        return sublists

    translated_text, wav, sr = translator.predict(
            [text],#args.input,
            't2tt',
            'amh',
            src_lang='eng',
            ngram_filtering=False,
        )

    return " ".join([str(item) for item in translated_text])

def translate_non_code_snippets_and_stitch_back(text, translate_client):

    error_suspicious = False
    if not ('```' in text or '`' in text):

        x = [do_nonbatch_translation(translate_client, text, 250)]

        return x

    delimiters = ['```', '`']  # Prefer longer delimiter
    delimiters.sort(key=len, reverse=True)  # Sorting delimiters by length in descending order

    translated_length = 0

    # first, split the text into code snippets and non-code snippets
    non_code_snippets = []
    code_snippets = []
    code_delimiters = []

    while len(text) > 0:
        # find the next code snippet
        next_code_snippet_start = len(text)
        next_delimiter = None
        for delimiter in delimiters:
            temp_start = text.find(delimiter)
            if temp_start != -1 and temp_start < next_code_snippet_start:
                next_code_snippet_start = temp_start
                next_delimiter = delimiter

        if next_delimiter is None:
            # no more code snippets
            non_code_snippets.append(text)
            text = ''
        else:
            # there is a code snippet
            non_code_snippets.append(text[:next_code_snippet_start])
            text = text[next_code_snippet_start + len(next_delimiter):]
            next_code_snippet_end = text.find(next_delimiter)
            if next_code_snippet_end == -1:
                # there is no closing code snippet, treat the rest as code
                code_snippets.append(text)
                code_delimiters.append(next_delimiter)
                text = ''
                error_suspicious = True
            else:
                code_snippets.append(text[:next_code_snippet_end])
                code_delimiters.append(next_delimiter)
                text = text[next_code_snippet_end + len(next_delimiter):]

    
    # now, translate the non-code snippets
    translated_non_code_snippets = []
    for non_code_snippet in non_code_snippets:

        translated_non_code_snippets.append(do_nonbatch_translation(translate_client, non_code_snippet, 250))
        translated_length += len(non_code_snippet)

    # now, stitch the non-code snippets and code snippets back together
    translated_text = ''
    for i, non_code_snippet in enumerate(translated_non_code_snippets):
        translated_text += non_code_snippet
        if i < len(code_snippets):
            translated_text += code_delimiters[i] + code_snippets[i] + code_delimiters[i]
    if error_suspicious:
        print('WARNING!! ERROR !! !!')
        print(translated_text)
        print(text)
        assert not error_suspicious

    return translated_text

def assertImageBlip(base_filename):
    with open(base_filename, 'r') as infile:
        d = json.load(infile)
        for item in tqdm(d, desc='proc'):
            for idx, convo in enumerate(item['conversations']):
                if idx == 0 and convo['from'] == 'human':
                    assert convo['value'].startswith('<image>\n') or convo['value'].endswith('\n<image>')
            if not 'image' in item:
                abf = input('warning image is not in item {}'.format(item))


def contains_letter(s):
    return bool(re.search(r'[a-zA-Z]', s))

def extract_non_letter_bracketed_substrings(s):
    # Regular expression pattern
    pattern = r'\[[^\[\]a-zA-Z]*\]'

    # Find all matching substrings
    return re.findall(pattern, s)

def uniquely_key_item(item):
    convos = item['conversations']
    assert len(convos) > 0
    key = "".join([chat['value'] for chat in convos])
    key = key + '--' + str(item['id'])
    return key

def distribute_batches(batch):

    batch_for_processing = []
    idxs_for_reference = []

    for idx, item in enumerate(batch):
        if '.' in item and item != '.':
            new_splits = item.split('.')
            #abf = input(new_splits)

            for split in new_splits:
                if len(split) > 0:
                    batch_for_processing.append(split)
                    idxs_for_reference.append(idx)
        else:
            batch_for_processing.append(item)
            idxs_for_reference.append(idx)

    
    return batch_for_processing, idxs_for_reference

def reconstruct_processing(batch, idxs):
    
    processed_texts = []
    next_idx = 0

    while batch:
        upcoming_texts = []
        #abf = input(idxs)
        while idxs and idxs[0] == next_idx:
            upcoming_texts.append(batch.pop(0))
            idxs.pop(0)
        
        if len(upcoming_texts) == 0:
            processed_texts.append(upcoming_texts[0])
        else:
            processed_texts.append('።'.join(upcoming_texts))
        
        next_idx += 1
    
    return processed_texts


def prepareBlipLaion(base_filename, output_filename, process_thresh):
    with open(base_filename, 'r') as infile, open(output_filename, 'w', encoding='utf-8') as outfile:
        d = json.load(infile)
        batch = []
        metadata = []
        top_level_items = []
        for item in tqdm(d, desc='Translating blip laion cc file'):
            new_item = item.copy()
            old_convos = item['conversations']
            new_convos = []
            if not "image" in item:
                continue

            for convo in old_convos:
                metaitem = convo.copy()
                metaitem['text'] = '{}'
                if convo['from'] == 'human' and convo['value'].startswith('<image>\n'):
                    metaitem['text'] = '<image>\n{}'
                elif convo['from'] == 'human' and convo['value'].endswith('\n<image>'):
                    metaitem['text'] = '{}\n<image>'

                metaitem['id'] = uniquely_key_item(item)
                metadata.append(metaitem)
                batch.append(convo['value'].replace('\n<image>','').replace('<image>\n', ''))

            top_level_items.append(new_item)

            # accumulate more items to process before doing translation
            if len(batch) > process_thresh:
                
                real_batch, real_idxs = distribute_batches(batch)

                translated_text, wav, sr = translator.predict(
                    real_batch,#args.input,
                    't2tt',
                    'amh',
                    src_lang='eng',
                    ngram_filtering=False,
                )

                translated_text = reconstruct_processing([str(s) for s in translated_text], real_idxs)

                for idx, result in enumerate(translated_text):
                    if idx < len(metadata):
                        metadata[idx]['text'] = metadata[idx]['text'].format(result)

                        if not contains_letter(metadata[idx]['value']):
                            metadata[idx]['text'] = metadata[idx]['value']
                        
                        if metadata[idx]['value'] in {'A': 1, 'B': 1, 'C': 1, 'D': 1}:
                            metadata[idx]['text'] = metadata[idx]['value']
                        
                        bracketed_sections = extract_non_letter_bracketed_substrings(metadata[idx]['text'])
                        if len(bracketed_sections) > 0:
                            alternate_bracketed_sections = extract_non_letter_bracketed_substrings(metadata[idx]['value'])
                            if len(alternate_bracketed_sections) != len(bracketed_sections):
                                print(alternate_bracketed_sections)
                                print(bracketed_sections)
                                #print(metadata[idx])
                            else:
                                for section_idx, section in enumerate(bracketed_sections):
                                    metadata[idx]['text'] = metadata[idx]['text'].replace(section, alternate_bracketed_sections[section_idx])

                for overall_item in top_level_items:
                    new_item = overall_item.copy()
                    new_item['conversations'] = []
                    overall_item_key = uniquely_key_item(overall_item)
                    while metadata and metadata[0]['id'] == overall_item_key:
                        next_meta = metadata.pop(0)
                        del next_meta['id']
                        new_item['conversations'].append(next_meta)
                    
                    outfile.write(json.dumps(new_item) + "\n")
                top_level_items = []
                batch = []
                metadata = []

        if len(batch) > 0:
            real_batch, real_idxs = distribute_batches(batch)
            translated_text, wav, sr = translator.predict(
                    real_batch,#args.input,
                    't2tt',
                    'amh',
                    src_lang='eng',
                    ngram_filtering=False,
                )
            translated_text = reconstruct_processing([str(s) for s in translated_text], real_idxs)
            for idx, result in enumerate(translated_text):
                if idx < len(metadata):
                    metadata[idx]['text'] = metadata[idx]['text'].format(result)

                    if not contains_letter(metadata[idx]['value']):
                        metadata[idx]['text'] = metadata[idx]['value']
                    
                    if metadata[idx]['value'] in {'A': 1, 'B': 1, 'C': 1, 'D': 1}:
                        metadata[idx]['text'] = metadata[idx]['value']
                    
                    bracketed_sections = extract_non_letter_bracketed_substrings(metadata[idx]['text'])
                    if len(bracketed_sections) > 0:
                        alternate_bracketed_sections = extract_non_letter_bracketed_substrings(metadata[idx]['value'])
                        if len(alternate_bracketed_sections) == len(bracketed_sections):
                            for section_idx, section in enumerate(bracketed_sections):
                                metadata[idx]['text'] = metadata[idx]['text'].replace(section, alternate_bracketed_sections[section_idx])
                        
                

            for overall_item in top_level_items:
                new_item = overall_item.copy()
                new_item['conversations'] = []
                overall_item_key = uniquely_key_item(overall_item)
                while metadata and metadata[0]['id'] == overall_item_key:
                    next_meta = metadata.pop(0)
                    del next_meta['id']
                    new_item['conversations'].append(next_meta)

                outfile.write(json.dumps(new_item) + "\n")
            top_level_items = []
            batch = []
            metadata = []

def prepareJsonlConvos(base_filename, output_filename, max_sentence_length):
    with open(base_filename, 'r') as infile, open(output_filename, 'w', encoding='utf-8') as outfile:
        total = 0
        chars = 0
        chain_number = 0
        overall_number = 0
        last_time = time.time()
        i = 0

        memo = dict()

        for line in tqdm(infile, desc="Processing Dataset"):
            chain = json.loads(line)
            chain_snippet_number = 0
            # Make sure we have an even number of chains
            assert len(chain) % 2 == 0

            for message in chain:
                original_text = message['text']

                if original_text in memo:
                    processed_texts = memo[original_text]
                else:
                    processed_texts = translate_non_code_snippets_and_stitch_back(original_text, translator)
                    memo[original_text] = processed_texts

                if not isinstance(processed_texts, list):
                    processed_texts = [processed_texts]
                
                assert isinstance(processed_texts, list)
                assert len(processed_texts) == 1
                for sub_text in processed_texts:

                    text_to_write = sub_text
                    if isinstance(text_to_write, list):

                        text_to_write = '\n'.join(text_to_write)

                    outfile.write(json.dumps({
                        'processed_text': text_to_write,
                        'chain_number': chain_number,
                        'overall_number': overall_number,
                        'chain_snippet_number': chain_snippet_number,
                        'english': original_text
                    }) + "\n")
                overall_number += 1
                chain_snippet_number += 1

            chain_number += 1
                

            # Write the updated chain with translated text to the output file

            i += 1
            total += len(chain)

        print(total)
        print(chars)



if __name__ == "__main__":
    prepareBlipLaion('llava_v1_5_mix665k.json', 'llava_665k_amh.json', 112)
    prepareJsonlConvos('output.jsonl', 'output_eng.jsonl', 200)