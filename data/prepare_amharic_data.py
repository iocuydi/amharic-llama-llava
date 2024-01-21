import pandas as pd
import json 
import os
import time 

from google.cloud import translate_v2 as translate


def do_cloud_translation(translate_client, text):
    
    if len(text) == 0:
        return text

    result = translate_client.translate(text, target_language="am")
    translated_text = result['translatedText']
    return translated_text


def translate_dolly_15k_jsonl(input_file, output_destination, checkpoint_format_path, checkpoint_frequency=500):
    translate_client = translate.Client()

    characters_translated = 0

    outputs = []

    json_keys = ['category', 'instruction', 'response', 'context']
    translate_keys = {'instruction', 'response', 'context'}

    with open(input_file, 'r') as f:
        for i, line in enumerate(f):

            assert characters_translated < 14000000

            json_line = json.loads(line)
            translated_json = {}
            for key in json_keys:
                if key in translate_keys:
                    text = json_line[key]
                    translated_text = do_cloud_translation(translate_client, text)
                    characters_translated += len(text)
                    translated_json[key] = translated_text
                else:
                    translated_json[key] = json_line[key]

            translated_json['reference_response'] = json_line['response']
            
            scratch_file_path = 'scratch.txt'
            with open(scratch_file_path, 'w', encoding='utf-8') as f:
                f.write("Initial {}\n New {}".format(json_line, translated_json))
            #abf = input('set {}'.format(characters_translated))
            print('Translated {} lines and {} chars so far'.format(i, characters_translated))
            time.sleep(2.5)
            outputs.append(translated_json)
            if i % checkpoint_frequency == 0:
                print("Checkpoint: {}".format(i))
                print('Characters translated: {}'.format(characters_translated))
                checkpoint_path = checkpoint_format_path.format(i)
                with open(checkpoint_path, 'w', encoding='utf-8') as f:
                    json.dump(outputs, f, ensure_ascii=False, indent=4)

    with open(output_destination, 'w', encoding='utf-8') as f:
        # dump output list into one json 
        json.dump(outputs, f, ensure_ascii=False, indent=4)


def translate_non_code_snippets_and_stitch_back(text, translate_client):

    error_suspicious = False
    if not ('```' in text or '`' in text):
        return do_cloud_translation(translate_client, text), len(text), error_suspicious

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
        translated_non_code_snippets.append(do_cloud_translation(translate_client, non_code_snippet))
        translated_length += len(non_code_snippet)

    # now, stitch the non-code snippets and code snippets back together
    translated_text = ''
    for i, non_code_snippet in enumerate(translated_non_code_snippets):
        translated_text += non_code_snippet
        if i < len(code_snippets):
            translated_text += code_delimiters[i] + code_snippets[i] + code_delimiters[i]

    return translated_text, translated_length, error_suspicious

def translate_parquet_file(parquet_file_path, output_destination, checkpoint_format_path, checkpoint_frequency=500):

    outputs = []
    translate_client = translate.Client()

    characters_translated = 0
    i = 0
    data = pd.read_parquet(parquet_file_path, engine='pyarrow')

    for index, row in data.iterrows():

        if i <= 45000:
            i += 1
            continue
        #abf = input(row)

        assert characters_translated < 37000000

        prompt = row['prompt']
        chosen = row['chosen']

        # prompt begins with Human: and ends with Assistant: 
        if prompt.startswith('Human: '):
            prompt = prompt[7:]
        if prompt.endswith(' Assistant:'):
            prompt = prompt[:-11]


        translated_json = {}
        translated_json['prompt'], c_t_prompt, err_p = translate_non_code_snippets_and_stitch_back(prompt, translate_client)
        translated_json['chosen'], c_t_chosen, err_c = translate_non_code_snippets_and_stitch_back(chosen, translate_client)
        translated_json['reference_index'] = index
        translated_json['error_suspicion'] = err_p or err_c

        characters_translated += c_t_prompt
        characters_translated += c_t_chosen

        outputs.append(translated_json)

        scratch_file_path = 'scratch.txt'
        with open(scratch_file_path, 'w', encoding='utf-8') as f:
            f.write("Initial {}\n New {}".format({'prompt': row['prompt'], 'chosen': row['chosen']}, translated_json))
        #abf = input('set {}'.format(characters_translated))
        print('Translated {} lines and {} chars so far'.format(i, characters_translated))
        time.sleep(1)

        if i % checkpoint_frequency == 0:
            print("Checkpoint: {}".format(i))
            print('Characters translated: {}'.format(characters_translated))
            checkpoint_path = checkpoint_format_path.format(i)
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(outputs, f, ensure_ascii=False, indent=4)
        
        i += 1

    with open(output_destination, 'w', encoding='utf-8') as f:
        # dump output list into one json 
        json.dump(outputs, f, ensure_ascii=False, indent=4)

alpaca_src = ''
alpaca_dest = ''
alpaca_checkpoint = ''

dolly_src = ''
dolly_dest = ''
dolly_checkpoint = ''

translate_parquet_file(alpaca_src, alpaca_dest, alpaca_checkpoint)
translate_dolly_15k_jsonl(dolly_src, dolly_dest, dolly_checkpoint)


