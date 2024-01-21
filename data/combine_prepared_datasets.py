import json
import csv
import random
import pandas as pd
import prompts



def load_parquet_to_json_list(parquet_file_path):
    data = pd.read_parquet(parquet_file_path, engine='pyarrow')

    json_list = []

    for index, row in data.iterrows():
        jso = json.loads(row.to_json())
        jso['index'] = index
        json_list.append(jso)
    
    return json_list

def load_jsonl_to_json_list(jsonl_file_path):
    json_list = []

    with open(jsonl_file_path) as f:
        for line in f:
            json_list.append(json.loads(line))
    
    return json_list

def get_prompt_no_context(language='Amharic'):
    return random.choice(prompts.PREFIX_LIST_NO_CONTEXT).format(language)

def get_prompt_with_context(language='Amharic'):
    return random.choice(prompts.PREFIX_LIST_CONTEXT).format(language)

def get_prompt_translation(src_lang, targ_lang):
    return random.choice(prompts.PREFIX_LIST_TRANSLATION).format(src_lang, targ_lang)

def get_prompt_headline_from_article():
    return random.choice(prompts.PREFIX_LIST_HEADLINE)

def get_prompt_article_from_headline():
    return random.choice(prompts.PREFIX_LIST_STORY_FROM_HEADLINE)

def get_prompt_summary_from_article():
    return random.choice(prompts.PREFIX_LIST_SUMMARY)

def get_prompt_article_from_summary():
    return random.choice(prompts.PREFIX_LIST_STORY_FROM_SUMMARY)

def join_alpaca_english_amharic(english_list, amharic_list, with_translation=True, allow_english=True, with_amharic=False):

    new_list = []
    for idx, item in enumerate(english_list):

        prompt_format = get_prompt_no_context() + "\nHuman: {}\nAssistant{}: "

        prompt_format_translation_eng_am = get_prompt_translation('English', 'Amharic') + "\nEnglish: {}\nAmharic: "
        prompt_format_translation_am_eng = get_prompt_translation('Amharic', 'English') + "\nAmharic: {}\nEnglish: "

        english_prompt = item['prompt']
        if english_prompt.startswith('Human: '):
            english_prompt = english_prompt[7:]
        if english_prompt.endswith(' Assistant:'):
            english_prompt = english_prompt[:-11]

        amharic_prompt = amharic_list[idx]['prompt']

        english_response = item['chosen']
        amharic_response = amharic_list[idx]['chosen']

        if allow_english:

            # append the english to amharic
            new_list.append({'input': prompt_format.format(english_prompt, ' [Amharic] '), 'output': amharic_response})

            # append the amharic to english
            new_list.append({'input': prompt_format.format(amharic_prompt, ' [English] '), 'output': english_response})

        if with_amharic:

            # append the amharic to amharic
            new_list.append({'input': prompt_format.format(amharic_prompt, ' [Amharic] '), 'output': amharic_response})

        if with_translation:

            # append translation variant for prompt, english to amharic
            new_list.append({'input': prompt_format_translation_eng_am.format(english_prompt), 'output': amharic_prompt})

            # append translation variant for prompt, amharic to english
            new_list.append({'input': prompt_format_translation_am_eng.format(amharic_prompt), 'output': english_prompt})

            # append translation variant for response, english to amharic
            new_list.append({'input': prompt_format_translation_eng_am.format(english_response), 'output': amharic_response})

            # append translation variant for response, amharic to english
            new_list.append({'input': prompt_format_translation_am_eng.format(amharic_response), 'output': english_response})
        

    return new_list

def join_dolly_english_amharic(english_list, amharic_list, with_translation=True, allow_english=True, with_amharic=False):


    new_list = []
    for idx, item in enumerate(english_list):
        
        prompt_format_no_context = get_prompt_no_context() + "\nHuman: {}\nAssistant{}: "

        prompt_format_context = get_prompt_with_context() + "\nContext: {}\nHuman: {}\nAssistant{}: "


        prompt_format_translation_eng_am = get_prompt_translation('English', 'Amharic') + "\nEnglish: {}\nAmharic: "
        prompt_format_translation_am_eng = get_prompt_translation('Amharic', 'English') + "\nAmharic: {}\nEnglish: "


        enlglish_prompt = item['instruction']
        amharic_prompt = amharic_list[idx]['instruction']

        english_response = item['response']
        amharic_response = amharic_list[idx]['response']

        context = item['context']
        amharic_context = amharic_list[idx]['context']

        if len(context) > 0:
            if allow_english:
                new_list.append({'input': prompt_format_context.format(context, enlglish_prompt, ' [Amharic] '), 'output': amharic_response})
                new_list.append({'input': prompt_format_context.format(context, amharic_prompt, ' [English] '), 'output': english_response})
                new_list.append({'input': prompt_format_context.format(context, amharic_prompt, ' [Amharic] '), 'output': amharic_response})
                new_list.append({'input': prompt_format_context.format(amharic_context, enlglish_prompt, ' [Amharic] '), 'output': amharic_response})
                new_list.append({'input': prompt_format_context.format(amharic_context, amharic_prompt, ' [English] '), 'output': english_response})

            if with_amharic:
                new_list.append({'input': prompt_format_context.format(amharic_context, amharic_prompt, ' [Amharic] '), 'output': amharic_response})

            if with_translation:

                # translation english to amharic context
                new_list.append({'input': prompt_format_translation_eng_am.format(context), 'output': amharic_context})
                # translation amharic to english context
                new_list.append({'input': prompt_format_translation_am_eng.format(amharic_context), 'output': context})
        else:
            if allow_english:
                new_list.append({'input': prompt_format_no_context.format(enlglish_prompt, ' [Amharic] '), 'output': amharic_response})
                new_list.append({'input': prompt_format_no_context.format(amharic_prompt, ' [English] '), 'output': english_response})


            if with_amharic:
                new_list.append({'input': prompt_format_no_context.format(amharic_prompt, ' [Amharic] '), 'output': amharic_response})

        if with_translation:

            # append translation variant for prompt, english to amharic
            new_list.append({'input': prompt_format_translation_eng_am.format(enlglish_prompt), 'output': amharic_prompt})

            # append translation variant for prompt, amharic to english
            new_list.append({'input': prompt_format_translation_am_eng.format(amharic_prompt), 'output': enlglish_prompt})

            # append translation variant for response, english to amharic
            new_list.append({'input': prompt_format_translation_eng_am.format(english_response), 'output': amharic_response})

            # append translation variant for response, amharic to english
            new_list.append({'input': prompt_format_translation_am_eng.format(amharic_response), 'output': english_response})
        #abf = input(new_list)

    return new_list


def make_sanity_test(english_list):

    new_list = []
    for idx, item in enumerate(english_list):
        #abf = input(item)
        new_list.append({'input': item['prompt'], 'output': item['chosen']})

    return new_list

def prepare_tsv_dataset(tsv_files, with_amharic=False):

    if not with_amharic:
        return []

    new_list = []

    for tsv_file in tsv_files:
        with open(tsv_file, 'r', encoding='utf-8') as file:
            
            prompt_headline_to_text = get_prompt_article_from_headline() + "\nHuman: {}\nAssistant: "
            prompt_text_to_headline = get_prompt_headline_from_article() + "\nHuman: {}\nAssistant: "

            # Read the file using the csv DictReader, specifying the delimiter as a tab character
            reader = csv.DictReader(file, delimiter='\t')
            
            # Iterate over each row in the file
            for row in reader:
                rd = {}
                for key, value in row.items():
                    rd[key] = value
                
                headline = rd['headline']
                text = rd['text']

                # append the headline to text
                new_list.append({'input': prompt_headline_to_text.format(headline), 'output': text})

                # append the text to headline
                new_list.append({'input': prompt_text_to_headline.format(text), 'output': headline})

    return new_list

def prepare_jsonl_dataset(jsonl_files, with_amharic=False):

    if not with_amharic:
        return []
    new_list = []

    for jsonl_file in jsonl_files:


        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                json_line = json.loads(line)

                summary = json_line['summary']
                headline = json_line['title']
                text = json_line['text']

                prompt_headline_to_text = get_prompt_article_from_headline() + "\nHuman: {}\nAssistant: "
                prompt_text_to_headline = get_prompt_headline_from_article() + "\nHuman: {}\nAssistant: "

                prompt_summary_to_text = get_prompt_article_from_summary() + "\nHuman: {}\nAssistant: "
                prompt_text_to_summary = get_prompt_summary_from_article() + "\nHuman: {}\nAssistant: "

                # append the headline to text
                new_list.append({'input': prompt_headline_to_text.format(headline), 'output': text})

                # append the text to headline
                new_list.append({'input': prompt_text_to_headline.format(text), 'output': headline})

                # append the summary to text
                new_list.append({'input': prompt_summary_to_text.format(summary), 'output': text})

                # append the text to summary
                new_list.append({'input': prompt_text_to_summary.format(text), 'output': summary})
        
    
    return new_list

def prepare_paired_translation_dataset(txt_files, with_translation=True):

    new_list = []

    if not with_translation:
        return new_list

    for target in txt_files:
        alt_file = target.replace('.am', '.en').replace('opus.en', 'opus.am')

        # get all lines inside target
        with open(target, 'r', encoding='utf-8') as f:
            target_lines = f.readlines()
        # get all lines inside alt
        with open(alt_file, 'r', encoding='utf-8') as f:
            alt_lines = f.readlines()
        
        assert len(target_lines) == len(alt_lines)

        for idx, line in enumerate(target_lines):
            prompt_format_translation_eng_am = get_prompt_translation('English', 'Amharic') + "\nEnglish: {}\nAmharic: "
            prompt_format_translation_am_eng = get_prompt_translation('Amharic', 'English') + "\nAmharic: {}\nEnglish: "

            new_list.append({'input': prompt_format_translation_eng_am.format(alt_lines[idx]), 'output': line})
            new_list.append({'input': prompt_format_translation_am_eng.format(line), 'output': alt_lines[idx]})

    #with open('test_headlines3.json', 'w', encoding='utf-8') as f:
    #    json.dump(new_list, f, ensure_ascii=False)
    
    return new_list


def join_english_and_amharic():

    dolly_jsonl_path = ''
    output_destination = ''


    alpaca_train_path = ''
    alpaca_test_path = ''


    alpaca_test_destination = ''
    alpaca_train_destination = ''


    alpaca_english_test = load_parquet_to_json_list(alpaca_test_path)
    alpaca_english_train = load_parquet_to_json_list(alpaca_train_path)

    sanity_test = make_sanity_test(alpaca_english_test)

    dolly_english = load_jsonl_to_json_list(dolly_jsonl_path)

    dolly_amharic = json.load(open(output_destination, 'r', encoding='utf-8'))

    alpaca_amharic_test = json.load(open(alpaca_test_destination,  'r', encoding='utf-8'))

    alpaca_amharic_train = json.load(open(alpaca_train_destination,  'r', encoding='utf-8'))



    all_tsv_data = prepare_tsv_dataset([''])

    all_jsonl_data = prepare_jsonl_dataset([''])

    all_paired_translation_data = prepare_paired_translation_dataset([''])

    tsv_len = len(all_tsv_data)
    jsonl_len = len(all_jsonl_data)
    paired_translation_len = len(all_paired_translation_data)

    tsv_bound = int(tsv_len * 0.9)
    jsonl_bound = int(jsonl_len * 0.9)
    paired_translation_bound = int(paired_translation_len * 0.9)

    all_tsv_train = all_tsv_data[:tsv_bound]
    all_tsv_test = all_tsv_data[tsv_bound:]

    all_jsonl_train = all_jsonl_data[:jsonl_bound]
    all_jsonl_test = all_jsonl_data[jsonl_bound:]

    all_paired_translation_train = all_paired_translation_data[:paired_translation_bound]
    all_paired_translation_test = all_paired_translation_data[paired_translation_bound:]


    # we will create three version of each list: english to amharic, amharic to english, and amharic to amharic
    # then we will combine alpaca and dolly together

    # we don't want to leak the test data into the training data, so we will split dolly train and test before we do this
    # first shuffle the dolly data
    dolly_len = len(dolly_amharic)
    bound = int(dolly_len * 0.9)
    dolly_amharic_train = dolly_amharic[:bound]
    dolly_amharic_test = dolly_amharic[bound:]

    dolly_english_train = dolly_english[:bound]
    dolly_english_test = dolly_english[bound:]


    alpaca_train = join_alpaca_english_amharic(alpaca_english_train, alpaca_amharic_train)
    alpaca_test = join_alpaca_english_amharic(alpaca_english_test, alpaca_amharic_test)

    dolly_train = join_dolly_english_amharic(dolly_english_train, dolly_amharic_train)
    dolly_test = join_dolly_english_amharic(dolly_english_test, dolly_amharic_test)




    print('LENGTHS: ')
    print(len(alpaca_train))
    print(len(alpaca_test))
    print(len(dolly_train))
    print(len(dolly_test))

    print('additional dataset lengths: ')

    print(len(all_tsv_train))
    print(len(all_tsv_test))
    print(len(all_jsonl_train))
    print(len(all_jsonl_test))
    print(len(all_paired_translation_train))
    print(len(all_paired_translation_test))


    # join both sets together
    all_train = alpaca_train + dolly_train + all_tsv_train + all_jsonl_train + all_paired_translation_train
    all_test = alpaca_test + dolly_test + all_tsv_test + all_jsonl_test + all_paired_translation_test

    print('LENGTHS: ')
    print(len(all_train))
    print(len(all_test))

    # save results
    save_dir = 'final_multi\\{}'

    with open(save_dir.format('second_all_train.json'), 'w', encoding='utf-8') as f:
        json.dump(all_train, f, ensure_ascii=False)
    
    with open(save_dir.format('second_all_test.json'), 'w', encoding='utf-8') as f:
        json.dump(all_test, f, ensure_ascii=False)


join_english_and_amharic()