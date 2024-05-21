import json
import glob
import os

def convert_json_to_jsonl(input_file, output_file):
    # JSON 파일을 읽습니다.
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 원본 데이터를 변환합니다.
    original_data = data['sessionInfo']

    # 변환된 데이터를 저장할 리스트를 만듭니다.
    converted_data = []

    for conversation in original_data:
        current_conversation = {'messages': []}
        
        for item in conversation['dialog']:
            role = 'assistant' if item['speaker'] == 'speaker1' else 'user'
            current_conversation['messages'].append({
                'role': role,
                'content': item['utterance']
            })
        
        converted_data.append(current_conversation)
    
    # jsonl 파일로 저장합니다.
    with open(output_file, 'a', encoding='utf-8') as f:
        for conversation in converted_data:
            f.write(json.dumps(conversation, ensure_ascii=False) + '\n')


def process_all_json_files(input_folder, output_file):
    # 폴더 내 모든 JSON 파일을 찾습니다.
    json_files = glob.glob(os.path.join(input_folder, '*.json'))
    
    for json_file in json_files:
        convert_json_to_jsonl(json_file, output_file)
        print(f"{json_file} 파일이 {output_file} 파일로 변환되었습니다.")


input_folder = 'dataset'  # JSON 파일이 있는 폴더 경로
output_filename = 'output.jsonl'



# 변환 함수를 호출합니다.
process_all_json_files(input_folder, output_filename)

print(f"{output_filename} 파일로 변환이 완료되었습니다.")