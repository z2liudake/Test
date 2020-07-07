import torch
import torch.nn as nn

def clean_words(input_str):
    punctuation = '.,;:"!?”“_-'
    word_list = input_str.lower().replace('\n',' ').split()
    word_list = [word.strip(punctuation) for word in word_list]
    return word_list

if __name__ == '__main__':
    with open('C:/Users/LQ/Desktop/1324-0.txt',encoding='utf-8') as f:
        text = f.read()
        lines = text.split('\n')
        line = lines[200]
        word_list = sorted(set(clean_words(text)))
        word2index_dict = {word:i for (i,word) in enumerate(word_list)}
        print(len(word2index_dict))
        print(word2index_dict['he'])
        words_in_line = clean_words(line)
        word_tensor = torch.zeros(len(words_in_line),len(word2index_dict))
        for i,word in enumerate(words_in_line):
            word_index = word2index_dict[word]
            word_tensor[i][word_index]=1.0
            print('{:2} {:4} {}'.format(i, word_index, word))
        