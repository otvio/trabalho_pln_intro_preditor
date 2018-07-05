import math
import string

import nlpnet
nlpnet.set_data_dir('pos-pt/')
nlpnet_POSTagger = nlpnet.POSTagger()

ttl_words = 0
min_words = math.inf
max_words = -math.inf
avg_words = 0

filenames = [ ('ENSINO_FUNDAMENTAL_amostras_corpus/part' + str(i) + '_ENSINO_FUNDAMENTAL_historia_e_geografia.txt') for i in range(171) ]
for i in range(70):
    filenames.append('ENSINO_MEDIO_amostras_corpus/part' + str(i) + '_ENSINO_MEDIO_ciencias_humanas.txt')
for i in range(127):
    filenames.append('ENSINO_MEDIO_amostras_corpus/part' + str(i) + '_ENSINO_MEDIO_ciencias_humanas_II.txt')

for filename in filenames:

    file_content = ''
    with open(filename, encoding='utf-8') as fp:
        file_content = fp.read()

    file_content = file_content.replace('<title>', '')
    file_content = file_content.replace('</title>', '')
    file_content = file_content.replace('<subtitle>', '')
    file_content = file_content.replace('</subtitle>', '')
    file_content = file_content.replace('<imagem>', '')
    file_content = file_content.replace('<figura>', '')
    file_content = file_content.replace('<tabela>', '')
    file_content = file_content.replace('<gráfico>', '')
    file_content = file_content.replace('[Figura]', '')

    tokens = []
    for sentence in nlpnet_POSTagger.tag(file_content):
        for (token, tag) in sentence:
            if (tag != 'PU' and (token not in string.punctuation)):
                tokens.append(token)

    ttl_words += len(tokens)
    min_words = min(min_words, len(tokens))
    max_words = max(max_words, len(tokens))

avg_words = (1.0 * ttl_words) / len(filenames)

print('ttl_words:', ttl_words)
print('min_words:', min_words)
print('max_words:', max_words)
print('avg_words:', avg_words)
