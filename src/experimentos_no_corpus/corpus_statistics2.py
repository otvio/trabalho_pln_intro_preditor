import nlpnet
nlpnet.set_data_dir('pos-pt/')
nlpnet_POSTagger = nlpnet.POSTagger()

filenames = [ ('ENSINO_FUNDAMENTAL_amostras_corpus/part' + str(i) + '_ENSINO_FUNDAMENTAL_historia_e_geografia.txt') for i in range(171) ]
for i in range(70):
    filenames.append('ENSINO_MEDIO_amostras_corpus/part' + str(i) + '_ENSINO_MEDIO_ciencias_humanas.txt')
for i in range(127):
    filenames.append('ENSINO_MEDIO_amostras_corpus/part' + str(i) + '_ENSINO_MEDIO_ciencias_humanas_II.txt')

tags = ['<imagem>', '<tabela>', '<gráfico>', '[Figura]']
counting = {}
for tag in tags:
    counting[tag] = 0

for filename in filenames:

    file_content = ''
    with open(filename, encoding='utf-8') as fp:
        file_content = fp.read()

    for tag in tags:
        counting[tag] += file_content.count(tag)

print('Counting tags:')
for tag in tags:
    print(tag + ': ', counting[tag])

