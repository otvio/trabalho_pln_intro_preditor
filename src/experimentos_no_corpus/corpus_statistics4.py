# PCP,

import matplotlib.pyplot as plt
import nlpnet
nlpnet.set_data_dir('pos-pt/')
nlpnet_POSTagger = nlpnet.POSTagger()

FUND, MEDIO = 0, 1
filenames = list()
filenames.append([ ('ENSINO_FUNDAMENTAL_amostras_corpus/part' + str(i) + '_ENSINO_FUNDAMENTAL_historia_e_geografia.txt') for i in range(171) ])
filenames.append(list())
for i in range(70):
    filenames[MEDIO].append('ENSINO_MEDIO_amostras_corpus/part' + str(i) + '_ENSINO_MEDIO_ciencias_humanas.txt')
for i in range(127):
    filenames[MEDIO].append('ENSINO_MEDIO_amostras_corpus/part' + str(i) + '_ENSINO_MEDIO_ciencias_humanas_II.txt')

disconsidered_text_tags = ['<title>', '</title>', '<subtitle>', '</subtitle>', '<imagem>', '<figura>', '<tabela>', '<gráfico>', '[Figura]']

tag_list = ['ADJ', 'ADV', 'ADV-KS', 'ADV-KS-REL', 'ART', 'CUR', 'IN', 'KC', 'KS', 'N', 'NPROP', 'NUM', 'PCP', 'PDEN', 'PREP', 'PREP+ADV', 'PREP+ART', 'PREP+PROADJ', 'PREP+PRO-KS', 'PREP+PRO-KS-REL', 'PREP+PROPESS', 'PREP+PROSUB', 'PROADJ', 'PRO-KS', 'PRO-KS-REL', 'PROPESS', 'PROSUB', 'PU', 'V', 'VAUX']

tag_being_tested = 'NPROP'

#print(nlpnet_POSTagger.tag('O menino feio é feio. Após isso, ele é feio.'))
#exit(0)

arr = list()
for level_id in [FUND, MEDIO]:
    arr.append(list())

    for filename in filenames[level_id]:
        file_content = ''
        with open(filename, encoding='utf-8') as fp:
            file_content = fp.read()

        for tag in disconsidered_text_tags:
            file_content = file_content.replace(tag, '')

        count = 0
        last, bef_last = '', ''
        for sentence in nlpnet_POSTagger.tag(file_content):
            for (token, tag) in sentence:
                if bef_last == tag_being_tested and last == tag_being_tested and tag == tag_being_tested:
                    count += 1
                bef_last = last
                last = tag

        arr[level_id].append(count)

plot_id = 0
plot_var = list()
for level_id in [FUND, MEDIO]:
    title = 'TRIPLE_' + tag_being_tested + '_' + ['histograma_ensino_fundamental', 'histograma_ensino_medio'][level_id]
    plot_var.append(plt.figure(plot_id))
    plt.hist(arr[level_id])
    plt.title(title)
    plt.xlabel('Contagem de ' + 'TRIPLE_' + tag_being_tested + ' por amostra')
    plt.ylabel('Quantidade de amostras')
    #plot_var[level_id].show()
    plot_var[level_id].savefig(title + '.png')
    plot_id += 1

