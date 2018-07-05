
# coding: utf-8


#cohmetrix
import coh
import logging


# # Usando  SVM
# ## 99% precisao

# In[130]:


import sys
import os
import time

from sklearn import svm
from sklearn.metrics import classification_report


# In[127]:


import numpy as np
#from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split

# In[1]:


#from sklearn.feature_extraction.text import CountVectorizer


# In[2]:


#from sklearn.feature_extraction.text import TfidfTransformer


# In[3]:


from sklearn.naive_bayes import MultinomialNB


# In[4]:


import numpy as np


# In[5]:


from sklearn.linear_model import SGDClassifier


# In[6]:


from sklearn import metrics


# In[119]:


categories = [    
    'Ensino Fundamental',
    'Ensino Medio',
]


# In[106]:


corpus = open("result_interno.csv")
cont_materiais=0
dados_d=[]
target=[]
linha=corpus.readline()
cont_m=0
cont_f=0
while(linha!=""):

    vetor_saida = []
    linha_s=linha.split(",")
    if (int(linha_s[0])==0):
        target.append(0)
        cont_f+=1
    elif(int(linha_s[0])==1):
        target.append(1)
        cont_m+=1
    for i in linha_s[1:len(linha_s)]:
        vetor_saida.append(float(i))
    #for i in range(0,len(linha.split(","))):
    #    vetor_saida.append(i)
    #    print (linha[i])
    #    print(type(linha[i]))
    #print(vetor_saida)
    #for i in linha[1:len(linha)]:
     #   print(i)
    #print(type())
        #print(type(i))
        #print(type()==float)
        #if (type(eval(i))==float):
        #    vetor_saida.append(float(i))


    dados_d.append(vetor_saida)
    cont_materiais+=1
#    dados.append(linha)
#    dados_d[cont_materiais]=dados
    linha=corpus.readline()
#print(cont_materiais)
#print(target)
#print(dados_d)
print("Carregando corpus..")
#print(cont_f,cont_m)
# In[75]:


#dados_d[0] #64 metricas
#len(target)


# In[105]:


#len('ensinomedio') #17enf #11enm


# In[107]:

print("Convertendo vetores em numpy...")
X = np.array(dados_d)
#y[0]


# In[108]:


y = np.array(target)


# In[109]:




print("Separando dados para (1) Treinamento (80%) e para (2) Teste (20%)")
X_train, X_test, y_train, y_test = train_test_split(
    dados_d, target, test_size=0.20, random_state=2018)


# In[131]:


#X_train, X_test, y_train, y_test
# Perform classification with SVM, kernel=rbf
print("Treinando SVM - kernel RBF")
classifier_rbf = svm.SVC()
t0 = time.time()
classifier_rbf.fit(X_train, y_train)
t1 = time.time()
prediction_rbf = classifier_rbf.predict(X_test)
t2 = time.time()
time_rbf_train = t1-t0
time_rbf_predict = t2-t1

print("Treinando SVM - kernel Linear")
# Perform classification with SVM, kernel=linear
classifier_linear = svm.SVC(kernel='linear')
t0 = time.time()
classifier_linear.fit(X_train, y_train)
t1 = time.time()
prediction_linear = classifier_linear.predict(X_test)
t2 = time.time()
time_linear_train = t1-t0
time_linear_predict = t2-t1


# In[132]:


print("Treinando SVM - kernel LinearSVC otimizado")
# Perform classification with SVM, kernel=linear
classifier_liblinear = svm.LinearSVC()
t0 = time.time()
classifier_liblinear.fit(X_train, y_train)
t1 = time.time()
prediction_liblinear = classifier_liblinear.predict(X_test)
t2 = time.time()
time_liblinear_train = t1-t0
time_liblinear_predict = t2-t1

# Print results in a nice table
print("+--------------------------------------------+---------------------+")
print("Results for SVC(kernel=rbf)")
print("Training time: %fs; Prediction time: %fs" % (time_rbf_train, time_rbf_predict))
print(classification_report(y_test, prediction_rbf, target_names=categories))
print("Results for SVC(kernel=linear)")
print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
print(classification_report(y_test, prediction_linear, target_names=categories))
print("Results for LinearSVC()")
print("Training time: %fs; Prediction time: %fs" % (time_liblinear_train, time_liblinear_predict))
print(classification_report(y_test, prediction_liblinear, target_names=categories))
print("+--------------------------------------------+---------------------+")

# In[122]:


#from sklearn import metrics
#print metrics.confusion_matrix(y_test, prediction_linear)


# # Predicao
# ## documentos fora do teste e treino

# In[362]:


print("Criando documento para predizer...")
#docs_new = ["O que aconteceria se você não conseguisse se lembrar do que fez hoje, de onde nasceu, das pessoas de que gosta, de suas preferências pessoais, do endereço de sua casa, de seus familiares? Obviamente não conseguiria constituir sua identidade pessoal, tendo dificuldade até mesmo de organizar sua vida cotidiana. Ao longo de nossa vida, nos lembramos de algumas coisas e nos esquecemos de muitas outras. Fazemos uma seleção nem sempre consciente do que devemos guardar. Lembramonos de pessoas de que gostamos, de eventos que consideramos importantes, enfim, daquilo que tem um significado para nós. Essa memória pode ser obtida de diversas formas: através da leitura, de imagens, da televisão, da música, ou ainda de diálogos que estabelecemos com diferentes pessoas, ou seja, das várias formas de interação que estabelecemos com o mundo. Em algumas sociedades indígenas, por exemplo, são muito importantes as histórias que os mais velhos contam para as crianças, pois é dessa maneira que elas começam a entrar em contato com valores e regras básicas da cultura. Ao ouvir histórias, a criança pode aprender sobre o significado de certos deuses, sobre a origem de seu povo, sobre suas funções na comunidade, quer dizer, ela começa a descobrir a si mesma, bem como seu papel naquela comunidade. Quando resolvo registrar minhas memórias através da linguagem escrita, ou mesmo fazer uma gravação em vídeo ou fita-cassete, provavelmente selecionarei aqueles eventos que me trouxeram alegria, tristeza, marcaram mudanças, que foram importantes ao longo da minha vida e que estão presentes em minha memória. Um ex-funcionário da Companhia de Tecidos Paulista, importante indústria do setor entre os anos 1930 e 1950, fez um relato oral explicando a maneira pela qual o proprietário da fábrica, localizada próximo a Recife (PE), contratava novos trabalhadores. Nas suas lembranças, destacou que: Quem escolhia (o lugar onde a pessoa ia trabalhar, ao sair do depósito) era o Coronel Frederico. Quando chegavam as famílias do interior, no dia de sair do depósito, ele botava um sofá assim em frente da casa grande e sentava. Aí, aqueles agentes, aqueles empregados mandavam a gente ficar assim, de fora numa fila, e ele ia chamando família por família... O exame que ele fazia era “cada um apresente a mão!” Ele passava a mão assim, olhava: “esse aqui ta bom pra tal serviço... LOPES, José Sérgio Leite. A tecelagem dos conflitos de classe na “cidade das chaminés”. São Paulo: Marco Zero; Brasília, DF: Ed. UNB, 1988. p. 51. (Coleção Pensamento Antropológico).","Os primeiros vestígios de atividade contábil situam-se por volta de 2030 d.C., em Uruk, cidade da antiga Mesopotâmia, no território atual do Iraque. Uruk era um centro da civilização sumeriana. Esses primeiros registros contábeis constituíam-se em fichas de barro, guardadas em receptáculos de barro, que eram utilizadas na contagem do patrimônio. Por exemplo, uma ficha de barro poderia representar um boi. Se esse boi fosse transferido para outra pastagem ou fosse emprestado, a sua ficha seria igualmente transferida para um outro receptáculo de barro, registrando dessa forma o evento ocorrido e auxiliando o controle do patrimônio por parte do proprietário. Dessa forma, um único evento contábil (por exemplo, um empréstimo de um boi) envolveria dois receptáculos de barro: um, representando o estoque de bois do dono do boi, forneceria uma ficha e outro, representando o direito do dono do boi sobre a pessoa que estava tomando o boi emprestado, receberia esta ficha. Isto seria um duplo registro da transação, ou em outras palavras, um lançamento de partida dobrada. Após a criação das fichas de barro para o controle da contabilidade, houve a criação de tábuas com escritos cuneiformes, para a contabilização de pão, cerveja, materiais e trabalho escravo, em Uruk e em Ur, também na Suméria. Dessa forma, a invenção da escrita pelo homem está intimamente ligada ao surgimento da contabilidade. O Antigo Egito também contribuiu com grandes avanços na ciência contábil, principalmente devido à necessidade do governo de organizar a arrecadação de impostos. Os antigos egípcios inovaram ao efetuar os registros contábeis utilizando valores monetários, no caso o shat de ouro e prata.","As cidades passaram a concentrar cada vez mais indústrias, comércio e serviços, atraindo trabalhadores de todo o país. Nos anos 70, a maior leva de migrantes dirigiu-se do Nordeste para as grandes cidades do Sudeste (sobretudo São Paulo e Rio de Janeiro), buscando emprego e melhores salários. No Planalto Central, ergue-se Brasília como sede do poder político e “ponte” para ocupar o Brasil central. A partir da ampliação dos transportes e das comunicações, mais mercadorias, pessoas e informações passam a circular no país, tendo por base as cidades e sob o comando das metrópoles do Sudeste. Nos anos 80, os deslocamentos de pessoas tomam várias direções, como do sul para a Amazônia, expandindo fronteiras agrícolas e gerando novas cidades. <title> SINOP </title> Sinop é um município localizado no norte de Mato Grosso, a 505 km de Cuiabá, capital do Estado. Com a vinda de agricultores do Estado do Paraná para a região, em 1970, criou-se o município que hoje é um bom exemplo de rápido crescimento dos núcleos urbanos no Brasil. Na época, uma empresa com sede em Maringá (PR) comprou uma imensa área em Mato Grosso. Ali viria a ser fundada, em 1974, uma pequena vila, que recebeu o nome da sigla da empresa, SINOP – Sociedade Imobiliária Noroeste do Paraná. Em 1976, o pequeno povoado passa a pertencer ao município de Chapada dos Guimarães. Três anos depois, já era independente, com prefeitura e vereadores. O município cresceu aceleradamente. Segundo o IBGE, em 2000 – portanto, em menos de 30 anos – atinge cerca de 75 mil habitantes, a maior parte vivendo na cidade. Hoje, um dos grandes desafios em Sinop é evitar a redução de florestas e da fauna da região, retiradas para agricultura, criação de animais e exploração de madeira. Disponível em: http://www.sinop.mt.gov.br . desenvolvimento: capitalista refere-se ao desenvolvimento social e econômico do capitalismo, forma de organização social marcada pela propriedade privada, divisão em classes sociais e apropriação de riquezas pelos setores dominantes. De forma geral, esse desenvolvimento supõe a ênfase em aspectos como o crescimento da produção de bens e riquezas, ampliação de mercados consumidores e, em alguma medida, a melhoria de índices sociais. Neste último caso, não é o que vem ocorrendo em países que vivem sob este sistema, como o Brasil e boa parte da América do Sul, África e Ásia, onde há profundas desigualdades sociais.","Agora já podemos prosseguir com nossa conversa. Começávamos a falar do surgimento do trabalho mecânico e automático realizado pelas máquinas. Quando se trata desse assunto, é preciso levar em conta necessariamente as formas de energias utilizadas, pois, sem elas, as máquinas não se mantêm em movimento. A força humana e a dos animais são limitadas para colocar em funcionamento dezenas de máquinas durante muito tempo, assim como a água também cria uma série de dificuldades. Já pensou como uma locomotiva movida a força hidráulica poderia se deslocar? Impossível, não é? O vapor foi a grande fonte de energia que revolucionou o funcionamento das máquinas e aparelhos, alterando bastante seu modo de funcionar. Sabe qual é seu princípio básico de funcionamento? É semelhante ao da panela de pressão que você tem em casa: o vapor da água fervente que circula em uma caldeira bem fechada, produz forte pressão e precisa achar uma saída, sob pena de explodir o recipiente. Ao deixar escapar essa pressão por uma pequena saída direcionada (pinos, bicos etc.), ela produz força capaz de movimentar algo (um pistão, uma catraca, engrenagens ou rodas). As primeiras máquinas com essa concepção surgiram no fim do século XVIII (1701-1800), inventadas por um escocês chamado James Watt (1769), mas elas só começaram a se expandir no começo do século XIX."]
#X_new_counts = count_vect.transform(docs_new)
#X_new_tfidf = vectorizer.transform(docs_new)
#ensinomedio
#docs_new =["Entretanto, percebeu-se que o controle na esfera local não tem sido suficiente para barrar a degradação ambiental acelerada que está em curso. Em resposta a isso, buscaram-se, por meio de convenções internacionais, acordos entre países para controlar os problemas ambientais e impor uma ordem ambiental internacional para deter a devastação dos ambientes naturais. Diversas reuniões internacionais foram realizadas para elaborar acordos sobre a emissão de gases que aumentam o efeito-estufa, a desertificação (formação de deserto em áreas antes recobertas com vegetação) e o acesso à informação genética (os genes usados na manipulação genética promovida pela biotecnologia e pela engenharia genética) contida nas áreas protegidas, aquelas em que é proibida a devastação ambiental. Elas ainda não foram suficientes para barrar a visão imediatista que prevalece na relação com o ambiente, mas indicam caminhos alternativos para uma sociedade mais organizada em escala internacional no futuro. No Brasil, criou-se um sistema ambiental diversificado que abrange diversos temas relacionados à conservação ambiental. Nele está prevista a participação da sociedade organizada com representantes em conselhos, nas audiências públicas e também nos comitês de bacia hidrográfica, que cuidam dos principais rios do país. De maneira organizada, é possível influenciar as decisões que afetam o ambiente e as reservas naturais de nosso país. Também é possível combinar a agenda “verde” com a agenda “marrom”, aquela que trata dos temas referentes à saúde. Afinal, sem saneamento básico, água de qualidade, coleta de lixo, e sem controle da poluição em suas diversas manifestações, fica difícil ter uma boa saúde. Por isso, é importante lutar pela conservação ambiental. É uma maneira de melhorar a qualidade de vida de cada um de nós e, desta forma, vislumbrar um mundo melhor. Mas precisamos ter pressa, pois a devastação tem sido rápida e implacável! Você pode participar dessa empreitada!"]
#docs_new =["O adubo orgânico consiste no aproveitamento de dejetos animais para proteger o solo e repor os nutrientesp ara viabilizar o uso do adubo orgânico é preciso coletar os dejetos animais, acondicionar e depois secar. Eles são aplicados diretamente ao solo com duas vantagens que permitem um destino final mais adequado com menores impactos ambientais para os dejetos orgânicos de animais e evitam a presença de insumos químicos no solo, os nutrientes artificiais que são lançados ao solo para repor sua capacidade produtiva. Outra vantagem da agricultura orgânica e tradicional é o uso intenso de mao de obra. Todas as etapas da produção são desenvolvidas por mãos humanas, desde o preparo do solo, a semeadura até a colheita. Em tempos de elevados índices de desemprego, a agricultura tradicional pode representar uma alternativa para milhares de trabalhadores que vivem sem trabalho e sem dignidade em cidades. Veja, no quadro abaixo, que a agricultura alternativa não é uma novidade no Brasil e no mundo. Nas décadas anteriroes a oposição com sedimentação do padrão químico, motomecânico e genético da agricultura moderna impulsionou o surgimento de movimentos rebeldes que valorizavam o potencial biológico e vegetativo dos processos produtivos. Na Europa, surgiram as vertentes biodinâmica, orgânica e biológica e no Japão, a agricultura natural. Muito hostilizados, esses movimentos se mantiveram à margem da produção agrícola mundial e da comunidade científica agronômica. Nos anos 70 as evidências dos efeitos adversos provocados pelo padrão predominante, que passava a ser chamado agricultura convencional, fortalecem um conjunto de propostas rebeldes que passam a ser conhecidas como alternativas. Na década de 80, cresce o interesse pelas práticas alternativas, principalmente no sistema oficial de pesquisa norteamericano. A hostilidade, aos poucos, vai se transformando em curiosidade. O movimento alternativo também tem desdobramentos no Brasil e, a partir dos anos 70, durante o auge da modernização agrícola, chegam ao país as principais vertentes internacionais. Nos anos 80, já havia dezenas de organizações não governamentais que criticávamos efeitos adversos do padrão convencional e divulgavam as propostas alternativas. A ação dessas entidades contribuiu para que alguns ideais alternativos penetrassem em certas esferas do poder público. É difícil mensurar o impacto desse movimento na agricultura brasileira, pois, assim como em outros países, os sistemas alternativos continuam ocupando uma posição marginal em relação às práticas convencionais. Mas, sem dúvida, cresceu no Brasil o interesse e a preocupação com as questões que relacionam a produção agrícola e o meio ambiente."]
docs_new =["A ação do homem no meio ambiente. A ação antrópica tem se provado fundamental na modificação das condições terrestres, principalmente quanto ao desmatamento e ao ampliamento do efeito estufa. No entanto é preciso que fique claro que nem todos os desastres ambientais são causados pelo homem: abalos sísmicos e vulcanismos, por exemplo, não são influenciados por tais ações, pois ocorrerem abaixo da superfície terrestre, por isso não deve haver generalização. Impacto ambiental, a ação do homem geralmente tem impactos ambientais negativos, porque as consequências ambientais de suas atividades são geralmente desconsideradas ou ignoradas. Além disso, acidentes que causam grandes danos ambientais podem ocorrer, como vazamento de óleo."]

""
logging.basicConfig(level=logging.INFO)

all_metrics = coh.MetricsSet([coh.BasicCounts(),
                              coh.LogicOperators(),
                              coh.Frequencies(),
                              coh.Hypernyms(),
                              coh.Tokens(),
                              coh.Connectives(),
                              coh.Ambiguity(),
                              coh.SyntacticalComplexity(),
                              coh.SemanticDensity(),
                              coh.Constituents(),
                              coh.Anaphoras(),
                              coh.Coreference(),
                              coh.Lsa(),
                              # coh.Disfluencies(),
                          ])

print("Carregando métricas...")

t = coh.Text(docs_new[0])

r = all_metrics.values_for_text(t)

print(r.as_table())


print("                     Possível classificação do nível escolar")
#predicted = classifier_liblinear.predict(r.as_array())
#predicted = classifier_linear.predict(r.as_array())
predicted = classifier_rbf.predict(r.as_array())
#print("predicted:",predicted)
for doc, category in zip(docs_new, predicted):    
    #print("                     Num_Rotulo:",category, categories[category])
    print(                     categories[category])
   
print("+--------------------------------------------+---------------------+")



