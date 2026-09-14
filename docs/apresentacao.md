# ShellRefine, segmentação da cabeça fetal em ultrassom 3D

Texto de apoio para apresentação. Cada secção corresponde mais ou menos a um slide.

---

## 1. O problema

Segmentar a cabeça fetal em volumes de ultrassom 3D. Temos 90 casos anotados.
O ultrassom é difícil porque o crânio aparece como um anel muito brilhante, mas
o resto do contorno esbate-se, e há sombra acústica que apaga partes da imagem.

## 2. De onde veio a ideia

A primeira versão do sistema previa uma banda de fronteira e depois preenchia o
interior com um flood fill. Isso tinha uma fragilidade grave. Se um único voxel
da banda estivesse errado, o flood fill escapava-se por essa fuga e trocava o
dentro pelo fora no volume inteiro. Um erro local destruía o resultado global.

A redesenho parte daí. Em vez de prever a fronteira, a fase 1 passa a prever um
signed distance field, ou seja, um campo que em cada voxel diz a que distância
está da superfície e de que lado. O interior é simplesmente a região onde o
campo é negativo. Não há flood fill, não há propagação, e um erro local
continua local.

## 3. Como funciona, passo a passo

Primeiro, cada volume é reduzido a um cubo de 32³ e calcula-se o signed distance
field verdadeiro a partir da anotação, truncado aos 10mm.

Depois, a fase 1 treina uma U-Net 3D pequena para regredir esse campo. A saída
usa tanh, portanto os valores ficam entre menos um e um.

A seguir, o campo previsto é levado para a resolução nativa por interpolação
trilinear. Isto funciona bem porque um signed distance field é localmente
linear, portanto o nível zero sobe de resolução com precisão inferior ao voxel.

Depois disso, a máscara grosseira é simplesmente a região onde o campo previsto
é negativo.

A seguir, extrai-se uma banda de mais ou menos 5mm à volta da superfície, e o
volume é cortado pela bounding box dessa banda com 10 voxels de folga.

Depois, a fase 2 treina uma segunda U-Net dentro desse crop, em patches de 64³,
recebendo dois canais. O primeiro canal é a imagem. O segundo canal é o
distance field da fase 1, que funciona como prior sobre onde a superfície deve
estar.

Por fim, a máscara final combina as duas fases. Fora da banda mantém-se a
resposta da fase 1. Dentro da banda usa-se a resposta da fase 2.

A ideia central é esta: gastar resolução nativa apenas na faixa fina onde a
fronteira realmente está, e resolver tudo o resto de forma barata.

## 4. O tamanho do modelo

Cada uma das duas redes tem 1.40 milhões de parâmetros, com 16 canais na base.
O sistema completo tem 2.80 milhões de parâmetros, o que ocupa cerca de 10.7MB.

## 5. Como testámos, o desenho da experiência

Testar bem custa caro, portanto separámos em dois estágios.

O Estágio A é um sweep largo e barato. Cada ponto corre uma só vez, sem k-fold e
sem repetição de seeds. Serve apenas para escolher candidatos.

O Estágio C é a validação a sério, com 5 folds e várias seeds, e só corre nos
finalistas escolhidos pelo Estágio A.

A razão desta separação é simples. Correr k-fold em 41 configurações seria
proibitivo, mas confiar numa única run para decidir também não serve. Então
usa-se o barato para filtrar e o caro para decidir.

## 6. Estágio A, o que foi testado

Na fase 1 testámos primeiro a loss e o learning rate. Comparámos l1 contra mse,
com learning rates de 2e-4, 5e-4 e 1e-3. A l1 ganhou, com 0.9477 de Dice.

Depois testámos ponderar a loss junto da fronteira, dando menos peso aos voxels
longe dela. O melhor resultado foi 0.9464, portanto pior do que não ponderar
nada. A ponderação não ajudou a fase 1.

A seguir testámos acrescentar uma gradient difference loss, que compara a
inclinação local do campo previsto com a do campo real. Isto ajudou, e com
coeficiente 0.1 chegámos a 0.9494, o melhor resultado da fase 1. Faz sentido,
porque a inclinação do campo é máxima exactamente junto à superfície.

Depois testámos combinar a ponderação com o gradiente. Deu 0.9461, pior do que o
gradiente sozinho. A combinação não compensa.

Na fase 2 testámos seis famílias de loss cruzadas com três learning rates, num
total de 18 pontos. As losses foram mse, l1, dice, bce, dice_bce e mse_bce. A
melhor foi dice_bce a 3e-4, com 0.9507 de Dice refinado.

Depois testámos ponderação de fronteira também na fase 2, e a melhor combinação
deu 0.9508.

Por fim fizemos uma verificação de interacção, cruzando as duas melhores fases 1
com as duas melhores fases 2. Nenhuma combinação alternativa chegou perto do
topo, com máximo de 0.9482.

## 6b. Tabela das loss functions testadas

### Fase 1, regressão do signed distance field

Melhor learning rate de cada variante. As losses bce, dice, dice_bce e mse_bce
não entram aqui porque a saída é tanh, com valores entre menos um e um, e essas
quatro exigem valores entre zero e um.

| Loss | lr | Dice | MAE | Leitura |
|---|---|---|---|---|
| mse | 5e-4 | 0.9445 | 0.0503 | pior que l1 em Dice e muito pior em MAE |
| l1 | 5e-4 | 0.9477 | 0.0367 | base de comparação |
| l1 + boundary weight | 5e-4 | 0.9464 | 0.0383 | ponderar a fronteira piorou |
| l1 + boundary weight + gradient | 5e-4 | 0.9461 | 0.0386 | combinar as duas piorou mais |
| **l1 + gradient difference 0.1** | **5e-4** | **0.9494** | **0.0362** | **escolhido, único ganho real** |

Amplitude entre o melhor e o pior ponto da fase 1: 0.0104 de Dice.

### Fase 2, refinamento na banda

Melhor learning rate de cada família de loss, sobre 24 pontos testados.

| Loss | lr | Dice refinado | HD95 mm | ASD mm |
|---|---|---|---|---|
| dice_bce + boundary weight 0.2 | 3e-4 | 0.9508 | 2.860 | 0.9530 |
| dice_bce | 3e-4 | 0.9507 | 2.842 | 0.9547 |
| mse | 5e-4 | 0.9506 | 2.838 | 0.9599 |
| dice | 5e-4 | 0.9504 | 2.851 | 0.9643 |
| l1 | 5e-4 | 0.9504 | 2.869 | 0.9648 |
| mse_bce | 1e-3 | 0.9503 | 2.874 | 0.9654 |
| bce | 3e-4 | 0.9503 | 2.875 | 0.9659 |

Amplitude entre o melhor e o pior ponto da fase 2: 0.0013 de Dice.

### A leitura das duas tabelas juntas

As duas fases contam histórias opostas, e é isso que interessa mostrar.

Na fase 1 a escolha da loss importa. A amplitude é de 0.0104, a mse é claramente
pior que a l1, e o termo de gradient difference dá um ganho real de 0.0017 sobre
a l1 sozinha.

Na fase 2 a escolha da loss não importa. As seis famílias ficam todas dentro de
0.0013 umas das outras, muito abaixo do ruído de seed que é cerca de 0.005. Seis
formulações matemáticas diferentes, três learning rates cada, e o resultado é
indistinguível.

A explicação provável é que a fase 2 só decide dentro de uma banda estreita onde
a resposta já está quase determinada pelo prior da fase 1. O trabalho difícil já
foi feito, e qualquer loss razoável chega ao mesmo sítio.

## 7. Estágio B, os finalistas

Sobraram três configurações separadas por 0.0002 de Dice, ou seja, muito abaixo
do ruído de seed que já tínhamos medido, que ronda 0.005. As três partilham a
mesma fase 1 vencedora, que é l1 com learning rate 5e-4 e gradient coefficient
0.1, e diferem apenas na fase 2.

A primeira, f1_bw, usa dice_bce a 3e-4 com ponderação de fronteira ligada.

A segunda, f2_plain, usa a mesma loss e o mesmo learning rate, mas sem
ponderação nenhuma. Serve para testar se a ponderação é real ou ruído.

A terceira, f3_mse, usa mse a 5e-4, também sem ponderação. Serve para testar se
a família de loss importa.

## 8. Estágio C, a validação

Corremos 5 folds vezes 2 seeds vezes 3 finalistas, o que dá 30 runs completas.
A intenção original eram 3 seeds, mas a terceira falhou a meio por um problema
de recursos no cluster e decidimos ficar pelas duas.

O resultado agregado sobre as 30 runs é Dice de 0.9456 com desvio de 0.0041,
HD95 de 3.199mm e ASD de 1.064mm.

## 9. O resultado mais importante, e é negativo

As três configurações finalistas ficaram empatadas. As diferenças aparecem na
terceira e quarta casa decimal, dentro do próprio ruído de seed.

O que domina a variância não é a escolha do modelo, é o fold. Os folds 0 e 4 dão
sempre cerca de 0.941, e os folds 1 a 3 dão entre 0.947 e 0.950,
independentemente de qual dos três finalistas se usa.

Isto tem duas leituras, e ambas interessam.

A primeira é metodológica. A ordenação que o Estágio A tinha produzido era
ruído. Se tivéssemos parado no Estágio A e declarado um vencedor, estaríamos a
reportar sorte de seed como se fosse resultado.

A segunda é prática. Como a configuração mais simples não é pior, podemos
escolhê-la e defendê-la. Menos hiperparâmetros para justificar, menos risco de
estar a afinar o modelo aos caprichos deste dataset.

## 10. Comparação com o que existe

Comparámos com sete baselines corridas sobre o mesmo dataset e com o mesmo
protocolo, vindas de um projeto de benchmark paralelo.

Em qualidade ficamos em terceiro, com 0.9456, praticamente empatados com o
nnU-Net que faz 0.9466 e com o BiSegMamba que faz 0.9457. A diferença entre os
três é menor do que o desvio-padrão de qualquer um deles.

Em eficiência a diferença é enorme. Temos 2.80 milhões de parâmetros contra
valores entre 31 e 151 milhões nas outras arquitecturas. O pico de VRAM de
treino é de 3.5GB contra valores entre 12 e 55GB.

Em Dice por milhão de parâmetros ficamos cerca de onze vezes acima do segundo
melhor.

A mensagem é que atingimos paridade de qualidade com um modelo uma ordem de
grandeza mais pequeno.

## 11. O que aprendemos sobre os erros

Ao olhar para as figuras qualitativas, a primeira impressão foi que o modelo
sobre-segmentava de forma sistemática, ou seja, que previa uma orla a mais à
volta da cabeça.

Fomos medir e essa impressão estava errada. A impressão vinha de olhar para os
piores casos, que são precisamente os que não representam o conjunto.

Medindo em 90 casos, a fronteira final está essencialmente sem viés. O raio
previsto é 1.0044 vezes o raio anotado, ou seja, cerca de meio voxel para fora.

O que existe mesmo é um viés na fase 1, e no sentido contrário. A máscara
grosseira tem raio 0.973, portanto fica sistematicamente por dentro da anotação.

O mecanismo ficou claro no perfil radial de intensidade. A crista do anel
brilhante do crânio está a 0.933 do raio anotado, ou seja, por dentro da
anotação, que segue a borda externa do crânio. A fase 1 é atraída por esse
brilho e assenta a meio caminho. A fase 2 depois empurra a fronteira de 0.973
para 1.004, portanto está a fazer exactamente aquilo para que foi desenhada.

A conclusão prática é que o alvo não é o viés, é a dispersão. A largura da
distribuição é uma ordem de grandeza maior do que o desvio da média.

## 12. A experiência em curso, resolução da fase 1

A fase 1 corre a 32³ e o upsample para a resolução nativa é cerca de seis vezes.
Isso significa que cada voxel da grelha grosseira decide onde fica a superfície
dentro de um bloco de seis voxels nativos. Há portanto um piso de precisão que
não depende do treino.

Testámos 32³, 48³ e 64³ num fold, mantendo tudo o resto igual. As três métricas
melhoram de forma monotónica.

O Dice da máscara grosseira sobe de 0.9313 para 0.9380 e depois para 0.9423.

O ASD desce de 1.385mm para 1.242mm e depois para 1.167mm.

O viés de raio encolhe de 0.9744 para 0.9805 e depois para 0.9877, portanto mais
de metade dele desaparece.

E o mais relevante, o desvio-padrão do Dice quase parte ao meio, de 0.0210 para
0.0118. Isto ataca precisamente a dispersão, que tínhamos identificado como o
problema real.

O custo é pequeno. O treino da fase 1 passa de 152 para 401 segundos e a VRAM de
696MB para 2034MB. Como a fase 2 já usa 3532MB, o pico do sistema não muda. O
argumento de eficiência mantém-se intacto.

Depois corremos a fase 2 por cima do 64³, no mesmo fold e com a mesma seed, para
ver se o ganho chega ao fim. O Dice refinado sobe de 0.9394 para 0.9436, ou seja
mais 0.0042. O desvio-padrão desce de 0.0129 para 0.0092. O ASD desce de 1.219mm
para 1.143mm.

O ganho sobrevive, mas atenuado. Dos 0.0110 ganhos na fase 1 chegam 0.0042 ao
resultado final, cerca de 38%. Isto é coerente com o que já sabíamos, porque a
fase 2 já corrigia sozinha grande parte do defeito da fase 1, portanto absorve
parte da melhoria.

Aqui é preciso ser honesto sobre o que está e o que não está demonstrado. O
ruído de seed neste sistema ronda 0.005 de Dice. Os 0.0042 do resultado final
ficam abaixo desse limiar, e foram medidos num só fold e numa só seed. Pela
regra que aplicámos a nós próprios durante todo o Estágio C, isto ainda não é um
resultado, é uma diferença dentro do ruído.

O que está sólido é a melhoria da fase 1, com 0.0110 de ganho, monotónica em
três resoluções e com três métricas independentes a concordar. O que falta
demonstrar é que essa melhoria se traduz em resultado final.

O teste que fecha a questão é barato. O braço de 32³ já existe completo do
Estágio C, com 5 folds e 2 seeds. Basta correr o braço de 64³, mais 10 runs, e
comparar caso a caso de forma emparelhada.

## 13. Ideias ainda por testar

Juntar as duas seeds dentro do mesmo fold, o que é legítimo porque ambas viram o
mesmo split de treino, e ataca directamente a dispersão sem treinar nada de novo.

Fazer test time augmentation, ou seja, prever com o volume e com versões
espelhadas e rodadas, e tirar a média.

Caracterizar os casos das caudas, para perceber se têm alguma coisa em comum,
como qualidade de imagem ou orientação.

Há aqui uma tensão que convém dizer em voz alta. O ensemble e o test time
augmentation melhoram o número, mas multiplicam o custo de inferência, e isso
ataca precisamente o argumento de modelo pequeno e barato, que é o nosso ponto
mais forte.

## 14. Lições de método

Filtrar com experiências baratas e decidir apenas com experiências caras.

Medir antes de acreditar numa impressão visual, porque olhar para os piores
casos leva a conclusões erradas sobre o conjunto.

Estabelecer qual é o ruído antes de interpretar diferenças. Neste sistema o
ruído de seed é cerca de 0.005 de Dice, portanto qualquer diferença abaixo disso
não é resultado, é acaso.
