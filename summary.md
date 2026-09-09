# Resumo do projeto — Segmentação da cabeça fetal em ultrassom 3D

## Como funciona, em duas fases

Fase 1 — localização grosseira. Para cada caso calcula-se, a partir do
label, o SDF em resolução nativa e faz-se downsample da imagem e do SDF para
um volume pequeno (32×32×32), uma espécie de shell. Estas shells são o dataset que treina o modelo,
a rede aprende a prever o SDF (32³) a partir da shell fornecida reduzida (32³). 
Dá uma estimativa rápida e barata de onde está a
cabeça, mas com pouco detalhe.

A shell depois passa pela rede da Fase 1, e a previsão
do SDF a 32³ é depois upsampled ( com interpolação trilinear) de volta para a
resolução nativa do caso — dá um SDF previsto voxel a voxel, à escala
original. O sinal desse SDF dá logo o dentro/fora do contorno.
Define-se uma banda estreita à volta do zero (o contorno
previsto — onde a Fase 1 está menos segura) e recorta-se, em resolução
nativa, a zona à volta dessa banda: é este recorte ("crop") que passa para a
Fase 2, não o volume nativo inteiro.

Fase 2 — refinamento local. Dentro do crop nativo, a segunda rede treina e
prevê em patches pequenos (64³) amostrados só na banda, usando dois canais de
entrada: a imagem original e o SDF upsampled da Fase 1. A rede decide a
máscara binária final apenas nesses patches; fora da banda mantém-se a
resposta da Fase 1, sem gastar computação onde não é preciso.


## O que foi testado

Foi feita uma pesquisa alargada de hiperparâmetros — **41 configurações
diferentes** — para escolher a melhor combinação de:
- função de perda (*loss*) usada em cada fase;
- taxa de aprendizagem;
- duas técnicas extra: uma perda que dá mais peso aos erros perto da
  fronteira do que aos erros longe dela, e outra perda que compara a
  forma/inclinação da superfície prevista com a real (não só os valores
  ponto a ponto).

A pesquisa foi feita em duas etapas, para ser mais barata: primeiro escolheu-se
a melhor configuração para a Fase 1 e fixou-se essa escolha; só depois se
procurou a melhor configuração para a Fase 2 em cima dela. No fim, testaram-se
também algumas combinações cruzadas entre as melhores opções de cada fase,
para confirmar que não havia nenhuma combinação escondida claramente melhor —
não havia.

## Resultados da pesquisa inicial

**Fase 1** — melhor configuração encontrada: perda L1 combinada com a perda de
forma/gradiente.
- Dice: **0.949**
- Erro médio absoluto (MAE): **0.036**

(A perda que dá mais peso à fronteira foi testada isoladamente na Fase 1 e
**não ajudou** — piorou ligeiramente o resultado. Foi reportado como resultado
negativo, não escondido.)

**Fase 2** (construída em cima da Fase 1 escolhida) — melhor configuração
encontrada: perda "Dice+BCE" combinada com um pequeno peso extra na zona da
fronteira.
- Dice da máscara final, em resolução original: **0.951**
- HD95 (distância de Hausdorff a 95%): **2.86 mm**
- ASD (distância média de superfície): **0.95 mm**

Para comparação, a Fase 1 sozinha (sem o refinamento da Fase 2) dá Dice =
0.940 — o que confirma que a Fase 2 traz um ganho real, e não é apenas ruído.

## Validação em curso

Os resultados acima vêm de **uma única run de treino cada**, sem semente
(*seed*) fixa. Já se confirmou que só a aleatoriedade da inicialização do
modelo pode mover o Dice entre ~0.5 e 1 ponto percentual entre duas runs
"idênticas" — ou seja, não se pode tirar conclusões definitivas de um único
treino.

Por isso, as **3 melhores configurações** encontradas estão agora a ser
validadas a sério: cada uma está a ser treinada do zero 5 vezes (validação
cruzada em 5 partições, ou "folds") × 3 vezes com sementes diferentes = **15
treinos completos por configuração, 45 no total**. O resultado final será a
média e o desvio-padrão entre estas 15 repetições, não um número isolado —
isto é o que vai decidir qual configuração é realmente a melhor.

**Estado atual: os 45 treinos estão a correr no cluster.** Esta secção será
atualizada com os resultados finais assim que a validação terminar.

## Eficiência

Números medidos para a configuração vencedora (Fase 1: L1+gradiente; Fase 2:
Dice+BCE ponderada), treino em 72 casos / validação em 18 casos, numa única
GPU:

| | Nº parâmetros | Inferência | VRAM treino | VRAM inferência |
|---|---|---|---|---|
| Fase 1 | 1.40 M | ~16 ms/caso | 696 MB | 524 MB |
| Fase 2 | 1.40 M | ~1.01 s/caso | 3.5 GB | 1.8 GB |
| **Total** | **2.80 M** | **~1.03 s/caso** | — | — |

*(Inferência = só o forward pass do modelo + junção dos tiles, sem contar o
cálculo de HD95/ASD contra o ground truth — esse é um custo só de validação,
que em produção não existe porque não há máscara real para comparar. Número
preliminar, fp32, sem warmup — a medição final com warmup+repetições fica
para depois do Estágio C.)*

- O modelo todo tem menos de 3 milhões de parâmetros (~10.7 MB entre as duas
  fases) — muito leve para uma rede 3D.
- Inferência por caso demora **~1.0 segundo**, quase todo gasto na Fase 2
  (resolução nativa, por tiles, só na banda); a Fase 1 (32³, volume inteiro)
  é praticamente instantânea (~16 ms).
- A Fase 2 pesa mais em VRAM (3.5 GB a treinar) do que a Fase 1 (<1 GB) por
  operar em patches de resolução nativa em vez do volume 32³ reduzido — é o
  preço do detalhe extra, mas ainda assim ligeiro para uma rede 3D.
