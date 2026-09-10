# Experimento: loss / lr / coeficientes por fase, validado com k-fold

Todas as tabelas abaixo são geradas por `python -m tools.aggregate_results` a partir de
`profile/config.json`, `profile/summary.json` e `eval_val/summary.json` já gravados por
cada run — sem logging novo de qualidade. As secções entre marcadores `<!-- ..._START -->`
/ `<!-- ..._END -->` são regeneradas a cada `python -m tools.aggregate_results`; o resto deste ficheiro
é escrito à mão e sobrevive a essas regenerações.

## Desenho

Dois estágios: **A** (sweep largo, uma run por ponto, sem k-fold nem seeds) escolhe os
finalistas; **C** (5-fold × 3 seeds, só nos 3 finalistas) valida a decisão a sério. Ver
`/home/users/2ai12_1/.claude/plans/reactive-munching-thompson.md` para o racional completo
e a aritmética de custo.

- Fase 1 fixa: 250 épocas. Fase 2 fixa: 100 épocas. `trunc-mm=10.0`, `band-mm=5.0`.
- Losses BCE (`bce`/`dice_bce`/`mse_bce`) excluídas do grid da fase 1 — saída `tanh`
  ([-1,1]) incompatível, `train.py` recusa-as agora com erro claro em vez de crash CUDA.
- Folds independentes (seed própria, não os do projeto survey).

## Glossário das tabelas

### Nomes dos runs

Cada `run_id` começa pelo estágio que o gerou. `a1`, `a1b`, `a1c`, `a1d` e `a3`
treinam só a fase 1 ou fazem a comparação fase1×fase2; `a2` e `a2b` treinam a
fase 2 completa em cima da fase 1 vencedora. Os finalistas do estágio C usam
nomes próprios, `f1_bw`, `f2_plain`, `f3_mse`, explicados na secção do
Estágio B.

### Colunas comuns

- `status`: `ok` se o resultado de avaliação foi encontrado no disco, `MISSING`
  caso contrário.
- `loss_p1` / `loss_p2`: a loss principal usada na fase 1 ou na fase 2. Ver
  "Losses" abaixo para o que cada uma faz.
- `lr_p1` / `lr_p2`: learning rate dessa fase.
- `train_wall_s`: tempo total de treino em segundos.
- `train_vram_mb`: pico de VRAM ocupado durante o treino, em MB.
- `params_mb`: memória ocupada pelos pesos do modelo, em MB, não o número de
  parâmetros.

### Colunas só da fase 1

- `dice` e `mae`: qualidade da fase 1 isolada. A fase 1 prevê um campo de
  distância com sinal, `mae` mede o erro absoluto médio desse campo. `dice` é
  o Dice do contorno obtido ao binarizar esse campo em zero.

### Colunas só da fase 2

- `dice_coarse`: Dice da máscara da fase 1 sozinha, já com upsample para a
  resolução final. É a base que a fase 2 tem de bater.
- `dice_refined`, `hd95_refined`, `asd_refined`: Dice, Hausdorff 95 (mm) e
  distância de superfície média (mm) depois da fase 2 reescrever a banda à
  volta do contorno. `hd95_refined` e `asd_refined` só existem depois da fase
  2 porque a fase 1 sozinha não passa pelo cálculo de distância de superfície.

### Boundary-weight (`bw_floor`, `bw_coef`)

O boundary-weight dá mais peso aos voxels perto do contorno durante o treino.
Funciona em dois passos. Primeiro, `bw_floor` define o peso dos voxels longe
do contorno: com `bw_floor=1.0` todos os voxels pesam o mesmo e a ponderação
está desligada, esse é o valor por omissão; com `bw_floor=0.2`, por exemplo,
os voxels longe do contorno passam a pesar só 20% do que pesam os voxels
perto dele. Depois, esse termo ponderado é somado à loss normal, multiplicado
por `bw_coef`. Por isso a loss final fica `loss_base + bw_coef * loss_ponderada`,
e só faz sentido olhar para `bw_coef` quando `bw_floor` é diferente de 1.0.

### Gradient-difference (`gd_coef`)

Só existe na fase 1. Compara a inclinação local do campo previsto com a do
campo real em vez de comparar os valores diretamente, e essa inclinação é
maior exactamente perto da superfície, onde o campo de distância muda mais
depressa. Também é somado à loss normal, multiplicado por `gd_coef`. Com
`gd_coef=0.0` está desligado.

### Losses

- `l1`: erro absoluto médio entre a previsão e o alvo. Usada na fase 1.
- `mse`: erro quadrático médio.
- `bce`: binary cross-entropy, precisa de valores entre 0 e 1, por isso só é
  usada na fase 2.
- `dice`: 1 menos o coeficiente de Dice entre previsão e alvo.
- `dice_bce`: soma de `dice` com `bce`.
- `mse_bce`: soma de `mse` com `bce`.

`bce`, `dice_bce` e `mse_bce` ficam de fora do grid da fase 1 porque a fase 1
prevê valores entre -1 e 1 e essas três losses só aceitam valores entre 0 e 1.

## Estágio A — sweep largo

<!-- STAGE_A1_START -->
### Estágio A1 — fase 1: loss × lr (sem ponderação)

| run_id | status | loss_p1 | lr_p1 | bw_floor_p1 | bw_coef_p1 | gd_coef_p1 | dice | mae | train_wall_s | train_vram_mb | params_mb |
|---|---|---|---|---|---|---|---|---|---|---|---|
| a1_l1_lr0.0002 | ok | l1 | 0.0002 | 1.0 | 1.0 | 0.0 | 0.9477 | 0.0393 | 158.17 | 694.00 | 5.35 |
| a1_l1_lr0.0005 | ok | l1 | 0.0005 | 1.0 | 1.0 | 0.0 | 0.9477 | 0.0367 | 156.20 | 694.00 | 5.35 |
| a1_l1_lr0.001 | ok | l1 | 0.001 | 1.0 | 1.0 | 0.0 | 0.9456 | 0.0371 | 127.91 | 694.00 | 5.35 |
| a1_mse_lr0.0005 | ok | mse | 0.0005 | 1.0 | 1.0 | 0.0 | 0.9445 | 0.0503 | 148.38 | 694.00 | 5.35 |
| a1_mse_lr0.0002 | ok | mse | 0.0002 | 1.0 | 1.0 | 0.0 | 0.9424 | 0.0732 | 149.36 | 694.00 | 5.35 |
| a1_mse_lr0.001 | ok | mse | 0.001 | 1.0 | 1.0 | 0.0 | 0.9390 | 0.0480 | 149.32 | 694.00 | 5.35 |
<!-- STAGE_A1_END -->

<!-- STAGE_A1B_START -->
### Estágio A1b — fase 1: perda ponderada por fronteira (floor × coef)

| run_id | status | loss_p1 | lr_p1 | bw_floor_p1 | bw_coef_p1 | gd_coef_p1 | dice | mae | train_wall_s | train_vram_mb | params_mb |
|---|---|---|---|---|---|---|---|---|---|---|---|
| a1b_f0.2_c0.5 | ok | l1 | 0.0005 | 0.2 | 0.5 | 0.0 | 0.9464 | 0.0383 | 153.55 | 694.00 | 5.35 |
| a1b_f0.2_c1.0 | ok | l1 | 0.0005 | 0.2 | 1.0 | 0.0 | 0.9463 | 0.0375 | 155.59 | 694.00 | 5.35 |
| a1b_f0.1_c1.0 | ok | l1 | 0.0005 | 0.1 | 1.0 | 0.0 | 0.9447 | 0.0403 | 153.83 | 694.00 | 5.35 |
| a1b_f0.1_c0.5 | ok | l1 | 0.0005 | 0.1 | 0.5 | 0.0 | 0.9436 | 0.0393 | 134.84 | 694.00 | 5.35 |
<!-- STAGE_A1B_END -->

<!-- STAGE_A1C_START -->
### Estágio A1c — fase 1: perda de gradiente (gd-coef)

| run_id | status | loss_p1 | lr_p1 | bw_floor_p1 | bw_coef_p1 | gd_coef_p1 | dice | mae | train_wall_s | train_vram_mb | params_mb |
|---|---|---|---|---|---|---|---|---|---|---|---|
| a1c_gd0.1 | ok | l1 | 0.0005 | 1.0 | 1.0 | 0.1 | 0.9494 | 0.0362 | 156.20 | 696.00 | 5.35 |
| a1c_gd0.2 | ok | l1 | 0.0005 | 1.0 | 1.0 | 0.2 | 0.9481 | 0.0367 | 156.21 | 696.00 | 5.35 |
| a1c_gd0.05 | ok | l1 | 0.0005 | 1.0 | 1.0 | 0.05 | 0.9444 | 0.0393 | 158.65 | 696.00 | 5.35 |
<!-- STAGE_A1C_END -->

<!-- STAGE_A1D_START -->
### Estágio A1d — fase 1: ponderação + gradiente combinados

| run_id | status | loss_p1 | lr_p1 | bw_floor_p1 | bw_coef_p1 | gd_coef_p1 | dice | mae | train_wall_s | train_vram_mb | params_mb |
|---|---|---|---|---|---|---|---|---|---|---|---|
| a1d_combined | ok | l1 | 0.0005 | 0.2 | 0.5 | 0.1 | 0.9461 | 0.0386 | 156.37 | 696.00 | 5.35 |
<!-- STAGE_A1D_END -->

<!-- STAGE_A2_START -->
### Estágio A2 — fase 2: loss × lr (sem ponderação)

| run_id | status | loss_p2 | lr_p2 | bw_floor_p2 | bw_coef_p2 | dice_coarse | dice_refined | hd95_refined | asd_refined | train_wall_s | train_vram_mb | params_mb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a2_dice_bce_lr0.0003 | ok | dice_bce | 0.0003 | 1.0 | 1.0 | 0.9395 | 0.9507 | 2.8415 | 0.9547 | 4115.16 | 3532.00 | 5.35 |
| a2_mse_lr0.0005 | ok | mse | 0.0005 | 1.0 | 1.0 | 0.9395 | 0.9506 | 2.8376 | 0.9599 | 3586.00 | 3532.00 | 5.35 |
| a2_mse_lr0.001 | ok | mse | 0.001 | 1.0 | 1.0 | 0.9395 | 0.9504 | 2.8814 | 0.9631 | 3281.16 | 3532.00 | 5.35 |
| a2_dice_lr0.0005 | ok | dice | 0.0005 | 1.0 | 1.0 | 0.9395 | 0.9504 | 2.8509 | 0.9643 | 4446.99 | 3532.00 | 5.35 |
| a2_l1_lr0.0005 | ok | l1 | 0.0005 | 1.0 | 1.0 | 0.9395 | 0.9504 | 2.8691 | 0.9648 | 3062.55 | 3532.00 | 5.35 |
| a2_dice_lr0.0003 | ok | dice | 0.0003 | 1.0 | 1.0 | 0.9395 | 0.9503 | 2.8547 | 0.9649 | 3994.70 | 3532.00 | 5.35 |
| a2_dice_bce_lr0.0005 | ok | dice_bce | 0.0005 | 1.0 | 1.0 | 0.9395 | 0.9503 | 2.8761 | 0.9649 | 3905.08 | 3532.00 | 5.35 |
| a2_mse_bce_lr0.001 | ok | mse_bce | 0.001 | 1.0 | 1.0 | 0.9395 | 0.9503 | 2.8737 | 0.9654 | 4237.38 | 3532.00 | 5.35 |
| a2_bce_lr0.0003 | ok | bce | 0.0003 | 1.0 | 1.0 | 0.9395 | 0.9503 | 2.8745 | 0.9659 | 3504.66 | 3532.00 | 5.35 |
| a2_mse_bce_lr0.0005 | ok | mse_bce | 0.0005 | 1.0 | 1.0 | 0.9395 | 0.9502 | 2.8896 | 0.9668 | 3544.34 | 3532.00 | 5.35 |
| a2_mse_bce_lr0.0003 | ok | mse_bce | 0.0003 | 1.0 | 1.0 | 0.9395 | 0.9502 | 2.8631 | 0.9665 | 4544.59 | 3532.00 | 5.35 |
| a2_dice_lr0.001 | ok | dice | 0.001 | 1.0 | 1.0 | 0.9395 | 0.9502 | 2.8735 | 0.9680 | 4035.96 | 3532.00 | 5.35 |
| a2_l1_lr0.001 | ok | l1 | 0.001 | 1.0 | 1.0 | 0.9395 | 0.9501 | 2.8560 | 0.9713 | 4048.00 | 3532.00 | 5.35 |
| a2_bce_lr0.001 | ok | bce | 0.001 | 1.0 | 1.0 | 0.9395 | 0.9501 | 2.8735 | 0.9691 | 3347.17 | 3532.00 | 5.35 |
| a2_dice_bce_lr0.001 | ok | dice_bce | 0.001 | 1.0 | 1.0 | 0.9395 | 0.9500 | 2.8452 | 0.9697 | 4318.03 | 3532.00 | 5.35 |
| a2_bce_lr0.0005 | ok | bce | 0.0005 | 1.0 | 1.0 | 0.9395 | 0.9500 | 2.8779 | 0.9691 | 4330.27 | 3532.00 | 5.35 |
| a2_mse_lr0.0003 | ok | mse | 0.0003 | 1.0 | 1.0 | 0.9395 | 0.9499 | 2.8509 | 0.9696 | 3834.40 | 3532.00 | 5.35 |
| a2_l1_lr0.0003 | ok | l1 | 0.0003 | 1.0 | 1.0 | 0.9395 | 0.9497 | 2.8830 | 0.9788 | 4263.73 | 3532.00 | 5.35 |
<!-- STAGE_A2_END -->

<!-- STAGE_A2B_START -->
### Estágio A2b — fase 2: perda ponderada por fronteira (floor × coef)

| run_id | status | loss_p2 | lr_p2 | bw_floor_p2 | bw_coef_p2 | dice_coarse | dice_refined | hd95_refined | asd_refined | train_wall_s | train_vram_mb | params_mb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a2b_f0.2_c1.0 | ok | dice_bce | 0.0003 | 0.2 | 1.0 | 0.9395 | 0.9508 | 2.8600 | 0.9530 | 2718.02 | 3532.00 | 5.35 |
| a2b_f0.1_c1.0 | ok | dice_bce | 0.0003 | 0.1 | 1.0 | 0.9395 | 0.9506 | 2.8444 | 0.9583 | 2721.65 | 3532.00 | 5.35 |
| a2b_f0.2_c0.5 | ok | dice_bce | 0.0003 | 0.2 | 0.5 | 0.9395 | 0.9502 | 2.8898 | 0.9646 | 2742.13 | 3532.00 | 5.35 |
| a2b_f0.4_c0.5 | ok | dice_bce | 0.0003 | 0.4 | 0.5 | 0.9395 | 0.9502 | 2.8636 | 0.9663 | 2203.53 | 3532.00 | 5.35 |
| a2b_f0.1_c0.5 | ok | dice_bce | 0.0003 | 0.1 | 0.5 | 0.9395 | 0.9501 | 2.8743 | 0.9672 | 2694.51 | 3532.00 | 5.35 |
| a2b_f0.4_c1.0 | ok | dice_bce | 0.0003 | 0.4 | 1.0 | 0.9395 | 0.9495 | 2.8734 | 0.9772 | 2658.70 | 3532.00 | 5.35 |
<!-- STAGE_A2B_END -->

<!-- STAGE_A3_START -->
### Stage A3 -- interaction check (top-2 phase1 x top-2 phase2)

| run_id | status | loss_p2 | lr_p2 | bw_floor_p2 | bw_coef_p2 | dice_coarse | dice_refined | hd95_refined | asd_refined | train_wall_s | train_vram_mb | params_mb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a3_1 | ok | dice_bce | 0.0003 | 0.2 | 1.0 | 0.9323 | 0.9482 | 2.8802 | 1.0023 | 2118.12 | 3532.00 | 5.35 |
| a3_2 | ok | dice_bce | 0.0003 | 0.1 | 1.0 | 0.9318 | 0.9469 | 3.1051 | 1.0346 | 2078.95 | 3532.00 | 5.35 |
| a3_0 | ok | dice_bce | 0.0003 | 0.1 | 1.0 | 0.9339 | 0.9462 | 3.0743 | 1.0405 | 2155.08 | 3532.00 | 5.35 |
<!-- STAGE_A3_END -->

## Estágio B — finalistas

Regra aplicada (do plano): ordenar todas as combinações fase1×fase2 avaliadas
end-to-end (A2 + A2b + A3) por Dice-refined médio; dentro do ruído de seed já medido
(~0.005) desempatar por HD95 mais baixo. Nenhuma config alternativa de fase 1 testada no
A3 (`gd_coef=0.2`) chega perto do topo (máx. 0.9482 vs ~0.950+), por isso os 3 finalistas
partilham a fase-1 vencedora (`l1, lr=5e-4, gd_coef=0.1`, Dice=0.9494/MAE=0.0362 na
métrica própria da fase 1) e variam só a fase 2:

| # | run_id origem | loss_p2 | lr_p2 | bw_floor_p2 | bw_coef_p2 | dice_refined | hd95_refined | Porquê |
|---|---|---|---|---|---|---|---|---|
| 1 | a2b_f0.2_c1.0 | dice_bce | 3e-4 | 0.2 | 1.0 | 0.9508 | 2.8600 | Vencedor mecânico da regra |
| 2 | a2_dice_bce_lr0.0003 | dice_bce | 3e-4 | 1.0 (off) | - | 0.9507 | 2.8415 | Mesma loss/lr sem ponderação — testa se o boundary-weight é real ou ruído |
| 3 | a2_mse_lr0.0005 | mse | 5e-4 | 1.0 (off) | - | 0.9506 | 2.8376 | Família de loss diferente, menor HD95/ASD do estágio A — testa se a loss importa |

Os 3 estão dentro de 0.0002 de Dice um do outro — abaixo do ruído de seed medido
(~0.005). O estágio C existe precisamente para decidir se esta ordem se mantém com
seeds e folds fixos, ou se é só ruído.

## Estágio C — validação k-fold

<!-- STAGE_C_START -->
Seeds 42 e 43 completas (5 folds x 2 seeds x 3 finalistas = 30 runs). A seed 44
falhou a meio (ver nota abaixo) e ficou-se pelas 2 seeds -- decisão do
utilizador, não um problema de dados.

### Dice refinado por fold (média entre as 2 seeds)

| Fold | f1_bw | f2_plain | f3_mse |
|---|---|---|---|
| fold0 | 0.9408 ± 0.0013 | 0.9409 ± 0.0015 | 0.9407 ± 0.0013 |
| fold1 | 0.9492 ± 0.0007 | 0.9494 ± 0.0006 | 0.9494 ± 0.0005 |
| fold2 | 0.9501 ± 0.0009 | 0.9502 ± 0.0005 | 0.9501 ± 0.0010 |
| fold3 | 0.9469 ± 0.0012 | 0.9468 ± 0.0010 | 0.9466 ± 0.0010 |
| fold4 | 0.9411 ± 0.0008 | 0.9411 ± 0.0009 | 0.9411 ± 0.0012 |

### Dice refinado por seed (média entre os 5 folds)

| Seed | f1_bw | f2_plain | f3_mse |
|---|---|---|---|
| seed42 | 0.9450 ± 0.0045 | 0.9450 ± 0.0045 | 0.9450 ± 0.0047 |
| seed43 | 0.9462 ± 0.0035 | 0.9464 ± 0.0035 | 0.9462 ± 0.0034 |

### Global (10 runs por modelo: 5 folds x 2 seeds)

| Modelo | Dice | HD95 (mm) | ASD (mm) |
|---|---|---|---|
| f1_bw | 0.9456 ± 0.0041 | 3.196 ± 0.333 | 1.0648 ± 0.0917 |
| f2_plain | 0.9457 ± 0.0041 | 3.202 ± 0.332 | 1.0627 ± 0.0917 |
| f3_mse | 0.9456 ± 0.0041 | 3.198 ± 0.338 | 1.0656 ± 0.0925 |

**Leitura:** os 3 finalistas continuam praticamente empatados (diferença de
Dice na 3ª/4ª casa decimal, dentro do próprio ruído de seed). O que domina a
variância não é a escolha do modelo -- é o fold: fold0 e fold4 dão sempre
~0.941 (mais difíceis, incluem `44_I0000039`), fold1-3 dão ~0.947-0.950,
independentemente de qual dos 3 finalistas é usado. A decisão entre f1_bw /
f2_plain / f3_mse não parece ter impacto real nos resultados finais.

*(Seed 44 e as configs perdedoras do Estágio A foram removidas do disco após
gravados os números aqui e em `summary.md` -- ver limpeza de 2026-09-09.)*
<!-- STAGE_C_END -->

## Decisão final

*(preencher depois do estágio C)*

## Reprodução

*(comandos exatos, uma vez fechado)*
