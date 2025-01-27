#!/bin/sh
env="SingleCombat"
# scenario="1v1/NoWeapon/Selfplay"
scenario="1v1/NoWeapon/test/test_pursue"
algo="ppo"
exp="v1"
seed=1

prev_l=""

    # --render-idx 124 --render-opponent-idx 125 --model-dir results/SingleCombat/1v1/NoWeapon/Selfplay/ppo/v1/run6 \
echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
CUDA_VISIBLE_DEVICES=1 python render/render_jsbsim.py \
    --env-name ${env} --algorithm-name ${algo} --scenario-name ${scenario} --experiment-name ${exp} \
    --seed ${seed} --n-training-threads 1 --n-rollout-threads 1 --cuda --log-interval 1 --save-interval 1 \
    --use-selfplay --selfplay-algorithm "fsp" --n-choose-opponents 1 \
    --use-eval --n-eval-rollout-threads 1 --eval-interval 1 --eval-episodes 1 \
    --num-mini-batch 5 --buffer-size 3000 --num-env-steps 1e8 \
    --lr 3e-4 --gamma 0.99 --ppo-epoch 4 --clip-params 0.2 --max-grad-norm 2 --entropy-coef 1e-3 \
    --hidden-size "128 128" --act-hidden-size "128 128" --recurrent-hidden-size 128 --recurrent-hidden-layers 1 --data-chunk-length 8 \
    --render-idx 124 --render-opponent-idx 125 --model-dir results/SingleControl/1/heading/ppo/v1/full \
    2>&1 |
while read line
do
    if [[ "$line" =~ .*f16.xml.*|Engine.* ]] || [[ $line = $prev_l ]]; then
        continue
    fi
    prev_l=$line
    echo $line
done 