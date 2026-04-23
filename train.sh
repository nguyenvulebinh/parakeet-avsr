export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node=8 training.py \
    --model-id "nguyenvulebinh/parakeet-avsr" \
    --train_files data/train.jsonl,data/avsr_en_vox.jsonl \
    --eval_files data/test.jsonl \
    --batch-size 8 \
    --model-id nguyenvulebinh/parakeet-avsr \
    --output-dir model-bin/test \
    --freeze-vis-feat 1 \
    --learning-rate 2e-4 \
    --custom-tokenizer 0 \
    --custom-tokenizer-path vocab/vi_en_vocab.model \
    --save-steps 5000 \
    --eval-steps 5000 \
