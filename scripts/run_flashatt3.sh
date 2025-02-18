export USE_FLASH_ATTENTION3=0
export CUDA_VISIBLE_DEVICES=0
python test_t2v.py --config configs/test/4_step_t2v.yaml --quantization False