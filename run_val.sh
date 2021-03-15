CUDA_VISIBLE_DEVICES=1 python main_pky.py \
--version 'v1.0-trainval' --data_type 'real' \
--ploss_type 'map' \
--beta 0.1 --batch_size 32 \
--test_times 2 \
--test_ckpt 'experiment/0309_AttTest__09_March__01_21_/ck_91_-12.0503_57.8873_0.7136_1.5714.pth.tar' \
--test_dir 'results' \
--load_dir='../nus_dataset'