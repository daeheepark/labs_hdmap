CUDA_VISIBLE_DEVICES=0 python main_pky.py \
--model_type 'AttGlobal_Scene_CAM_NFDecoder' \
--version 'v1.0-trainval' --data_type 'real' \
--ploss_type 'map' \
--beta 0.1 --batch_size 32 --gpu_devices 0 \
--test_times 3 \
--test_ckpt 'experiment/Transfer__27_January__08_08_/ck_74_-4.8840_13.0271_0.2140_0.4126.pth.tar' \
--test_dir 'results'