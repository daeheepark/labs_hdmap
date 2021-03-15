CUDA_VISIBLE_DEVICES=3 python main_pky.py --learning_rate=1e-6 \
--tag 'Transfer' --model_type 'AttGlobal_Scene_CAM_NFDecoder' \
--version 'v1.0-trainval' --data_type 'real' \
--batch_size 4 --num_epochs 100 --gpu_devices 0 --agent_embed_dim 128 \
--ploss_type 'map' \
--beta 0.1 --load_ckpt "./experiment/ck_45_-4.6663_12.8040_0.6640_1.5014.pth.tar"