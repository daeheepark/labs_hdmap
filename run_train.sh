python3 main_pky.py \
--tag='Global_Scene_CAM_NFDecoder' --model_type='Global_Scene_CAM_NFDecoder' \
--dataset='nuscenes' --batch_size=4 --num_epochs=100 --gpu_devices=0 --agent_embed_dim=128 \
--train_cache "./data/nuscenes_train_cache.pkl" --val_cache "./data/nuscenes_val_cache.pkl" \
--num_candidates=6 --map_version '2.0' 
