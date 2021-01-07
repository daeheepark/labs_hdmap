python3 main.py \
--tag='Global_Scene_CAM_NFDecoder' --model_type='Global_Scene_CAM_NFDecoder' --dataset='nuscenes' --batch_size=100 --num_epochs=100 --gpu_devices=0 --agent_embed_dim=128 --train_cache "./data/aug/nuscene_0.02_0.3.pickle_train" --val_cache "./data/aug/nuscene_0.02_0.3.pickle_val" --num_candidates=6 --map_version '2.0'
