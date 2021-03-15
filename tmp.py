from pkyutils import NusToolkit
    
toolkit = NusToolkit(root='/home/vilab/daehee/nus_dataset/original_small/v1.0-trainval', version='v1.0-trainval', load_dir='../nus_dataset', split='train_val')
toolkit.save_dataset()