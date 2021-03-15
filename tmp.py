from pkyutils import NusToolkit
    
toolkit = NusToolkit(root='/home/vilab/daehee/nus_dataset/original_small/v1.0-mini', version='v1.0-mini', load_dir='../nus_dataset')
toolkit.save_dataset()