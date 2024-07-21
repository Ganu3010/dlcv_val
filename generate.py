from renderer import * 
from main import sample_Unet_DDPM

if __name__=='__main__':

    model_path = 'model_1K_8e-6/unet_800.pt'
    sample_dir = 'model_1K_8e-5/samples/'

    sample_Unet_DDPM(model_path = model_path, output_path=sample_dir, batch_size = 4)

    for kps in os.listdir(sample_dir):
        key_points = load_keypoints(os.path.join(sample_dir, kps))
        save_path = os.path.join(sample_dir, kps.split('.')[0]+'.gif')
        render_seq(key_points, save_path)

    print('Samples saved at: ', sample_dir)