# FID SCORES (pretrained)
# python FID/fid_score.py generated_data/original_images/ generated_data/fsvae/0 --gpu 0 >> fid_scores/fsvae_1.txt

# python FID/fid_score.py generated_data/original_images/ generated_data/wgan/1 --gpu 0 >> fid_scores/wgan_1.txt
# python FID/fid_score.py generated_data/original_images/ generated_data/wgan/10 --gpu 0 >> fid_scores/wgan_10.txt
# python FID/fid_score.py generated_data/original_images/ generated_data/wgan/20 --gpu 0 >> fid_scores/wgan_20.txt
# python FID/fid_score.py generated_data/original_images/ generated_data/wgan/30 --gpu 0 >> fid_scores/wgan_30.txt
# python FID/fid_score.py generated_data/original_images/ generated_data/wgan/40 --gpu 0 >> fid_scores/wgan_40.txt
# python FID/fid_score.py generated_data/original_images/ generated_data/wgan/50 --gpu 0 >> fid_scores/wgan_50.txt

# python FID/fid_score.py generated_data/original_images/ generated_data/spectral/1 --gpu 0 >> fid_scores/spectral_1.txt
# python FID/fid_score.py generated_data/original_images/ generated_data/spectral/10 --gpu 0 >> fid_scores/spectral_10.txt
# python FID/fid_score.py generated_data/original_images/ generated_data/spectral/20 --gpu 0 >> fid_scores/spectral_20.txt
# python FID/fid_score.py generated_data/original_images/ generated_data/spectral/30 --gpu 0 >> fid_scores/spectral_30.txt
# python FID/fid_score.py generated_data/original_images/ generated_data/spectral/40 --gpu 0 >> fid_scores/spectral_40.txt
# python FID/fid_score.py generated_data/original_images/ generated_data/spectral/50 --gpu 0 >> fid_scores/spectral_50.txt

# python FID/fid_score.py generated_data/original_images/ generated_data/c_wgan/1 --gpu 0 >> fid_scores/c_wgan_1.txt
# python FID/fid_score.py generated_data/original_images/ generated_data/c_wgan/10 --gpu 0 >> fid_scores/c_wgan_10.txt
# python FID/fid_score.py generated_data/original_images/ generated_data/c_wgan/20 --gpu 0 >> fid_scores/c_wgan_20.txt
# python FID/fid_score.py generated_data/original_images/ generated_data/c_wgan/30 --gpu 0 >> fid_scores/c_wgan_30.txt
# python FID/fid_score.py generated_data/original_images/ generated_data/c_wgan/40 --gpu 0 >> fid_scores/c_wgan_40.txt
# python FID/fid_score.py generated_data/original_images/ generated_data/c_wgan/50 --gpu 0 >> fid_scores/c_wgan_50.txt

python FID/fid_score.py generated_data/original_images/ generated_data/c_spectral/1 --gpu 0 >> fid_scores/c_spectral_1.txt
python FID/fid_score.py generated_data/original_images/ generated_data/c_spectral/10 --gpu 0 >> fid_scores/c_spectral_10.txt
python FID/fid_score.py generated_data/original_images/ generated_data/c_spectral/20 --gpu 0 >> fid_scores/c_spectral_20.txt
python FID/fid_score.py generated_data/original_images/ generated_data/c_spectral/30 --gpu 0 >> fid_scores/c_spectral_30.txt
python FID/fid_score.py generated_data/original_images/ generated_data/c_spectral/40 --gpu 0 >> fid_scores/c_spectral_40.txt
python FID/fid_score.py generated_data/original_images/ generated_data/c_spectral/50 --gpu 0 >> fid_scores/c_spectral_50.txt

# FID SCORES (retrained on Fashion MNIST)

python FID/fid_score.py generated_data/original_images/ generated_data/fsvae/0 --gpu 0 --retrained=True >> fid_scores/retrained_fsvae_1.txt

python FID/fid_score.py generated_data/original_images/ generated_data/wgan/1 --gpu 0 --retrained=True >> fid_scores/retrained_wgan_1.txt
python FID/fid_score.py generated_data/original_images/ generated_data/wgan/10 --gpu 0 --retrained=True >> fid_scores/retrained_wgan_10.txt
python FID/fid_score.py generated_data/original_images/ generated_data/wgan/20 --gpu 0 --retrained=True >> fid_scores/retrained_wgan_20.txt
python FID/fid_score.py generated_data/original_images/ generated_data/wgan/30 --gpu 0 --retrained=True >> fid_scores/retrained_wgan_30.txt
python FID/fid_score.py generated_data/original_images/ generated_data/wgan/40 --gpu 0 --retrained=True >> fid_scores/retrained_wgan_40.txt
python FID/fid_score.py generated_data/original_images/ generated_data/wgan/50 --gpu 0 --retrained=True >> fid_scores/retrained_wgan_50.txt

python FID/fid_score.py generated_data/original_images/ generated_data/spectral/1 --gpu 0 --retrained=True >> fid_scores/retrained_spectral_1.txt
python FID/fid_score.py generated_data/original_images/ generated_data/spectral/10 --gpu 0 --retrained=True >> fid_scores/retrained_spectral_10.txt
python FID/fid_score.py generated_data/original_images/ generated_data/spectral/20 --gpu 0 --retrained=True >> fid_scores/retrained_spectral_20.txt
python FID/fid_score.py generated_data/original_images/ generated_data/spectral/30 --gpu 0 --retrained=True >> fid_scores/retrained_spectral_30.txt
python FID/fid_score.py generated_data/original_images/ generated_data/spectral/40 --gpu 0 --retrained=True >> fid_scores/retrained_spectral_40.txt
python FID/fid_score.py generated_data/original_images/ generated_data/spectral/50 --gpu 0 --retrained=True >> fid_scores/retrained_spectral_50.txt

python FID/fid_score.py generated_data/original_images/ generated_data/c_wgan/1 --gpu 0 --retrained=True >> fid_scores/retrained_c_wgan_1.txt
python FID/fid_score.py generated_data/original_images/ generated_data/c_wgan/10 --gpu 0 --retrained=True >> fid_scores/retrained_c_wgan_10.txt
python FID/fid_score.py generated_data/original_images/ generated_data/c_wgan/20 --gpu 0 --retrained=True >> fid_scores/retrained_c_wgan_20.txt
python FID/fid_score.py generated_data/original_images/ generated_data/c_wgan/30 --gpu 0 --retrained=True >> fid_scores/retrained_c_wgan_30.txt
python FID/fid_score.py generated_data/original_images/ generated_data/c_wgan/40 --gpu 0 --retrained=True >> fid_scores/retrained_c_wgan_40.txt
python FID/fid_score.py generated_data/original_images/ generated_data/c_wgan/50 --gpu 0 --retrained=True >> fid_scores/retrained_c_wgan_50.txt

python FID/fid_score.py generated_data/original_images/ generated_data/c_spectral/1 --gpu 0 --retrained=True >> fid_scores/retrained_c_spectral_1.txt
python FID/fid_score.py generated_data/original_images/ generated_data/c_spectral/10 --gpu 0 --retrained=True >> fid_scores/retrained_c_spectral_10.txt
python FID/fid_score.py generated_data/original_images/ generated_data/c_spectral/20 --gpu 0 --retrained=True >> fid_scores/retrained_c_spectral_20.txt
python FID/fid_score.py generated_data/original_images/ generated_data/c_spectral/30 --gpu 0 --retrained=True >> fid_scores/retrained_c_spectral_30.txt
python FID/fid_score.py generated_data/original_images/ generated_data/c_spectral/40 --gpu 0 --retrained=True >> fid_scores/retrained_c_spectral_40.txt
python FID/fid_score.py generated_data/original_images/ generated_data/c_spectral/50 --gpu 0 --retrained=True >> fid_scores/retrained_c_spectral_50.txt

