# ADVANCED CNN
python run_cnn.py --train=1 --eval=0 --type=advanced
python run_cnn.py --train=0 --eval=1 --type=advanced >> accuracies/advanced_accuracy.txt

########## FSVAE MODELS ##########

# Epoch Models
python run_cnn.py --train=1 --eval=0 --type=advanced --augment=6000 --augmentor=fsvae --aug_epoch=1
python run_cnn.py --train=0 --eval=1 --type=advanced --augment=6000 --augmentor=fsvae --aug_epoch=1 >> accuracies/advanced_fsvae_1.txt

# Augmentation Models
python run_cnn.py --train=1 --eval=0 --type=advanced --augment=1000 --augmentor=fsvae --aug_epoch=1
python run_cnn.py --train=0 --eval=1 --type=advanced --augment=1000 --augmentor=fsvae --aug_epoch=1 >> accuracies/advanced_fsvae_1000.txt

python run_cnn.py --train=1 --eval=0 --type=advanced --augment=2000 --augmentor=fsvae --aug_epoch=1
python run_cnn.py --train=0 --eval=1 --type=advanced --augment=2000 --augmentor=fsvae --aug_epoch=1 >> accuracies/advanced_fsvae_2000.txt

python run_cnn.py --train=1 --eval=0 --type=advanced --augment=3000 --augmentor=fsvae --aug_epoch=1
python run_cnn.py --train=0 --eval=1 --type=advanced --augment=3000 --augmentor=fsvae --aug_epoch=1 >> accuracies/advanced_fsvae_3000.txt

python run_cnn.py --train=1 --eval=0 --type=advanced --augment=4000 --augmentor=fsvae --aug_epoch=1
python run_cnn.py --train=0 --eval=1 --type=advanced --augment=4000 --augmentor=fsvae --aug_epoch=1 >> accuracies/advanced_fsvae_4000.txt

python run_cnn.py --train=1 --eval=0 --type=advanced --augment=5000 --augmentor=fsvae --aug_epoch=1
python run_cnn.py --train=0 --eval=1 --type=advanced --augment=5000 --augmentor=fsvae --aug_epoch=1 >> accuracies/advanced_fsvae_5000.txt

# See accuracies/advanced_fsvae_1.txt
# python run_cnn.py --train=1 --eval=0 --type=advanced --augment=6000 --augmentor=fsvae --aug_epoch=1
# python run_cnn.py --train=0 --eval=1 --type=advanced --augment=6000 --augmentor=fsvae --aug_epoch=1 >> accuracies/advanced_fsvae_6000.txt

python run_cnn.py --train=1 --eval=0 --type=advanced --augment=7000 --augmentor=fsvae --aug_epoch=1
python run_cnn.py --train=0 --eval=1 --type=advanced --augment=7000 --augmentor=fsvae --aug_epoch=1 >> accuracies/advanced_fsvae_7000.txt

python run_cnn.py --train=1 --eval=0 --type=advanced --augment=8000 --augmentor=fsvae --aug_epoch=1
python run_cnn.py --train=0 --eval=1 --type=advanced --augment=8000 --augmentor=fsvae --aug_epoch=1 >> accuracies/advanced_fsvae_8000.txt

python run_cnn.py --train=1 --eval=0 --type=advanced --augment=9000 --augmentor=fsvae --aug_epoch=1
python run_cnn.py --train=0 --eval=1 --type=advanced --augment=9000 --augmentor=fsvae --aug_epoch=1 >> accuracies/advanced_fsvae_9000.txt

########## WGAN-GP MODELS ##########

# Epoch Models
python run_cnn.py --train=1 --eval=0 --type=advanced --augment=6000 --augmentor=wgan --aug_epoch=1
python run_cnn.py --train=0 --eval=1 --type=advanced --augment=6000 --augmentor=wgan --aug_epoch=1 >> accuracies/advanced_wgan_1.txt

python run_cnn.py --train=1 --eval=0 --type=advanced --augment=6000 --augmentor=wgan --aug_epoch=10
python run_cnn.py --train=0 --eval=1 --type=advanced --augment=6000 --augmentor=wgan --aug_epoch=10 >> accuracies/advanced_wgan_10.txt

python run_cnn.py --train=1 --eval=0 --type=advanced --augment=6000 --augmentor=wgan --aug_epoch=20
python run_cnn.py --train=0 --eval=1 --type=advanced --augment=6000 --augmentor=wgan --aug_epoch=20 >> accuracies/advanced_wgan_20.txt

python run_cnn.py --train=1 --eval=0 --type=advanced --augment=6000 --augmentor=wgan --aug_epoch=30
python run_cnn.py --train=0 --eval=1 --type=advanced --augment=6000 --augmentor=wgan --aug_epoch=30 >> accuracies/advanced_wgan_30.txt

python run_cnn.py --train=1 --eval=0 --type=advanced --augment=6000 --augmentor=wgan --aug_epoch=40
python run_cnn.py --train=0 --eval=1 --type=advanced --augment=6000 --augmentor=wgan --aug_epoch=40 >> accuracies/advanced_wgan_40.txt

python run_cnn.py --train=1 --eval=0 --type=advanced --augment=6000 --augmentor=wgan --aug_epoch=50
python run_cnn.py --train=0 --eval=1 --type=advanced --augment=6000 --augmentor=wgan --aug_epoch=50 >> accuracies/advanced_wgan_50.txt

# Augmentation Models
python run_cnn.py --train=1 --eval=0 --type=advanced --augment=1000 --augmentor=wgan --aug_epoch=30
python run_cnn.py --train=0 --eval=1 --type=advanced --augment=1000 --augmentor=wgan --aug_epoch=30 >> accuracies/advanced_wgan_1000.txt

python run_cnn.py --train=1 --eval=0 --type=advanced --augment=2000 --augmentor=wgan --aug_epoch=30
python run_cnn.py --train=0 --eval=1 --type=advanced --augment=2000 --augmentor=wgan --aug_epoch=30 >> accuracies/advanced_wgan_2000.txt

python run_cnn.py --train=1 --eval=0 --type=advanced --augment=3000 --augmentor=wgan --aug_epoch=30
python run_cnn.py --train=0 --eval=1 --type=advanced --augment=3000 --augmentor=wgan --aug_epoch=30 >> accuracies/advanced_wgan_3000.txt

python run_cnn.py --train=1 --eval=0 --type=advanced --augment=4000 --augmentor=wgan --aug_epoch=30
python run_cnn.py --train=0 --eval=1 --type=advanced --augment=4000 --augmentor=wgan --aug_epoch=30 >> accuracies/advanced_wgan_4000.txt

python run_cnn.py --train=1 --eval=0 --type=advanced --augment=5000 --augmentor=wgan --aug_epoch=30
python run_cnn.py --train=0 --eval=1 --type=advanced --augment=5000 --augmentor=wgan --aug_epoch=30 >> accuracies/advanced_wgan_5000.txt

# See accuracies/advanced_wgan_30.txt
# python run_cnn.py --train=1 --eval=0 --type=advanced --augment=6000 --augmentor=wgan --aug_epoch=30
# python run_cnn.py --train=0 --eval=1 --type=advanced --augment=6000 --augmentor=wgan --aug_epoch=30 >> accuracies/advanced_wgan_6000.txt

python run_cnn.py --train=1 --eval=0 --type=advanced --augment=7000 --augmentor=wgan --aug_epoch=30
python run_cnn.py --train=0 --eval=1 --type=advanced --augment=7000 --augmentor=wgan --aug_epoch=30 >> accuracies/advanced_wgan_7000.txt

python run_cnn.py --train=1 --eval=0 --type=advanced --augment=8000 --augmentor=wgan --aug_epoch=30
python run_cnn.py --train=0 --eval=1 --type=advanced --augment=8000 --augmentor=wgan --aug_epoch=30 >> accuracies/advanced_wgan_8000.txt

python run_cnn.py --train=1 --eval=0 --type=advanced --augment=9000 --augmentor=wgan --aug_epoch=30
python run_cnn.py --train=0 --eval=1 --type=advanced --augment=9000 --augmentor=wgan --aug_epoch=30 >> accuracies/advanced_wgan_9000.txt

########## SNGAN MODELS ##########

# Epoch Models
python run_cnn.py --train=1 --eval=0 --type=advanced --augment=6000 --augmentor=spectral --aug_epoch=1
python run_cnn.py --train=0 --eval=1 --type=advanced --augment=6000 --augmentor=spectral --aug_epoch=1 >> accuracies/advanced_spectral_1.txt

python run_cnn.py --train=1 --eval=0 --type=advanced --augment=6000 --augmentor=spectral --aug_epoch=10
python run_cnn.py --train=0 --eval=1 --type=advanced --augment=6000 --augmentor=spectral --aug_epoch=10 >> accuracies/advanced_spectral_10.txt

python run_cnn.py --train=1 --eval=0 --type=advanced --augment=6000 --augmentor=spectral --aug_epoch=20
python run_cnn.py --train=0 --eval=1 --type=advanced --augment=6000 --augmentor=spectral --aug_epoch=20 >> accuracies/advanced_spectral_20.txt

python run_cnn.py --train=1 --eval=0 --type=advanced --augment=6000 --augmentor=spectral --aug_epoch=30
python run_cnn.py --train=0 --eval=1 --type=advanced --augment=6000 --augmentor=spectral --aug_epoch=30 >> accuracies/advanced_spectral_30.txt

python run_cnn.py --train=1 --eval=0 --type=advanced --augment=6000 --augmentor=spectral --aug_epoch=40
python run_cnn.py --train=0 --eval=1 --type=advanced --augment=6000 --augmentor=spectral --aug_epoch=40 >> accuracies/advanced_spectral_40.txt

python run_cnn.py --train=1 --eval=0 --type=advanced --augment=6000 --augmentor=spectral --aug_epoch=50
python run_cnn.py --train=0 --eval=1 --type=advanced --augment=6000 --augmentor=spectral --aug_epoch=50 >> accuracies/advanced_spectral_50.txt

# Augmentation Models
python run_cnn.py --train=1 --eval=0 --type=advanced --augment=1000 --augmentor=spectral --aug_epoch=50
python run_cnn.py --train=0 --eval=1 --type=advanced --augment=1000 --augmentor=spectral --aug_epoch=50 >> accuracies/advanced_spectral_1000.txt

python run_cnn.py --train=1 --eval=0 --type=advanced --augment=2000 --augmentor=spectral --aug_epoch=50
python run_cnn.py --train=0 --eval=1 --type=advanced --augment=2000 --augmentor=spectral --aug_epoch=50 >> accuracies/advanced_spectral_2000.txt

python run_cnn.py --train=1 --eval=0 --type=advanced --augment=3000 --augmentor=spectral --aug_epoch=50
python run_cnn.py --train=0 --eval=1 --type=advanced --augment=3000 --augmentor=spectral --aug_epoch=50 >> accuracies/advanced_spectral_3000.txt

python run_cnn.py --train=1 --eval=0 --type=advanced --augment=4000 --augmentor=spectral --aug_epoch=50
python run_cnn.py --train=0 --eval=1 --type=advanced --augment=4000 --augmentor=spectral --aug_epoch=50 >> accuracies/advanced_spectral_4000.txt

python run_cnn.py --train=1 --eval=0 --type=advanced --augment=5000 --augmentor=spectral --aug_epoch=50
python run_cnn.py --train=0 --eval=1 --type=advanced --augment=5000 --augmentor=spectral --aug_epoch=50 >> accuracies/advanced_spectral_5000.txt

# See accuracies/advanced_spectral_50.txt
# python run_cnn.py --train=1 --eval=0 --type=advanced --augment=6000 --augmentor=spectral --aug_epoch=50
# python run_cnn.py --train=0 --eval=1 --type=advanced --augment=6000 --augmentor=spectral --aug_epoch=50 >> accuracies/advanced_spectral_6000.txt

python run_cnn.py --train=1 --eval=0 --type=advanced --augment=7000 --augmentor=spectral --aug_epoch=50
python run_cnn.py --train=0 --eval=1 --type=advanced --augment=7000 --augmentor=spectral --aug_epoch=50 >> accuracies/advanced_spectral_7000.txt

python run_cnn.py --train=1 --eval=0 --type=advanced --augment=8000 --augmentor=spectral --aug_epoch=50
python run_cnn.py --train=0 --eval=1 --type=advanced --augment=8000 --augmentor=spectral --aug_epoch=50 >> accuracies/advanced_spectral_8000.txt

python run_cnn.py --train=1 --eval=0 --type=advanced --augment=9000 --augmentor=spectral --aug_epoch=50
python run_cnn.py --train=0 --eval=1 --type=advanced --augment=9000 --augmentor=spectral --aug_epoch=50 >> accuracies/advanced_spectral_9000.txt

########## Class WGAN-GP MODELS ##########

# Epoch Models
python run_cnn.py --train=1 --eval=0 --type=advanced --augment=6000 --augmentor=c_wgan --aug_epoch=1
python run_cnn.py --train=0 --eval=1 --type=advanced --augment=6000 --augmentor=c_wgan --aug_epoch=1 >> accuracies/advanced_c_wgan_1.txt

python run_cnn.py --train=1 --eval=0 --type=advanced --augment=6000 --augmentor=c_wgan --aug_epoch=10
python run_cnn.py --train=0 --eval=1 --type=advanced --augment=6000 --augmentor=c_wgan --aug_epoch=10 >> accuracies/advanced_c_wgan_10.txt

python run_cnn.py --train=1 --eval=0 --type=advanced --augment=6000 --augmentor=c_wgan --aug_epoch=20
python run_cnn.py --train=0 --eval=1 --type=advanced --augment=6000 --augmentor=c_wgan --aug_epoch=20 >> accuracies/advanced_c_wgan_20.txt

python run_cnn.py --train=1 --eval=0 --type=advanced --augment=6000 --augmentor=c_wgan --aug_epoch=30
python run_cnn.py --train=0 --eval=1 --type=advanced --augment=6000 --augmentor=c_wgan --aug_epoch=30 >> accuracies/advanced_c_wgan_30.txt

python run_cnn.py --train=1 --eval=0 --type=advanced --augment=6000 --augmentor=c_wgan --aug_epoch=40
python run_cnn.py --train=0 --eval=1 --type=advanced --augment=6000 --augmentor=c_wgan --aug_epoch=40 >> accuracies/advanced_c_wgan_40.txt

python run_cnn.py --train=1 --eval=0 --type=advanced --augment=6000 --augmentor=c_wgan --aug_epoch=50
python run_cnn.py --train=0 --eval=1 --type=advanced --augment=6000 --augmentor=c_wgan --aug_epoch=50 >> accuracies/advanced_c_wgan_50.txt

# Augmentation Models
python run_cnn.py --train=1 --eval=0 --type=advanced --augment=1000 --augmentor=c_wgan --aug_epoch=50
python run_cnn.py --train=0 --eval=1 --type=advanced --augment=1000 --augmentor=c_wgan --aug_epoch=50 >> accuracies/advanced_c_wgan_1000.txt

python run_cnn.py --train=1 --eval=0 --type=advanced --augment=2000 --augmentor=c_wgan --aug_epoch=50
python run_cnn.py --train=0 --eval=1 --type=advanced --augment=2000 --augmentor=c_wgan --aug_epoch=50 >> accuracies/advanced_c_wgan_2000.txt

python run_cnn.py --train=1 --eval=0 --type=advanced --augment=3000 --augmentor=c_wgan --aug_epoch=50
python run_cnn.py --train=0 --eval=1 --type=advanced --augment=3000 --augmentor=c_wgan --aug_epoch=50 >> accuracies/advanced_c_wgan_3000.txt

python run_cnn.py --train=1 --eval=0 --type=advanced --augment=4000 --augmentor=c_wgan --aug_epoch=50
python run_cnn.py --train=0 --eval=1 --type=advanced --augment=4000 --augmentor=c_wgan --aug_epoch=50 >> accuracies/advanced_c_wgan_4000.txt

python run_cnn.py --train=1 --eval=0 --type=advanced --augment=5000 --augmentor=c_wgan --aug_epoch=50
python run_cnn.py --train=0 --eval=1 --type=advanced --augment=5000 --augmentor=c_wgan --aug_epoch=50 >> accuracies/advanced_c_wgan_5000.txt

# See accuracies/advanced_c_wgan_50.txt
# python run_cnn.py --train=1 --eval=0 --type=advanced --augment=6000 --augmentor=c_wgan --aug_epoch=50
# python run_cnn.py --train=0 --eval=1 --type=advanced --augment=6000 --augmentor=c_wgan --aug_epoch=50 >> accuracies/advanced_c_wgan_6000.txt

python run_cnn.py --train=1 --eval=0 --type=advanced --augment=7000 --augmentor=c_wgan --aug_epoch=50
python run_cnn.py --train=0 --eval=1 --type=advanced --augment=7000 --augmentor=c_wgan --aug_epoch=50 >> accuracies/advanced_c_wgan_7000.txt

python run_cnn.py --train=1 --eval=0 --type=advanced --augment=8000 --augmentor=c_wgan --aug_epoch=50
python run_cnn.py --train=0 --eval=1 --type=advanced --augment=8000 --augmentor=c_wgan --aug_epoch=50 >> accuracies/advanced_c_wgan_8000.txt

python run_cnn.py --train=1 --eval=0 --type=advanced --augment=9000 --augmentor=c_wgan --aug_epoch=50
python run_cnn.py --train=0 --eval=1 --type=advanced --augment=9000 --augmentor=c_wgan --aug_epoch=50 >> accuracies/advanced_c_wgan_9000.txt

########## Class SNGAN MODELS ##########

# Epoch Models
python run_cnn.py --train=1 --eval=0 --type=advanced --augment=6000 --augmentor=c_spectral --aug_epoch=1
python run_cnn.py --train=0 --eval=1 --type=advanced --augment=6000 --augmentor=c_spectral --aug_epoch=1 >> accuracies/advanced_c_spectral_1.txt

python run_cnn.py --train=1 --eval=0 --type=advanced --augment=6000 --augmentor=c_spectral --aug_epoch=10
python run_cnn.py --train=0 --eval=1 --type=advanced --augment=6000 --augmentor=c_spectral --aug_epoch=10 >> accuracies/advanced_c_spectral_10.txt

python run_cnn.py --train=1 --eval=0 --type=advanced --augment=6000 --augmentor=c_spectral --aug_epoch=20
python run_cnn.py --train=0 --eval=1 --type=advanced --augment=6000 --augmentor=c_spectral --aug_epoch=20 >> accuracies/advanced_c_spectral_20.txt

python run_cnn.py --train=1 --eval=0 --type=advanced --augment=6000 --augmentor=c_spectral --aug_epoch=30
python run_cnn.py --train=0 --eval=1 --type=advanced --augment=6000 --augmentor=c_spectral --aug_epoch=30 >> accuracies/advanced_c_spectral_30.txt

python run_cnn.py --train=1 --eval=0 --type=advanced --augment=6000 --augmentor=c_spectral --aug_epoch=40
python run_cnn.py --train=0 --eval=1 --type=advanced --augment=6000 --augmentor=c_spectral --aug_epoch=40 >> accuracies/advanced_c_spectral_40.txt

python run_cnn.py --train=1 --eval=0 --type=advanced --augment=6000 --augmentor=c_spectral --aug_epoch=50
python run_cnn.py --train=0 --eval=1 --type=advanced --augment=6000 --augmentor=c_spectral --aug_epoch=50 >> accuracies/advanced_c_spectral_50.txt

# Augmentation Models
python run_cnn.py --train=1 --eval=0 --type=advanced --augment=1000 --augmentor=c_spectral --aug_epoch=50
python run_cnn.py --train=0 --eval=1 --type=advanced --augment=1000 --augmentor=c_spectral --aug_epoch=50 >> accuracies/advanced_c_spectral_1000.txt

python run_cnn.py --train=1 --eval=0 --type=advanced --augment=2000 --augmentor=c_spectral --aug_epoch=50
python run_cnn.py --train=0 --eval=1 --type=advanced --augment=2000 --augmentor=c_spectral --aug_epoch=50 >> accuracies/advanced_c_spectral_2000.txt

python run_cnn.py --train=1 --eval=0 --type=advanced --augment=3000 --augmentor=c_spectral --aug_epoch=50
python run_cnn.py --train=0 --eval=1 --type=advanced --augment=3000 --augmentor=c_spectral --aug_epoch=50 >> accuracies/advanced_c_spectral_3000.txt

python run_cnn.py --train=1 --eval=0 --type=advanced --augment=4000 --augmentor=c_spectral --aug_epoch=50
python run_cnn.py --train=0 --eval=1 --type=advanced --augment=4000 --augmentor=c_spectral --aug_epoch=50 >> accuracies/advanced_c_spectral_4000.txt

python run_cnn.py --train=1 --eval=0 --type=advanced --augment=5000 --augmentor=c_spectral --aug_epoch=50
python run_cnn.py --train=0 --eval=1 --type=advanced --augment=5000 --augmentor=c_spectral --aug_epoch=50 >> accuracies/advanced_c_spectral_5000.txt

# See accuracies/advanced_c_spectral_50.txt
# python run_cnn.py --train=1 --eval=0 --type=advanced --augment=6000 --augmentor=c_spectral --aug_epoch=50
# python run_cnn.py --train=0 --eval=1 --type=advanced --augment=6000 --augmentor=c_spectral --aug_epoch=50 >> accuracies/advanced_c_spectral_6000.txt

python run_cnn.py --train=1 --eval=0 --type=advanced --augment=7000 --augmentor=c_spectral --aug_epoch=50
python run_cnn.py --train=0 --eval=1 --type=advanced --augment=7000 --augmentor=c_spectral --aug_epoch=50 >> accuracies/advanced_c_spectral_7000.txt

python run_cnn.py --train=1 --eval=0 --type=advanced --augment=8000 --augmentor=c_spectral --aug_epoch=50
python run_cnn.py --train=0 --eval=1 --type=advanced --augment=8000 --augmentor=c_spectral --aug_epoch=50 >> accuracies/advanced_c_spectral_8000.txt

python run_cnn.py --train=1 --eval=0 --type=advanced --augment=9000 --augmentor=c_spectral --aug_epoch=50
python run_cnn.py --train=0 --eval=1 --type=advanced --augment=9000 --augmentor=c_spectral --aug_epoch=50 >> accuracies/advanced_c_spectral_9000.txt