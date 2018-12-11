
fid_scores = {
	'fsvae_1': 47.26,
	'wgan_1': 91.95,
	'wgan_10': 22.96,
	'wgan_20': 19.67,
	'wgan_30': 16.59,
	'wgan_40': 31.65,
	'wgan_50': 48.61,
	'spectral_1': 128.52,
	'spectral_10': 31.92,
	'spectral_20': 21.77,
	'spectral_30': 17.02,
	'spectral_40': 14.22,
	'spectral_50': 13.33,
	'c_wgan_1': 181.98,
	'c_wgan_10': 49.67,
	'c_wgan_20': 37.29,
	'c_wgan_30': 28.81,
	'c_wgan_40': 24.58,
	'c_wgan_50': 22.06,
	'c_spectral_1': 215.14,
	'c_spectral_10': 67.55,
	'c_spectral_20': 48.35,
	'c_spectral_30': 40.46,
	'c_spectral_40': 24.58,
	'c_spectral_50': 30.23,
}

retrained_fid_scores = {
	'fsvae_1': 2.73,
	'wgan_1': 8.46,
	'wgan_10': 1.14,
	'wgan_20': 0.47,
	'wgan_30': 0.53,
	'wgan_40': 2.01,
	'wgan_50': 2.60,
	'spectral_1': 7.94,
	'spectral_10': 1.45,
	'spectral_20': 1.35,
	'spectral_30': 0.28,
	'spectral_40': 0.55,
	'spectral_50': 0.12,
	'c_wgan_1': 9.78,
	'c_wgan_10': 2.40,
	'c_wgan_20': 1.12,
	'c_wgan_30': 1.05,
	'c_wgan_40': 0.71,
	'c_wgan_50': 0.71,
	'c_spectral_1': 10.26,
	'c_spectral_10': 4.35,
	'c_spectral_20': 3.06,
	'c_spectral_30': 1.14,
	'c_spectral_40': 0.93,
	'c_spectral_50': 1.21,
}

def get_accuracy(filename):
	f = open(filename)
	lines = [line.strip() for line in f]
	line = lines[-1][10:]
	accuracy = '{:.4f}'.format(float(line))
	f.close()
	return accuracy

def write_accuracy_vs_fid(out, model_type, model, epoch, retrained=False):
	accuracy = get_accuracy('accuracies/{}_{}_{}.txt'.format(model_type, model, epoch))
	if not retrained:
		fid_score = fid_scores['{}_{}'.format(model, epoch)]
	else:
		fid_score = retrained_fid_scores['{}_{}'.format(model, epoch)]
	print('{},{},{}'.format(epoch, accuracy, fid_score), file=out)

models = ['fsvae', 'wgan', 'spectral', 'c_wgan', 'c_spectral']
epochs = [1, 10, 20, 30, 40, 50]
def accuracy_vs_fid(model_type, retrained):
	if retrained:
		filename = 'graphs/accuracy_vs_retrained_fid_{}.txt'.format(model_type)
	else:
		filename = 'graphs/accuracy_vs_fid_{}.txt'.format(model_type)
	with open(filename, 'w') as out:
		print('# {} accuracy'.format(model_type), file=out)
		accuracy = get_accuracy('accuracies/{}_accuracy.txt'.format(model_type))
		print(accuracy, file=out)
		print('# fsvae', file=out)
		write_accuracy_vs_fid(out, model_type, 'fsvae', 1, retrained)
		for model in models[1:]:
			print('# {}'.format(model), file=out)
			for epoch in epochs:
				write_accuracy_vs_fid(out, model_type, model, epoch, retrained)

def write_accuracy_vs_augment(out, model_type, model, augment):
	file_augment = augment
	if augment == 6000:
		if model == 'fsvae': file_augment = 1
		elif model == 'wgan': file_augment = 30
		else: file_augment = 50
	accuracy = get_accuracy('accuracies/{}_{}_{}.txt'.format(model_type, model, file_augment))
	print('{},{}'.format(augment, accuracy), file=out)

augments = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]
def accuracy_vs_augment(model_type):
	filename = 'graphs/accuracy_vs_augment_{}.txt'.format(model_type)
	with open(filename, 'w') as out:
		print('# {} accuracy'.format(model_type), file=out)
		accuracy = get_accuracy('accuracies/{}_accuracy.txt'.format(model_type))
		print(accuracy, file=out)
		for model in models:
			print('# {}'.format(model), file=out)
			for augment in augments:
				write_accuracy_vs_augment(out, model_type, model, augment)

def main():
	accuracy_vs_fid('basic', False)
	accuracy_vs_fid('basic', True)
	accuracy_vs_fid('advanced', False)
	accuracy_vs_fid('advanced', True)
	accuracy_vs_augment('basic')
	accuracy_vs_augment('advanced')

if __name__ == '__main__':
	main()
