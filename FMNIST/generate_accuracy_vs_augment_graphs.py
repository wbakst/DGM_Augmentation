from matplotlib import pyplot as plt
import numpy as np

colors = ['blue', 'green', 'red', 'purple', 'yellow']
models = ['fsvae', 'wgan', 'spectral', 'c_wgan', 'c_spectral']
augments = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]
def graph(filename, title, output_file):
	accuracies = {
		'fsvae': [],
		'wgan': [],
		'spectral': [],
		'c_wgan': [],
		'c_spectral': []
	}
	
	FID_ACC = []
	with open(filename, 'r') as f:
		f.readline()
		BASE_ACC = float(f.readline())
		index = -1
		for line in f:
			if line[0] == '#': 
				index += 1
				continue
			line = line.strip().split(',')
			accuracies[models[index]].append(float(line[1]))

	# FID_ACC = sorted(FID_ACC, reverse=True)
	# FID = [fid for fid, _ in FID_ACC]
	# ACC = [acc for _, acc in FID_ACC]

	# Plot lines for each model
	for model, color in zip(models, colors):
		plt.plot(augments, accuracies[model], color=color, label=model)

	plt.plot([0] + augments, [BASE_ACC] * (len(augments) + 1), color='black', linestyle=':', label='Baseline')

	plt.xlabel('Augmentation')
	plt.ylabel('Classification Accuracy')
	plt.title(title)
	plt.legend()
	plt.savefig(output_file)
	plt.show()

graph('graphs/accuracy_vs_augment_basic.txt', 'Basic Accuracy vs. Augmentation', 'graphs/basic_accuracy_vs_augment.png')
graph('graphs/accuracy_vs_augment_advanced.txt', 'Advanced Accuracy vs. Augmentation', 'graphs/advanced_accuracy_vs_augment.png')