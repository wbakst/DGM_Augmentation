from matplotlib import pyplot as plt
import numpy as np

def graph(filename, title, output_file):
	FID_ACC = []
	with open(filename, 'r') as f:
		f.readline()
		BASIC_ACC = float(f.readline())
		for line in f:
			if line[0] == '#': continue
			line = line.strip().split(',')
			FID_ACC.append((float(line[2]), float(line[1])))

	FID_ACC = sorted(FID_ACC, reverse=True)
	FID = [fid for fid, _ in FID_ACC]
	ACC = [acc for _, acc in FID_ACC]

	plt.plot(FID, ACC, color='r', linestyle='--')
	plt.plot(FID + [0], [BASIC_ACC] * (len(FID) + 1), color='black', linestyle=':', label='Baseline')

	plt.gca().invert_xaxis()

	plt.xlabel('FID Score')
	plt.ylabel('Classification Accuracy')
	plt.title(title)
	plt.legend()
	plt.savefig(output_file)
	plt.show()

graph('graphs/accuracy_vs_fid_basic.txt', 'Basic Accuracy vs. FID Score', 'graphs/basic_accuracy_vs_fid.png')
graph('graphs/accuracy_vs_fid_advanced.txt', 'Advanced Accuracy vs. FID Score', 'graphs/advanced_accuracy_vs_fid.png')
graph('graphs/accuracy_vs_retrained_fid_basic.txt', 'Basic Accuracy vs. Retrained FID Score', 'graphs/basic_accuracy_vs_retrained_fid.png')
graph('graphs/accuracy_vs_retrained_fid_advanced.txt', 'Advanced Accuracy vs. Retrained FID Score', 'graphs/advanced_accuracy_vs_retrained_fid.png')