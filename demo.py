from ensemble import WeightedEnsembleClassifier
from skmultiflow.data.file_stream import FileStream

# prepare the stream
# reuse the electricity stream for test
path_data = 'data/'
stream = FileStream(path_data + 'elec.csv', n_targets=1, target_idx=-1)
stream.prepare_for_use()

# instantiate a classifier
clf = WeightedEnsembleClassifier()
