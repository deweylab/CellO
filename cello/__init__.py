from .cello import train_model
from .cello import load_training_set
from .cello import predict
from .cello import _retrieve_pretrained_model
from .cello import retreive_pretrained_model_from_local
from .plot_annotations import probabilities_on_graph
from .scanpy_cello import cello as scanpy_cello, cello_probs, normalize_and_cluster, write_to_tsv
