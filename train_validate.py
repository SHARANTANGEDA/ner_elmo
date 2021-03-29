import argparse
import logging
import os
from datetime import datetime

from elmo_model.model import train_test
import mlflow
import constants as c

logging.basicConfig(filename=os.path.join(c.LOGS_DIR, f'{datetime.now()}.txt'),
                    filemode='w+',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)
logging.getLogger().setLevel(logging.INFO)

parser = argparse.ArgumentParser(description='Train & Save Model')
parser.add_argument('--epochs', type=int, dest="epochs", help="Num of epochs to run", default=3)
parser.add_argument('--beta_1', type=float, dest="beta_1", help="beta_1 hyper-parameter", default=0.9)
parser.add_argument('--beta_2', type=float, dest="beta_2", help="beta_2 hyper-parameter", default=0.999)
parser.add_argument('--lr', type=float, dest="lr", help="Initial Learning rate(Default: 2e-5)", default=2e-5)

args = parser.parse_args()

experiment = mlflow.get_experiment(c.ML_FLOW_EXPERIMENT_ID)
logging.info("Name: {}".format(experiment.name))
logging.info("Experiment_id: {}".format(experiment.experiment_id))
logging.info("Artifact Location: {}".format(experiment.artifact_location))
logging.info("Tags: {}".format(experiment.tags))
logging.info("Lifecycle_stage: {}".format(experiment.lifecycle_stage))


mlflow.tensorflow.autolog(log_models=True, disable=False, exclusive=False)
with mlflow.start_run(experiment_id=c.ML_FLOW_EXPERIMENT_ID):
    save_dir_path, tags, signature = train_test(epochs=args.epochs, beta_1=args.beta_1, beta_2=args.beta_2,
                                                init_lr=args.lr)

