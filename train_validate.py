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

mlflow.tensorflow.autolog(log_models=True, disable=False, exclusive=False)
with mlflow.start_run():
    save_dir_path = train_test(epochs=3, beta_1=0.9, beta_2=0.999, init_lr=2e-5)
    mlflow.tensorflow.save_model(save_dir_path, path=c.ML_FLOW_SAVE_DIR)
