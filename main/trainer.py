import logging
from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning.loggers import TensorBoardLogger
from classification.model import ClassificationModel
from dataloader.pl_dataset import ClsDataset
from classification.callback import LogResourceConsumptionCallback
import os
from pytorch_lightning.callbacks import ModelCheckpoint

logging.basicConfig(level=logging.INFO)


class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--logger.save_dir", default='./runs')
        parser.add_argument("--logger.exp_name", default='test')
        parser.add_argument("--checkpoint", default=None)

        parser.add_argument('--set_of_epsilons', type=str, help='Comma separated list of epochs', default='0, 0')

    def instantiate_trainer(self, **kwargs):
        if 'fit' in self.config.keys():
            cfg = self.config['fit']
        elif 'validate' in self.config.keys():
            cfg = self.config['validate']
        else:
            cfg = self.config
        logger_name = cfg['logger']['exp_name'] + "_" + cfg['data']['name']
        if 'logger_postfix' in kwargs:
            logger_name += kwargs['logger_postfix']
            kwargs.pop('logger_postfix')
        logger = TensorBoardLogger(cfg['logger']['save_dir'], logger_name)
        kwargs['logger'] = logger
  
        # kwargs['accelerator']='gpu'
        # kwargs['devices']="auto"

          
        model_initialized = hasattr(self, 'model') and self.model is not None
        just_log_enabled = model_initialized and hasattr(self.model, 'just_log') and self.model.just_log
        if just_log_enabled:
            if 'callbacks' not in kwargs:
                kwargs['callbacks'] = []
            logging.info("just_log=True: Removing or disabling checkpoint saving")
            filtered_callbacks = []
            for callback in kwargs['callbacks']:
                if not isinstance(callback, ModelCheckpoint):
                    filtered_callbacks.append(callback)
            kwargs['callbacks'] = filtered_callbacks
            kwargs['enable_checkpointing'] = False
        ######################################

        trainer = super(CLI, self).instantiate_trainer(**kwargs)
        return trainer

    def reset_trainer(self, **kwargs):
        return self.instantiate_trainer(**kwargs)



from utils.perplexity_dp import Perplexity
import shutil

def delete_junk_folder(base_path):
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        if os.path.isdir(folder_path) and folder_name.startswith("version_"):
            shutil.rmtree(folder_path)

def run():
    cli = CLI(ClassificationModel, ClsDataset, run=False, save_config_overwrite=True)

    model = cli.model
    trainer = cli.trainer
    data = cli.datamodule
    ##############################################
    if model.measure_perplexity_HOSVD_var:
        total_linear_layer = len(model.all_linear_layers)

        set_of_epsilons = [float(item) for item in cli.config['set_of_epsilons'].split(',')]

        perplexity = Perplexity(set_of_epsilons=set_of_epsilons,
                                perplexity=[[None for epsilon_idx in range(len(set_of_epsilons))] for layer in range(total_linear_layer)],
                                ranks=[[None for epsilon_idx in range(len(set_of_epsilons))] for layer in range(total_linear_layer)],
                                layer_mems=[[None for epsilon_idx in range(len(set_of_epsilons))] for layer in range(total_linear_layer)])
        
        total_epoch = len(set_of_epsilons)
        trainer = cli.reset_trainer(max_epochs=total_epoch)

        callback = LogResourceConsumptionCallback(perplexity=perplexity)
        trainer.callbacks.append(callback)

        trainer.fit(model, data)

        perplexity.save(os.path.join(os.path.dirname(model.logger.log_dir), 'perplexity.pkl'))
        delete_junk_folder(os.path.dirname(model.logger.log_dir))


        return
    #################################################################################################
    log_activation_mem = True
    print("==============================", trainer.max_epochs)
    # Add call back to log activation memory
    callback = LogResourceConsumptionCallback(log_activation_mem=log_activation_mem)
    trainer.callbacks.append(callback)
    
    # logging.info(str(model))

    trainer.fit(model, data)


run()
