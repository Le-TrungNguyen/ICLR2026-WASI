from pytorch_lightning.callbacks import Callback
import torch


class LogResourceConsumptionCallback(Callback):
    """
        Callback to log resource consumption during training and validation.
    """

    def __init__(self, log_activation_mem=False, perplexity=None):
        self.log_activation_mem             = log_activation_mem    # if True: Log estimation of activation memory
        self.training_begin                 = False                 # a flag indicating the beginning of training
        self.num_train_batches              = None                  # number of batch of data for training

        if perplexity is not None:
            self.perplexity = perplexity
            self.total_epsilon = len(self.perplexity.perplexity[0])
            self.total_layer = len(self.perplexity.perplexity)
            self.epsilon_idx = 0
            self.layer_idx = 0

    def on_train_epoch_start(self, trainer, model):
        """
        Called at the beginning of a training epoch.
        Attaches a list to store memory information to the model (for SVD and HOSVD) if logging is enabled and it is the first epoch.
        """
        if not self.training_begin:
            if self.log_activation_mem:
                if hasattr(model, 'with_WSI') and model.with_WSI and hasattr(model, 'attach_memory_list_weight') and not model.WSI_with_sub_iter: # If using SVD to decompose weight at every iteration
                    model.attach_memory_list_weight()

                if hasattr(model, 'with_WSI_sparse') and model.with_WSI_sparse and hasattr(model, 'attach_memory_list_weight'):
                    model.attach_memory_list_weight()
                    
            self.training_begin = True



    def on_train_epoch_end(self, trainer, model):
        """
        Called at the end of a training epoch.
        Resets the attached memory list sizes for SVD or HOSVD methods after each epoch.
        """
        if self.log_activation_mem:
            
            if (hasattr(model, 'with_WSI') and model.with_WSI and hasattr(model, 'get_weight_size_WSI')) and not model.WSI_with_sub_iter: # If using SVD to decompose weight at every iteration
                model.get_weight_size_WSI(self.num_train_batches) # Decomposition only occurs during training 
                model.reset_memory_list_weight()

            
    
    def on_train_batch_start(self, trainer, model, batch, batch_idx, dataloader_idx):
        """
        Called at the start of a training batch.
        Sets the flag `train_batch_start` to True at the beginning of every batch.
        """
        if self.log_activation_mem:
            if model.current_epoch == 0 and batch_idx == 0:
                model.get_resource_consumption(register_hook=True)
            
    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx, dataloader_idx):
        """
        Called at the end of a training batch.
        Logs activation memory for the first batch if applicable (for Vanilla Training and Gradient Filter)
        """
        self.num_train_batches = batch_idx + 1
        if self.log_activation_mem:
            if model.current_epoch == 0 and batch_idx == 0:
                model.get_resource_consumption(register_hook=False)

        
        if model.just_log:
            trainer.should_stop = True
            trainer.limit_val_batches = 0

        if (hasattr(model, 'measure_perplexity_HOSVD_var') and model.measure_perplexity_HOSVD_var):

            for i in range(self.total_layer):
                self.perplexity.perplexity[i][self.epsilon_idx] = model.perplexity[i].item() if isinstance(model.perplexity[i], torch.Tensor) else model.perplexity[i]
                self.perplexity.ranks[i][self.epsilon_idx]      = model.measured_rank[i]
                self.perplexity.layer_mems[i][self.epsilon_idx] = model.layer_mem[i].item() if isinstance(model.layer_mem[i], torch.Tensor) else model.layer_mem[i]

            model.clear_measured_variables()

            self.epsilon_idx += 1
        




