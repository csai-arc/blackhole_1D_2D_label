To initiate stochasticity classification training, please use the below command.

> python stochasticity_classification_training.py -a effnetv2_s --data ../data/LSS_images --epochs 15 --gpu-id 0 -c <'path where checkpoints to be saved> --train-batch 10 --test-batch 2 --optuna_study_db sqlite:///.<'path where optuna db to be saved>


To initiate stochasticity classification testing, please use the below command.

> python stochasticity_classification_testing.py -a effnetv2_s --data ../data/LSS_images --epochs 15 --gpu-id 0 --weights_load weights/model_best_trial_3_epoch_3.pth.tar --evaluate --train-batch 10 --test-batch 1 --optuna_study_db sqlite:///.<'path where optuna db to be saved>






























