@echo off

REM Définition de la variable pour le domaine cible
set target_domain=%1

REM Exécution du script Python avec les arguments
python main.py ^
--experiment=ActivationShaping ^
--experiment_name=ActivationShaping/%target_domain%/ ^
--dataset_args="{'root': 'data/PACS', 'source_domain': 'art_painting', 'target_domain': '%target_domain%'}" ^
--batch_size=128 ^
--num_workers=5 ^
--grad_accum_steps=1
