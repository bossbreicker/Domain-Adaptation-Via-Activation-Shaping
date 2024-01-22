@echo off

REM Definition of the variable for the target domain
set target_domain=%1

REM Execute the Python script with arguments
python main.py ^
--experiment=DomainGeneralization ^
--experiment_name=DomainGeneralization/%target_domain%/ ^
--dataset_args="{'root': 'data/PACS', 'source_domains': ['art_painting', 'photo', 'cartoon', 'sketch'], 'target_domain': '%target_domain%'}" ^
--batch_size=128 ^
--num_workers=5 ^
--grad_accum_steps=1