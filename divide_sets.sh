#!/bin/bash

base_path=data/mediterranean/samples

for month in {11..12}; do
    mv ${base_path}/train/rea_data_2021${month}* ${base_path}/val/
    cp ${base_path}/train/forcing_2021${month}* ${base_path}/val/
done

for month in {01..07}; do
    mv ${base_path}/train/rea_data_2022${month}* ${base_path}/test/
    cp ${base_path}/train/forcing_2022${month}* ${base_path}/test/
done

for month in {04..05}; do
    mv ${base_path}/train/ana_data_2024${month}* ${base_path}/val/
    cp ${base_path}/train/forcing_2024${month}* ${base_path}/val/
done

for month in {06..08}; do
    mv ${base_path}/train/ana_data_2024${month}* ${base_path}/test/
    cp ${base_path}/train/forcing_2024${month}* ${base_path}/test/
done
