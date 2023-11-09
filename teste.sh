#!/usr/bin/bash

#data_inicial = '2023-04-01'
#data_final = (date -d "$data_inicial -30 days" +"%Y%m%d")


data_final=`date -d "-2 month" +%Y%m%d`
data_inicial=`date +%Y%m%d`

echo $data_inicial
echo $data_final