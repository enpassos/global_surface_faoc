export python=/usr/local/bin/python
export PATH="$PATH:$python/bin/"
motu='/root/miniconda3/lib/python3.11/site-packages/motuclient/motuclient.py' # local do motu-client.py

#--------------------------------------------------------------
# Parametros do usuario
user='epassos'			# login no MyOcean
pass='******'		# senha

#--------------------------------------------------------------
# Definição de Diretórios
dest='/faoc_site/downloads'

# Area de interesse
xmin='-180'
xmax='180'
ymin='-80'
ymax='90'


#mes_inicio='2023-04'
#mes_fim='2023-05'
mes_inicio=`date -d "-1 month" +%Y-%m`
mes_fim=`date -d "" +%Y-%m`

python ${motu} -m  http://nrt.cmems-du.eu/motu-web/Motu -s GLOBAL_ANALYSIS_FORECAST_BIO_001_028-TDS -d global-analysis-forecast-bio-001-028-monthly --longitude-min ${xmin} --longitude-max ${xmax} --latitude-min ${ymin} --latitude-max ${ymax} --date-min "${mes_inicio}-16 12:00:00" --date-max "${mes_fim}-16 12:00:00" --depth-min 0 --depth-max 1 --variable chl --variable o2 --variable po4 --variable si --variable fe --variable ph --variable no3 --out-dir ${dest} --out-name cmems_bio.nc --user ${user} --pwd ${pass}
python ${motu} -m  http://nrt.cmems-du.eu/motu-web/Motu -s GLOBAL_ANALYSISFORECAST_PHY_001_024-TDS -d cmems_mod_glo_phy-thetao_anfc_0.083deg_P1M-m --longitude-min ${xmin} --longitude-max ${xmax} --latitude-min ${ymin} --latitude-max ${ymax} --date-min "${mes_inicio}-16 00:00:00" --date-max "${mes_fim}-16 23:00:00" --depth-min 0 --depth-max 1 --variable thetao --out-dir ${dest} --out-name cmems_temp.nc --user ${user} --pwd ${pass}
python ${motu} -m  http://nrt.cmems-du.eu/motu-web/Motu -s GLOBAL_ANALYSISFORECAST_PHY_001_024-TDS -d cmems_mod_glo_phy-cur_anfc_0.083deg_P1M-m --longitude-min ${xmin} --longitude-max ${xmax} --latitude-min ${ymin} --latitude-max ${ymax} --date-min "${mes_inicio}-16 00:00:00" --date-max "${mes_fim}-16 23:00:00" --depth-min 0 --depth-max 1 --variable uo --variable vo --out-dir ${dest} --out-name cmems_vel.nc --user ${user} --pwd ${pass}
