import datetime as dt
from dateutil.relativedelta import relativedelta
import os

# Parametros do usuario
user = 'epassos'			# login no MyOcean
password = '6Tilxl:Q0'		# senha

#--------------------------------------------------------------
# Definição de Diretórios
dest = '/faoc/downloads'

# Area de interesse
xmin = '-180'
xmax = '180'
ymin = '-80'
ymax = '90'

# Data
hoje = dt.date.today()
if hoje.day > 15:
    inicio = (hoje + relativedelta(months=-2)).strftime('%Y-%m')
    final = (hoje + relativedelta(months=-1)).strftime('%Y-%m')
elif hoje.day < 15:
    inicio = (hoje + relativedelta(months=-3)).strftime('%Y-%m')
    final = (hoje + relativedelta(months=-2)).strftime('%Y-%m')

print('Mes inicio: ',inicio)
print('Mes final: ',final)

# Download

print(' ')
print('Download do Modelo Biogeoquímico')
print(' ')
comando = 'copernicusmarine subset -i global-analysis-forecast-bio-001-028-monthly --username '+user+' --password '+password+' -v chl -v o2 -v po4 -v si -v fe -v ph -v no3 -t "'+inicio+'-01T12:00:00" -T "'+final+'-01T12:00:00" -x '+xmin+' -X '+xmax+' -y '+ymin+' -Y '+ymax+' -z 0. -Z 1 -o '+dest+' -f cmems_bio.nc --force-download --disable-progress-bar --overwrite-output-data'
os.system(comando)


print(' ')
print(' ')
print('Download do Modelo Hidrodinâmico: Velocidade')
print(' ')
comando = 'copernicusmarine subset -i cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m --username '+user+' --password '+password+' -v uo -v vo -t "'+inicio+'-01T12:00:00" -T "'+final+'-01T12:00:00" -x '+xmin+' -X '+xmax+' -y '+ymin+' -Y '+ymax+' -z 0. -Z 1 -o '+dest+' -f cmems_vel.nc --force-download --disable-progress-bar --overwrite-output-data'
os.system(comando)


print(' ')
print(' ')
print('Download do Modelo Hidrodinâmico: Temperatura')
print(' ')
comando = 'copernicusmarine subset -i cmems_mod_glo_phy-thetao_anfc_0.083deg_P1M-m --username '+user+' --password '+password+' -v thetao -t "'+inicio+'-01T12:00:00" -T "'+final+'-01T12:00:00" -x '+xmin+' -X '+xmax+' -y '+ymin+' -Y '+ymax+' -z 0. -Z 1 -o '+dest+' -f cmems_temp.nc --force-download --disable-progress-bar --overwrite-output-data'
os.system(comando)