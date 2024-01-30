import os
import sys
import datetime as dt

a = os.getcwd()

print('Diretorio: ',a)
try:
    os.mkdir('./Output')
    os.mkdir('./downloads')
except:
    # pass
    os.chdir('./Output')
    [os.remove(arq) for arq in os.listdir()]
    os.chdir('../downloads')
    [os.remove(arq) for arq in os.listdir()]
    os.chdir('../')


hoje = dt.datetime.now()
os.chdir('./scripts')
arq = 'download_cmems.sh'
if hoje.day > 15:
    with open(arq) as r:
        text = r.read().replace('mes_inicio=`date -d "-2 month" +%Y-%m`','mes_inicio=`date -d "-1 month" +%Y-%m`').replace('mes_fim=`date -d "-1 month" +%Y-%m`','mes_fim=`date -d "" +%Y-%m`')
    with open(arq,'w') as w:
        w.write(text)
elif hoje.day < 15:
    with open(arq) as r:
        text = r.read().replace('mes_inicio=`date -d "-1 month" +%Y-%m`','mes_inicio=`date -d "-2 month" +%Y-%m`').replace('mes_fim=`date -d "" +%Y-%m`','mes_fim=`date -d "-1 month" +%Y-%m`')
    with open(arq,'w') as w:
        w.write(text)

os.chdir('../')
os.chmod("./scripts/download_cmems.sh", 755)
os.system("sh ./scripts/download_cmems.sh")

print(' ')
print('Download realizado com sucesso!')
print(' ')
print('Gerando figuras para o site: ')
os.system("python ./scripts/faoc_figuras_site_biod.py")


