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


os.system("python /faoc/scripts/cmems_download.py")

print(' ')
print('Download realizado com sucesso!')
print(' ')
print('Gerando figuras para o site: ')
os.system("python /faoc/scripts/faoc_figuras_site_biod.py")


