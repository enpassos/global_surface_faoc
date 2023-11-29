import os
import sys

a = os.getcwd()
print('Diretorio: ',a)

os.system("sh ./scripts/download_cmems.sh")
print(' ')
print('Download realizado com sucesso!')
print(' ')
print('Gerando figuras para o site: ')
os.system("python ./scripts/faoc_figuras_site_biod.py")

# Colocar aqui para chamar o script "faoc_figuras_site_biod.py"
# Deletar teste.sh download_cmems.sh environment.yml 
# Fazer a pasta de volume
