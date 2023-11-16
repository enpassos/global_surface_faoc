import os
import sys

a = os.getcwd()
print('Diretorio: ',a)

os.system("sh download_cmems2.sh")
print(' ')
print('Download realizado com sucesso!')
print(' ')
print('Gerando figuras para o site: ')

# Colocar aqui para chamar o script "faoc_figuras_site_biod.py"
# Deletar teste.sh download_cmems.sh environment.yml 
# Fazer a pasta de volume
