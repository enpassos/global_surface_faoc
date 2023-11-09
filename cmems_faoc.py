import os
import sys

a = os.getcwd()
print('Diretorio: ',a)

os.system("sh download_cmems2.sh")
print('Download realizado com sucesso!')
