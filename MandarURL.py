from flask import Flask
import sys
#import subprocess
import ListaRecUsOb

app = Flask(__name__)
#import os 
#os.system('python prueba.py')

@app.route('/Recomendacion/<num_casa>')
def Recomendacion(num_casa):
    s2_out = ListaRecUsOb.main(num_casa)
    return s2_out

if __name__ == '__main__':
    app.run()