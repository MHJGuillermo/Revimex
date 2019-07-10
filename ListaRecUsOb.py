import pandas as pd 
import numpy as np
import math
import random 
import matplotlib.pyplot as plt
from scipy import sparse
import csv
import time
import sys #Para tomar variables del entorno

#Definición de funciones de similitud
#Simulitud para datos numéricos
def Sim_num(P1,P2,max):
    return 1-abs(P1-P2)/max
#Similitud para datos binarios
def Sim_equal_dif(P1,P2):
    if P1==P2:
        return 1
    else:
        return 0
#Similitu en dirección
def Sim_direccion(P1,P2):
    con=0
    #Verifica si existen los datos
    if (type(4.)!=type(P1['Investor\'s State']) and type(4.)!=type(P1['Investor\'s State'])):
        con=con+1
        if (type(4.) != type(P1['Investor\'s City']) and type(4.) != type(P2['Investor\'s City'])):
            con=con+1
            if (type(4.) != type(P1['Investor\'s Zip Code']) and type(4.) != type(P2['Investor\'s Zip Code'])):
                con=con+1
                if (type(4.) != type(P1['Investor\'s Local City Address']) and type(4.) != type(P2['Investor\'s Local City Address'])):
                    con=con+1
    if con==4:                    
        if P1['Investor\'s State']==P2['Investor\'s State']:
            if P1['Investor\'s City']==P2['Investor\'s City']:
                if P1['Investor\'s Zip Code']==P2['Investor\'s Zip Code']:
                    if P1['Investor\'s Local City Address']==P2['Investor\'s Local City Address']:
                        return 1.0
                    else:
                        return 0.75
                else:
                    return 0.5
            else:
                return 0.25
        else:
            return 0
    elif con==3:
        if P1['Investor\'s State']==P2['Investor\'s State']:
            if P1['Investor\'s City']==P2['Investor\'s City']:
                if P1['Investor\'s Zip Code']==P2['Investor\'s Zip Code']:
                    return 1.0
                else:
                    return 2./3
            else:
                return 1./3
        else:
            return 0
    elif con==2:
        if P1['Investor\'s State']==P2['Investor\'s State']:
            if P1['Investor\'s City']==P2['Investor\'s City']:
                return 1.0
            else:
                return 1./2
        else:
            return 0
    elif con==1:
        if P1['Investor\'s State']==P2['Investor\'s State']:
            return 1.0
        else:
            return 0
    else:
        return float('NaN')       

#Estas funciones son para usar latitudes y longitudes        
def Distancia(x1,x2,y1,y2):
    return math.sqrt((x2-x1)**2+(y2-y1)**2)
def Sim_place(x1,x2,y1,y2,max):
    return 1-Distancia(x1,x2,y1,y2)/max

#Devuelve la lista de objetos similares
def List_sim_object(sf_ob_Training_box,
    object_target,
    vec_ob):
    list_sim=[]
    v_pesos=vec_ob
    for row in range(0,sf_ob_Training_box.shape[0]):
        v_val=np.zeros(6)
        v_sim=np.zeros(6)
        if ( not(np.isnan(sf_ob_Training_box.iloc[row]['Sale Value'])) and not(np.isnan(object_target['Sale Value']))):
            v_sim[0]=Sim_num(sf_ob_Training_box.iloc[row]['Sale Value'],object_target['Sale Value'],maxd_sale_value)
            v_val[0]=1
        if (type(np.nan)!=type(sf_ob_Training_box.iloc[row]['Property Type']) and type(np.nan)!=type(object_target['Property Type'])):
            v_sim[1]=Sim_equal_dif(sf_ob_Training_box.iloc[row]['Property Type'],object_target['Property Type'])
            v_val[1]=1
        if (not(np.isnan(sf_ob_Training_box.iloc[row]['Latitude'])) 
            and not(np.isnan(object_target['Latitude']))
            and not(np.isnan(sf_ob_Training_box.iloc[row]['Longitude']))
            and not(np.isnan(object_target['Longitude']))):
            v_sim[2]=Sim_place(x1=sf_ob_Training_box.iloc[row]['Latitude'],
            x2=object_target['Latitude'],
            y1=sf_ob_Training_box.iloc[row]['Longitude'],
            y2=object_target['Longitude'],
            max=maxd_dist_ob)
            v_val[2]=1
        if ( not(np.isnan(sf_ob_Training_box.iloc[row]['Bedrooms'])) and not(np.isnan(object_target['Bedrooms']))):            
            v_sim[3]=Sim_num(sf_ob_Training_box.iloc[row]['Bedrooms'],object_target['Bedrooms'],maxd_bedroom)
            v_val[3]=1
        if ( not(np.isnan(sf_ob_Training_box.iloc[row]['Baths'])) and not(np.isnan(object_target['Baths']))): 
            v_sim[4]=Sim_num(sf_ob_Training_box.iloc[row]['Baths'],object_target['Baths'],maxd_baths)
            v_val[4]=1
        if ( not(np.isnan(sf_ob_Training_box.iloc[row]['Construction m2'])) and not(np.isnan(object_target['Construction m2']))): 
            v_sim[5]=Sim_num(sf_ob_Training_box.iloc[row]['Construction m2'],object_target['Construction m2'],maxd_con)
            v_val[5]=1
        #redefinimos los pesos para que solo comparen caracteristicas que existen
        if np.dot(v_val,v_val)<6:
            den=np.dot(v_pesos,v_val)
            for xi in v_pesos:
                xi= (xi/den)
            simt=np.dot(v_sim,v_pesos)
            v_pesos=vec_ob
        else:    
            simt=np.dot(v_sim,v_pesos)
        list_sim.append(simt)    
    list_sim=pd.Series(data=list_sim,index=sf_ob_Training_box['Property ID'])
    list_sim=list_sim.sort_values(ascending=False)
    return list_sim

def Rec_list_user_for_ob (matrix_sim_us,
    sf_ob_Training_box,
    sf_user_Training_box,
    Training_box,
    object_target,
    vec_ob,
    peso_us=0):
    peso_ob=1-peso_us
    #lista de casas recomendadas para el object_target
    #Aquí solo llamaremos la matriz ya construida
    list_sim_object_target=List_sim_object (sf_ob_Training_box=sf_ob_Training_box,
        object_target=object_target,
        vec_ob=vec_ob)
    #Solo tomamos los primeros 10
    list_sim_object_target=list_sim_object_target.iloc[:10]
    user_of_object=sf_user_Training_box.loc[sf_user_Training_box['ID'] == Training_box.loc[Training_box['Property ID']==list_sim_object_target.index[0]].iloc[0]['ID']].iloc[0]
    list_sim_user_of_object=matrix_sim_us[user_of_object['ID']].sort_values(ascending=False)
    #Solo tomamos los primeros 10
    #list_sim_user_of_object=list_sim_user_of_object.iloc[:10]    
    #Añadimos el usuario que compró la casa más parecida
    aux=peso_us+peso_ob*list_sim_object_target.iloc[0]
    lt_rec_for_object_target=pd.Series(data=[aux],index=[user_of_object['ID']])
    #Añadimos usuarios parecidos al que compró la casa parecida
    for us in range(0,list_sim_user_of_object.shape[0]-1): 
        aux=peso_ob*list_sim_object_target.iloc[0]+peso_us*list_sim_user_of_object.iloc[us]
        lt_rec_for_object_target=lt_rec_for_object_target.append(pd.Series(data=[aux], index=[list_sim_user_of_object.index[us]]))
    #Depués repetimos el procedimiento para el resto de las casas más parecidos con sus compradores
    #Este programa para no repetir datos se esta tardando un par de segundos
    for ob in range(1,list_sim_object_target.shape[0]):
        user_of_object=sf_user_Training_box.loc[sf_user_Training_box['ID'] == Training_box.loc[Training_box['Property ID']==list_sim_object_target.index[ob]].iloc[0]['ID']].iloc[0]
        list_sim_user_of_object=matrix_sim_us[user_of_object['ID']].sort_values(ascending=False)
        #Añadimos los usuarios que compraron casas parecidas
        aux=peso_us+peso_ob*list_sim_object_target.iloc[ob]
        if list_sim_user_of_object.index[-1] not in lt_rec_for_object_target.index:
            lt_rec_for_object_target=pd.Series(data=[aux],index=[list_sim_user_of_object.index[-1]])
        #Solo quedaría esta
        else:
            if aux>lt_rec_for_object_target[list_sim_user_of_object.index[-1]]:
                lt_rec_for_object_target[list_sim_user_of_object.index[-1]]=aux
        #list_sim_user_of_object=list_sim_user_of_object[:10]
        #Despues añadimos los usuarios parecidos al usuario que compró la casa ob-parecida
        for us in range(0,list_sim_user_of_object.shape[0]): 
            aux=peso_ob*list_sim_object_target.iloc[ob]+peso_us*list_sim_user_of_object.iloc[us]
            #Me parece que esta parte no va a ser necesaria
            #print(list_sim_user_of_object.index[us])
            #print(lt_rec_for_object_target.index)
            if list_sim_user_of_object.index[us] not in lt_rec_for_object_target.index:
                lt_rec_for_object_target=lt_rec_for_object_target.append(pd.Series(data=[aux], index=[list_sim_user_of_object.index[us]]))
            #Solo quedaría esta
            else:
                if aux>lt_rec_for_object_target[list_sim_user_of_object.index[us]]:
                    lt_rec_for_object_target[list_sim_user_of_object.index[us]]=aux
        #print(ob,list_sim_object_target.shape[0])
    #Esta es la lista de los posibles compradores
    lt_rec_for_object_target=lt_rec_for_object_target.sort_values(ascending=False)
    return lt_rec_for_object_target

#Preparación
print('Cargando datos...')
Int_DataFrame_Inv = pd.read_csv('Inversionistas.csv',index_col='#')
#print(Int_DataFrame_Inv.head())

#Eliminar datos iguales 
al=list([])
for row in Int_DataFrame_Inv:
    al.append(row)
Int_DataFrame_Inv=Int_DataFrame_Inv.drop_duplicates(subset =al)
#Vamos a eliminar las casas que están repetidas
Int_DataFrame_Inv=Int_DataFrame_Inv.drop_duplicates(subset ="Property ID")

#Creo que sería buena idea seleccionar solo los atributos utiles y trabajar con esas listas para que se más rápido
sf_house_T_b = Int_DataFrame_Inv[
    ["Property ID",
    "Bedrooms",
    "Baths",
    "Latitude",
    "Longitude",
    "Construction m2",
    "Parking",
    "Levels",
    "Property Type",
    "Sale Value"]]
sf_user_T_b = Int_DataFrame_Inv[
    ["ID",
    "Investor's Name",
    "Age",
    "Place of Interest",
    "Investor\'s Latitude",
    "Investor\'s Longitude",
    "Investor\'s Country",
    "Investor's State",
    "Investor\'s City",
    "Investor\'s Zip Code",
    "Investor\'s Local City Address"]]

#Eliminar datos iguales en personas y en casas
sf_user_T_b=sf_user_T_b.drop_duplicates(subset ="ID")
sf_house_T_b=sf_house_T_b.drop_duplicates(subset="Property ID")

#Importar los datos para los máximos
maxd_import=pd.read_csv('./Datos/Maximos.csv',index_col=0)
maxd_ages=maxd_import['Maxd User']['maxd_ages']
maxd_dist_us=maxd_import['Maxd User']['maxd_dist_us']
maxd_sale_value=maxd_import['Maxd Object']['maxd_sale_value']
maxd_bedroom=maxd_import['Maxd Object']['maxd_bedroom']
maxd_baths=maxd_import['Maxd Object']['maxd_baths']
maxd_con=maxd_import['Maxd Object']['maxd_con']
maxd_dist_ob=maxd_import['Maxd Object']['maxd_dist_ob']

matrix_sim_us=pd.read_csv('./Datos/MatrizSimUser.csv',index_col=0)

with open('./Datos/Pesos30MLUsOb.csv', 'r') as f:
        reader = csv.reader(f)
        pesos = list(reader)
p_user=float(pesos[0][0])
vp_us=[]
for i in pesos[2]:
    vp_us.append(float(i))
vp_ob=[]
for i in pesos[4]:
    vp_ob.append(float(i))

# #Importar los pesos
# vector_pesos=pd.read_csv('./Datos/Pesos30Usuarios.csv',index_col=0)
# p_user=vector_pesos['p_user'].iloc[0]
# vp_us=[]
# for i in vector_pesos['vp_us']:
#     if np.isnan(i):
#         break
#     else:
#         vp_us.append(i)
# vp_ob=[]
# for i in vector_pesos['vp_ob']:
#     if np.isnan(i):
#         break
#     else:
#         vp_ob.append(i)

def main(num_casa):   
    #Leemos los datos de las nuevas casas que se quieren recomendar
    print('Realizando la recomendación...')
    DataValidacion = pd.read_csv('CasasValidacion.csv',index_col=0)
    #DataNuevasCasas = pd.read_csv('PruebaCasaInventada.csv',index_col=0)
    sf_house_nuevas=DataValidacion[
        ["Property ID",
        "Bedrooms",
        "Baths",
        "Latitude",
        "Longitude",
        "Construction m2",
        "Parking",
        "Levels",
        "Property Type",
        "Sale Value"]]
    #cambiar a DataValidacion por Casas nuevas
    #Aquí leemos el objeto que queremos recomendar

    #object_target=DataValidacion.loc[DataValidacion['Property ID']==sys.argv[1]].iloc[0]
    object_target=DataValidacion.loc[DataValidacion['Property ID']==num_casa].iloc[0]
    #object_target=DataValidacion.loc[DataValidacion['Property ID']=='IF-999100034'].iloc[0]
    #object_target=DataNuevasCasas.iloc[0]
    list_rec_object_target=Rec_list_user_for_ob(matrix_sim_us=matrix_sim_us,
        sf_ob_Training_box=sf_house_T_b,
        sf_user_Training_box=sf_user_T_b,
        Training_box=Int_DataFrame_Inv,
        object_target=object_target,
        vec_ob=vp_ob,
        peso_us=p_user)
    #Presentar los datos
    print('La lista de los 10 posibles compradores para la casa: ')
    # Wait for 5 seconds
    print('{ \n "responseCode": 1, \n"message": "Success", \n "data": ',object_target[["Property ID",
        "Bedrooms",
        "Baths",
        "Latitude",
        "Longitude",
        "Construction m2",
        "Parking",
        "Levels",
        "Property Type",
        "Sale Value"]].to_json(),'\n  }')
    #time.sleep(5)
    print(' son: ')
    print('{ \n "responseCode": 1, \n "message": "Success", \n "data": [ ')
    cadena_rec='{ \n "responseCode": 1, \n "message": "Success", \n "data": [ '
    for row in range(0,10):
        print('"entity":{ \n')
        cadena_rec=cadena_rec+'"entity": \n '
        tar_us=Int_DataFrame_Inv.loc[Int_DataFrame_Inv['ID']==list_rec_object_target.index[row]].iloc[0]
        print(tar_us[['Age','Investor\'s Name']].to_json(),'\n')
        cadena_rec=(cadena_rec + '{ \n id:' + '"'+tar_us[['ID']].iloc[0]+'"'
            + ', \n "address":'+ '"'+tar_us[['Full Address']].iloc[0]+'"'
            + ', \n "phone":' + '"'+str(tar_us[['Phone']].iloc[0])+'"'
            + ', \n "age":' + str(tar_us[['Age']].iloc[0])
            + ', \n "name":' + '"'+tar_us[['Investor\'s Name']].iloc[0]+'"'
            + '}, \n'+'"score":'+str(list_rec_object_target.iloc[row])+'\n },')   
        print('"score":',list_rec_object_target.iloc[row],'\n },')
        #time.sleep(1)
    cadena_rec=cadena_rec+'\n] \n }'
    print('\n] \n }')
    #programa para guardar el json
    return cadena_rec

#if __name__ == "__main__":
#    main(sys.argv[1])
