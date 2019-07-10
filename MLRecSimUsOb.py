import pandas as pd 
import numpy as np
import math
import random 
import matplotlib.pyplot as plt
from scipy import sparse
import csv
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

#Construcción de la matriz de similitud
def Matrix_sim_us (sf_user_Training_box,
    vec_pesos):
    #sf_user_Training_box: dataFrame en pandas con los atributos que se van a utilizar,
    #  esta función usa 'Age', 'Place of Interest', 'Investor\'s Latitude' and 'Investor\'s Longitude'
    #vec_pesos: es un np.array con los pesos
    v_pesos=vec_pesos
    #Definición de constantes para las funciones de similitud
    #Usuario
    maxd_ages=sf_user_Training_box['Age'].max()-sf_user_Training_box['Age'].min()
    list_distancias=[]
    for i in range(0,sf_user_Training_box.shape[0]):
        for j in range(i+1,sf_user_Training_box.shape[0]):
            dis=Distancia(sf_user_Training_box.iloc[j]['Investor\'s Latitude'],
                sf_user_Training_box.iloc[i]['Investor\'s Latitude'],
                sf_user_Training_box.iloc[j]['Investor\'s Longitude'],
                sf_user_Training_box.iloc[i]['Investor\'s Longitude'])
            list_distancias.append(dis)
    maxd_dist_us=np.nanmax(list_distancias)#Distancia maxima entre usuarios
    matrix = np.zeros([sf_user_Training_box.shape[0],sf_user_Training_box.shape[0]])
    for row in range(0,sf_user_Training_box.shape[0]):
        for col in range (row+1,sf_user_Training_box.shape[0]):
            v_val=np.zeros(3)
            v_sim=np.zeros(3)
            if (not(np.isnan(sf_user_Training_box.iloc[row]['Age'])) and not(np.isnan(sf_user_Training_box.iloc[col]['Age']))):
                v_sim[0]=Sim_num(sf_user_Training_box.iloc[row]['Age'],sf_user_Training_box.iloc[col]['Age'],maxd_ages)
                v_val[0]=1
            if (type(np.nan)!=type(sf_user_Training_box.iloc[row]['Place of Interest']) and type(np.nan)!=type(sf_user_Training_box.iloc[col]['Place of Interest'])):
                v_sim[1]=Sim_equal_dif(sf_user_Training_box.iloc[row]['Place of Interest'],sf_user_Training_box.iloc[col]['Place of Interest'])
                v_val[1]=1
            if (not(np.isnan(sf_user_Training_box.iloc[row]['Investor\'s Latitude'])) 
                and not(np.isnan(sf_user_Training_box.iloc[col]['Investor\'s Latitude']))
                and not(np.isnan(sf_user_Training_box.iloc[row]['Investor\'s Longitude']))
                and not(np.isnan(sf_user_Training_box.iloc[col]['Investor\'s Longitude']))):
                v_sim[2]=Sim_place(x1=sf_user_Training_box.iloc[row]['Investor\'s Latitude'],
                    x2=sf_user_Training_box.iloc[col]['Investor\'s Latitude'],
                    y1=sf_user_Training_box.iloc[row]['Investor\'s Longitude'],
                    y2=sf_user_Training_box.iloc[col]['Investor\'s Longitude'],
                    max=maxd_dist_us)
                v_val[2]=1
            #redefinimos los pesos para que solo comparen caracteristicas que existen
            if np.dot(v_val,v_val)<3:
                den=np.dot(v_pesos,v_val)
                v_pesos=v_pesos/den
                simt=np.dot(v_sim,v_pesos)
                v_pesos=vec_pesos
            else:    
                simt=np.dot(v_sim,v_pesos)
            matrix[row][col]=simt
            matrix[col][row]=simt
    #devuelvo un dataframe con índice del número de persona 
    matrix=pd.DataFrame(data=matrix,index=sf_user_Training_box['ID'],columns=sf_user_Training_box['ID'])
    return matrix
def Matrix_sim_ob (sf_ob_Training_box,
    vec_pesos):
    #sf_ob_Training_box: dataFrame en pandas con los atributos que se van a utilizar,
    #  esta función usa 'Sale Value', 'Property Type', 'Latitude', 'Longitude', 'Bedrooms',
    #   'Baths' and 'Construction m2'
    #vec_pesos: es un np.array con los pesos
    #Casas
    maxd_sale_value=sf_ob_Training_box['Sale Value'].max()-sf_ob_Training_box['Sale Value'].min()
    maxd_bedroom=sf_ob_Training_box['Bedrooms'].max()-sf_ob_Training_box['Bedrooms'].min()
    maxd_baths=sf_ob_Training_box['Baths'].max()-sf_ob_Training_box['Baths'].min()
    maxd_con=sf_ob_Training_box['Construction m2'].max()-sf_ob_Training_box['Construction m2'].min()
    list_distancias=[]
    for i in range(0,sf_ob_Training_box.shape[0]):
        for j in range(i+1,sf_ob_Training_box.shape[0]):
            dis=Distancia(sf_ob_Training_box.iloc[j]['Latitude'],
            sf_ob_Training_box.iloc[i]['Latitude'],
            sf_ob_Training_box.iloc[j]['Longitude'],
            sf_ob_Training_box.iloc[i]['Longitude'])
            list_distancias.append(dis)
    maxd_dist_ob=np.nanmax(list_distancias)#Distancia máxima entre casas    
    v_pesos=vec_pesos
    matrix = np.zeros([sf_ob_Training_box.shape[0],sf_ob_Training_box.shape[0]])
    for row in range(0,sf_ob_Training_box.shape[0]):
        for col in range (row+1,sf_ob_Training_box.shape[0]):
            v_val=np.zeros(6)
            v_sim=np.zeros(6)
            if ( not(np.isnan(sf_ob_Training_box.iloc[row]['Sale Value'])) and not(np.isnan(sf_ob_Training_box.iloc[col]['Sale Value']))):
                v_sim[0]=Sim_num(sf_ob_Training_box.iloc[row]['Sale Value'],sf_ob_Training_box.iloc[col]['Sale Value'],maxd_sale_value)
                v_val[0]=1
            if (type(np.nan)!=type(sf_ob_Training_box.iloc[row]['Property Type']) and type(np.nan)!=type(sf_ob_Training_box.iloc[col]['Property Type'])):
                v_sim[1]=Sim_equal_dif(sf_ob_Training_box.iloc[row]['Property Type'],sf_ob_Training_box.iloc[col]['Property Type'])
                v_val[1]=1
            if (not(np.isnan(sf_ob_Training_box.iloc[row]['Latitude'])) 
                and not(np.isnan(sf_ob_Training_box.iloc[col]['Latitude']))
                and not(np.isnan(sf_ob_Training_box.iloc[row]['Longitude']))
                and not(np.isnan(sf_ob_Training_box.iloc[col]['Longitude']))):
                v_sim[2]=Sim_place(x1=sf_ob_Training_box.iloc[row]['Latitude'],
                x2=sf_ob_Training_box.iloc[col]['Latitude'],
                y1=sf_ob_Training_box.iloc[row]['Longitude'],
                y2=sf_ob_Training_box.iloc[col]['Longitude'],
                max=maxd_dist_ob)
                v_val[2]=1
            if ( not(np.isnan(sf_ob_Training_box.iloc[row]['Bedrooms'])) and not(np.isnan(sf_ob_Training_box.iloc[col]['Bedrooms']))):            
                v_sim[3]=Sim_num(sf_ob_Training_box.iloc[row]['Bedrooms'],sf_ob_Training_box.iloc[col]['Bedrooms'],maxd_bedroom)
                v_val[3]=1
            if ( not(np.isnan(sf_ob_Training_box.iloc[row]['Baths'])) and not(np.isnan(sf_ob_Training_box.iloc[col]['Baths']))): 
                v_sim[4]=Sim_num(sf_ob_Training_box.iloc[row]['Baths'],sf_ob_Training_box.iloc[col]['Baths'],maxd_baths)
                v_val[4]=1
            if ( not(np.isnan(sf_ob_Training_box.iloc[row]['Construction m2'])) and not(np.isnan(sf_ob_Training_box.iloc[col]['Construction m2']))): 
                v_sim[5]=Sim_num(sf_ob_Training_box.iloc[row]['Construction m2'],sf_ob_Training_box.iloc[col]['Construction m2'],maxd_con)
                v_val[5]=1
            #redefinimos los pesos para que solo comparen caracteristicas que existen
            if np.dot(v_val,v_val)<6:
                den=np.dot(v_pesos,v_val)
                for xi in v_pesos:
                    xi= (xi/den)
                simt=np.dot(v_sim,v_pesos)
                v_pesos=vec_pesos
            else:    
                simt=np.dot(v_sim,v_pesos)
            matrix[row][col]=simt
            matrix[col][row]=simt
    matrix=pd.DataFrame(data=matrix,index=sf_ob_Training_box['Property ID'],columns=sf_ob_Training_box['Property ID'])
    return matrix

#Esta función devuelve la lista de recomendaciones de casas para un usuario
def Rec_list_user_for_ob (matrix_sim_us,
    matrix_sim_ob,
    sf_user_Training_box,
    Training_features,
    object_target,
    peso_us=0):
    peso_ob=1-peso_us
    #lista de casas recomendadas para el object_target
    #Aquí solo llamaremos la matriz ya construida
    list_sim_object_target=matrix_sim_ob[object_target['Property ID']].sort_values(ascending=False)
    #Construcción de similitud de usuarios
    user_of_object=sf_user_Training_box.loc[sf_user_Training_box['ID'] == Training_features.loc[Training_features['Property ID']==list_sim_object_target.index[0]].iloc[0]['ID']].iloc[0]
    list_sim_user_of_object=matrix_sim_us[user_of_object['ID']].sort_values(ascending=False)
    #Añadimos los usuarios que compraron casas parecidas
    aux=peso_us+peso_ob*list_sim_object_target.iloc[0]
    lt_rec_for_object_target=pd.Series(data=[aux],index=[list_sim_user_of_object.index[-1]])
    for us in range(0,list_sim_user_of_object.shape[0]-1): 
        aux=peso_ob*list_sim_object_target.iloc[0]+peso_us*list_sim_user_of_object.iloc[us]
        lt_rec_for_object_target=lt_rec_for_object_target.append(pd.Series(data=[aux], index=[list_sim_user_of_object.index[us]]))
    #Depués repetimos el procedimiento para el resto de las casas más parecidas con sus compradores
    #Este programa para no repetir datos se esta tardando un par de segundos
    for ob in range(1,list_sim_object_target.shape[0]):
        user_of_object=sf_user_Training_box.loc[sf_user_Training_box['ID'] == Training_features.loc[Training_features['Property ID']==list_sim_object_target.index[ob]].iloc[0]['ID']].iloc[0]
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
            if list_sim_user_of_object.index[us] not in lt_rec_for_object_target.index:
                lt_rec_for_object_target=lt_rec_for_object_target.append(pd.Series(data=[aux], index=[list_sim_user_of_object.index[us]]))
            #Solo quedaría esta
            else:
                if aux>lt_rec_for_object_target[list_sim_user_of_object.index[us]]:
                    lt_rec_for_object_target[list_sim_user_of_object.index[us]]=aux
    #Esta es la lista de los posibles compradores
    lt_rec_for_object_target=lt_rec_for_object_target.sort_values(ascending=False)    
    return lt_rec_for_object_target

#Funcion de error
def Error(lr_user_target,real_house_num):
    count=0
    for element in lr_user_target.index:
        if element == real_house_num:
            break
        else:
            count=count + 1
    return count

#Esta función calcula el error del modelo de recomendación de casas con un peso para usuario y casas dado
def Entrenamiento_error(sf_user_Training_box,
    sf_object_Training_box,
    Training_features,
    peso_us,
    vec_pesos_us,
    vec_pesos_ob):
    #sf_user_Training_box: DataFrame de pandas con las características del usuario
    #sf_object_Training_box: DataFrame de pandas con las características del objecto
    #peso_us: float del pesos que tiene la sim del usuario
    #vec_pesos_us: vector de pesos de las características del usuario
    #vec_pesos_ob: vector de pesos de las características del objecto
    er_norm=0
    #Construcción de las matrices de similitud
    matrix_sim_us=Matrix_sim_us(sf_user_Training_box=sf_user_Training_box,
        vec_pesos=vec_pesos_us)
    matrix_sim_house=Matrix_sim_ob(sf_ob_Training_box=sf_object_Training_box,
        vec_pesos=vec_pesos_ob)
    #print('Se construyeron las matrices de similitud correctamente')
    for obj in range(0,sf_object_Training_box.shape[0]):
        #print('Entramos a evaluar a un object')
        ob_target=sf_object_Training_box.iloc[obj]
        #Similitud con la primera casa del conjunto de entrenamiento
        lt_rec_object_target=Rec_list_user_for_ob(matrix_sim_us=matrix_sim_us,
            matrix_sim_ob=matrix_sim_house,
            sf_user_Training_box=sf_user_Training_box,
            Training_features=Training_features,
            object_target=ob_target,
            peso_us=peso_us)
        #Función de error
        #El usuario real que compró la casa
        real_user_of_object_target=sf_user_Training_box.loc[sf_user_Training_box['ID'] == Training_features.loc[ob_target.name]['ID']].iloc[0]        
        #Error normalizado
        er_norm=er_norm+Error(lt_rec_object_target,real_user_of_object_target['ID'])/lt_rec_object_target.shape[0]
    er_norm=er_norm/sf_user_Training_box.shape[0]
    return er_norm

#Función de entrenamiento usando descenso de gradientes
def Training_DescGrad (p_user,vp_us,vp_ob,
    Training_features,hpar=0.1,batch_size=1,epoca=1):
    #Implementa el descenso de gradientes
    #Training_features: Dataset con los features del usuario y las casas con los que se va a entrenar
    #hpar: Número correspondiente al learning rate hyperparameter
    #batch_size: Número de divisiones en que se parte la data para evaluar el error
    #epoca: Número de veces en que se recorre el dataset completo      
    #Variables
    for num_epocas in range(epoca):
        print('---EPOCA---',num_epocas+1)
        h=0.005
        grad_us=np.zeros(3)
        grad_ob=np.zeros(6)
        contador=0
        error2=0
        error1=0
        list_errores=[] #Guardamos el error en una lista
        div=int(Training_features.shape[0]/batch_size) #numeros de divisiones
        for num_batch in range(batch_size):
            print('Batch',num_batch+1)
            if num_batch==(batch_size-1):
                Training_f=Training_features.iloc[num_batch*div:]
            else:
                Training_f=Training_features.iloc[num_batch*div:(num_batch+1)*div]
            #Creo que sería buena idea seleccionar solo los atributos utiles y trabajar con esas listas para que se más rápido
            f_house_T_b = Training_f[
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
            f_user_T_b = Training_f[
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
            f_user_T_b=f_user_T_b.drop_duplicates(subset ="ID")
            f_house_T_b=f_house_T_b.drop_duplicates(subset="Property ID")    
            error1=Entrenamiento_error(sf_user_Training_box=f_user_T_b,
                        sf_object_Training_box=f_house_T_b,
                        Training_features=Training_f,
                        peso_us=p_user,
                        vec_pesos_us=vp_us,
                        vec_pesos_ob=vp_ob)   
            print('error1:',error1)
            list_errores.append(error1)
            contador=0
            while (contador<20):
                grad = (Entrenamiento_error(sf_user_Training_box=f_user_T_b,
                        sf_object_Training_box=f_house_T_b,
                        Training_features=Training_f,
                        peso_us=p_user+h,
                        vec_pesos_us=vp_us,
                        vec_pesos_ob=vp_ob)-error1)/h
                print(grad)
                for gu in range(len(grad_us)):
                    vp_usg=vp_us
                    vp_usg[gu]=vp_usg[gu]+h
                    grad_us[gu]=(Entrenamiento_error(sf_user_Training_box=f_user_T_b,
                        sf_object_Training_box=f_house_T_b,
                        Training_features=Training_f,
                        peso_us=p_user,
                        vec_pesos_us=vp_usg,
                        vec_pesos_ob=vp_ob)-error1)/h
                print(grad_us)
                for go in range(len(grad_ob)):
                    vp_obg=vp_ob
                    vp_obg[go]=vp_obg[go]+h
                    grad_ob[go]=(Entrenamiento_error(sf_user_Training_box=f_user_T_b,
                        sf_object_Training_box=f_house_T_b,
                        Training_features=Training_f,
                        peso_us=p_user,
                        vec_pesos_us=vp_us,
                        vec_pesos_ob=vp_obg)-error1)/h
                print(grad_ob)
                if (grad == 0 and np.all(grad_us==0)):
                    break
                #Para p_user
                if (p_user - grad*hpar)>=0:
                    if (p_user - grad*hpar)>1:
                        p_user=1
                    else:
                        p_user=p_user-grad*hpar
                else:
                    p_user=0
                #Para vp_us
                for i in range(len(vp_us)):
                    if (vp_us[i]-grad_us[i]*hpar)>=0:
                        vp_us[i]=vp_us[i]-grad_us[i]*hpar
                    else:
                        vp_us[i]=0
                norma=np.sum(vp_us)
                vp_us=vp_us/norma
                #Para vp_ob
                for i in range(len(vp_ob)):
                    if (vp_ob[i]-grad_ob[i]*hpar)>=0:
                        vp_ob[i]=vp_ob[i]-grad_ob[i]*hpar
                    else:
                        vp_ob[i]=0
                norma=np.sum(vp_ob)
                vp_ob=vp_ob/norma
                error2=Entrenamiento_error(sf_user_Training_box=f_user_T_b,
                        sf_object_Training_box=f_house_T_b,
                        Training_features=Training_f,
                        peso_us=p_user,
                        vec_pesos_us=vp_us,
                        vec_pesos_ob=vp_ob)   
                if abs(error2-error1)<=0.001:
                    print('el sistema ya convergió')
                    break
                print('iteracion terminada')
                list_errores.append(error2)
                error1=error2
                contador=contador+1
            #los pesos encontrados son
            print('p_user:',p_user)
            print('vp_us:',vp_us)
            print('vp_ob:',vp_ob)
    return p_user,vp_us,vp_ob,list_errores

#Función para gráficas
#Plot errores
def show_graph_error (data,guardar=False):
    #data: Lista de datos que se quieren graficar
    #guardar: True para guardar, False para no guardar
    plt.plot(data,'go')
    plt.title('Muestra errores, batch_size=10, epoca=2')
    plt.xlabel('Iteración')
    plt.ylabel('Error')
    if guardar:
        plt.savefig('./Graficas/Errores20190708.png')
    plt.show()

#Función para exportar las matrices
def Exportar_matrices(archivo_us,archivo_ob):
    #Nombre del archivo donde se va a exportar
    #Código para exportar las matrices de similitud con los pesos encontrados
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
    sf_user_T_b=sf_user_T_b.drop_duplicates(subset ="Investor's Name")
    sf_house_T_b=sf_house_T_b.drop_duplicates(subset="Property ID")

    #Definición de constantes para las funciones de similitud
    #Usuario
    maxd_ages=sf_user_T_b['Age'].max()-sf_user_T_b['Age'].min()
    list_distancias=[]
    for i in range(0,sf_user_T_b.shape[0]):
        for j in range(i+1,sf_user_T_b.shape[0]):
            dis=Distancia(sf_user_T_b.iloc[j]['Investor\'s Latitude'],sf_user_T_b.iloc[i]['Investor\'s Latitude'],sf_user_T_b.iloc[j]['Investor\'s Longitude'],sf_user_T_b.iloc[i]['Investor\'s Longitude'])
            list_distancias.append(dis)
    maxd_dist_us=np.nanmax(list_distancias)#Distancia maxima entre usuarios
    #Casas
    maxd_sale_value=sf_house_T_b['Sale Value'].max()-sf_house_T_b['Sale Value'].min()
    maxd_bedroom=sf_house_T_b['Bedrooms'].max()-sf_house_T_b['Bedrooms'].min()
    maxd_baths=sf_house_T_b['Baths'].max()-sf_house_T_b['Baths'].min()
    maxd_con=sf_house_T_b['Construction m2'].max()-sf_house_T_b['Construction m2'].min()
    list_distancias=[]
    for i in range(0,sf_house_T_b.shape[0]):
        for j in range(i+1,sf_house_T_b.shape[0]):
            dis=Distancia(sf_house_T_b.iloc[j]['Latitude'],sf_house_T_b.iloc[i]['Latitude'],sf_house_T_b.iloc[j]['Longitude'],sf_house_T_b.iloc[i]['Longitude'])
            list_distancias.append(dis)
    maxd_dist_ob=np.nanmax(list_distancias)#Distancia máxima entre casas

    #Exportar las matrices de similitud
    matrix_sim_us=Matrix_sim_us (sf_user_Training_box=sf_user_T_b,
        vec_pesos=vp_us)
    matrix_sim_us.to_csv(archivo_us)
    matrix_sim_ob=Matrix_sim_ob (sf_ob_Training_box=sf_house_T_b,
        vec_pesos=vp_ob)
    matrix_sim_ob.to_csv(archivo_ob)

#Función para imprimir el error del Test
def Print_error_test (Test_box):
    sf_house_T_b = Test_box[
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
    sf_user_T_b = Test_box[
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
    sf_user_T_b=sf_user_T_b.drop_duplicates(subset ="Investor's Name")
    sf_house_T_b=sf_house_T_b.drop_duplicates(subset="Property ID")
    #Definición de constantes para las funciones de similitud
    #Usuario
    maxd_ages=sf_user_T_b['Age'].max()-sf_user_T_b['Age'].min()
    list_distancias=[]
    for i in range(0,sf_user_T_b.shape[0]):
        for j in range(i+1,sf_user_T_b.shape[0]):
            dis=Distancia(sf_user_T_b.iloc[j]['Investor\'s Latitude'],sf_user_T_b.iloc[i]['Investor\'s Latitude'],sf_user_T_b.iloc[j]['Investor\'s Longitude'],sf_user_T_b.iloc[i]['Investor\'s Longitude'])
            list_distancias.append(dis)
    maxd_dist_us=np.nanmax(list_distancias)#Distancia maxima entre usuarios
    #Casas
    maxd_sale_value=sf_house_T_b['Sale Value'].max()-sf_house_T_b['Sale Value'].min()
    maxd_bedroom=sf_house_T_b['Bedrooms'].max()-sf_house_T_b['Bedrooms'].min()
    maxd_baths=sf_house_T_b['Baths'].max()-sf_house_T_b['Baths'].min()
    maxd_con=sf_house_T_b['Construction m2'].max()-sf_house_T_b['Construction m2'].min()
    list_distancias=[]
    for i in range(0,sf_house_T_b.shape[0]):
        for j in range(i+1,sf_house_T_b.shape[0]):
            dis=Distancia(sf_house_T_b.iloc[j]['Latitude'],sf_house_T_b.iloc[i]['Latitude'],sf_house_T_b.iloc[j]['Longitude'],sf_house_T_b.iloc[i]['Longitude'])
            list_distancias.append(dis)
    maxd_dist_ob=np.nanmax(list_distancias)#Distancia máxima entre casas
    error_test=Entrenamiento_error(sf_user_Training_box=sf_user_T_b,
        sf_object_Training_box=sf_house_T_b,
        Training_features=Test_box,
        peso_us=p_user,
        vec_pesos_us=vp_us,
        vec_pesos_ob=vp_ob)
    print('El error con los datos de prueba es:',error_test)

#Función para la prueba con el conjunto de validación 
def Print_lista_recomendacion(num_ob,matrix_sim_us,matrix_sim_ob):
    #num_ob: el número de la columna con la casa que se quiere imprimir la lista
    object_target=Valid_box.iloc[num_ob]
    #Crear lista de casas recomendadas
    l_rec_object_target=Rec_list_user_for_ob(matrix_sim_us=matrix_sim_us,
        matrix_sim_ob=matrix_sim_ob,
        sf_ob_Training_box=sf_house_T_b,
        Training_features=Valid_box,
        object_target=object_target,
        peso_us=p_user)
    print('La lista de usuarios recomendados para la casa: ')
    print(object_target[['Property ID','Property Type', 'Bedrooms','Sale Value']])
    print(' es: ')
    for row in range(0,10):
        print(row+1, ")")
        print(Valid_box.loc[Valid_box['ID']==l_rec_object_target.index[row]].iloc[0][['ID','Age','Place of Interest']])
    print('El usuario que realmente compro la casa:')
    print(user_target[['ID','Age','Place of Interest']])

#Función para definir si son casos de exito o no
def Presicion(Training_features,
    peso_us,
    vec_pesos_us,
    vec_pesos_ob):
    #Training_features: DataFrame de pandas con las características de usuario y casa
    #peso_us: float del pesos que tiene la sim del usuario
    #vec_pesos_us: vector de pesos de las características del usuario
    #vec_pesos_ob: vector de pesos de las características del objecto
    f_house_T_b = Training_features[
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
    f_user_T_b = Training_features[
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
    f_user_T_b=f_user_T_b.drop_duplicates(subset ="ID")
    f_house_T_b=f_house_T_b.drop_duplicates(subset="Property ID")    
    pres=0
    #Construcción de las matrices de similitud
    matrix_sim_us=Matrix_sim_us(sf_user_Training_box=f_user_T_b,
        vec_pesos=vec_pesos_us)
    matrix_sim_house=Matrix_sim_ob(sf_ob_Training_box=f_house_T_b,
        vec_pesos=vec_pesos_ob)
    for obj in range(f_house_T_b.shape[0]):
        ob_target=f_house_T_b.iloc[obj]
        #Similitud con el primer usuario del conjunto de entrenamiento
        lt_rec_user_target=Rec_list_user_for_ob(matrix_sim_us=matrix_sim_us,
            matrix_sim_ob=matrix_sim_house,
            sf_user_Training_box=f_user_T_b,
            Training_features=Training_features,
            object_target=ob_target,
            peso_us=peso_us)
        #Función de error
        #El usuario que compró la casa 
        real_user_of_ob_target=f_user_T_b.loc[f_user_T_b['ID'] == Training_features.loc[Training_features['Property ID']==ob_target['Property ID']].iloc[0]['ID']].iloc[0]
        #Error normalizado
        if lt_rec_user_target.iloc[:5].index.contains(real_user_of_ob_target['ID']):
            pres=pres+1
    pres=pres/f_house_T_b.shape[0]
    return pres

#----Inicio del Programa-----

#Estos pesos tienen que cambiar en el programa de entrenamiento
#pesos de usuario y casa con los que comenzamos
p_user=0.5
#peso para características de usuario
vp_us=np.ones(3)*1/3.
#Pesos para características de objeto 
vp_ob=np.ones(6)*1/6.
#Exportar los pesos
with open('./Datos/Pesos20190701.csv', 'r') as f:
        reader = csv.reader(f)
        pesos = list(reader)
p_user=float(pesos[0][0])
vp_us=[]
for i in pesos[2]:
    vp_us.append(float(i))
vp_ob=[]
for i in pesos[4]:
    vp_ob.append(float(i))
vp_us=np.array(vp_us)
vp_ob=np.array(vp_ob)

#Leer datos
print('Cargando datos')
Int_DataFrame_Inv = pd.read_csv('inversionistas.csv',index_col='#')
#print(Int_DataFrame_Inv.head())

#Eliminar datos iguales 
al=list([])
for row in Int_DataFrame_Inv:
    al.append(row)
Int_DataFrame_Inv=Int_DataFrame_Inv.drop_duplicates(subset =al)
#Vamos a eliminar las casas que están repetidas
Int_DataFrame_Inv=Int_DataFrame_Inv.drop_duplicates(subset ="Property ID")

#Separación de los datos 80% entrenamiento y 20% prueba
Int_DataFrame_Inv=Int_DataFrame_Inv.sample(frac=1)#shuffle the datas 
#Int_DataFrame_Inv.head()
num_total=Int_DataFrame_Inv.shape[0]
Training_box=Int_DataFrame_Inv.iloc[:int(0.8*num_total)]
Test_box=Int_DataFrame_Inv.iloc[int(0.8*num_total):int(0.9*num_total)]
Valid_box=Int_DataFrame_Inv.iloc[int(0.9*num_total):]
#Entrenamiento tomando como atributos:
#Valid_box.to_csv('CasasValidacion.csv')
#Personas: "Age", "Investor\'s Latitud, Investor\'s Longitud" and "Place of Interest"
#Casas: "Sale Value", "Latitud, Longitud", "Bedrooms", "Baths", "Construction m2" and "Property type"
print("Entrenando...")

#Descenso de gradientes
p_user,vp_us,vp_ob,errores=Training_DescGrad(p_user=p_user,vp_us=vp_us,vp_ob=vp_ob,
    Training_features=Training_box,hpar=0.1,batch_size=10,epoca=2)
#los pesos encontrados son
print('p_user:',p_user)
print('vp_us:',vp_us)
print('vp_ob:',vp_ob)

#Exportar los pesos
with open('./Datos/Pesos20190701.csv', 'w') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow([p_user])
    wr.writerow(vp_us)
    wr.writerow(vp_ob)

""" #Exportar los pesos con pandas
vectorp_uo=pd.Series([p_user])
vectorp_usuarios=pd.Series(vp_us)
vectorp_objetos=pd.Series(vp_ob)
vector_pesos=pd.DataFrame({'p_user':vectorp_uo,'vp_us':vectorp_usuarios,'vp_ob':vectorp_objetos})
vector_pesos.to_csv('./Datos/Pesos30Usuarios.csv')
 """

#Plot errores
show_graph_error(data=errores)

#Exportar las matrices
Exportar_matrices(archivo_us='./Datos/MatrizUs20190708.csv',archivo_ob='./Datos/MatrizOb20190708.csv')

#---Pruebas con el conjunto de Test----
#Se prueba el modelo ya entrenado 
Print_error_test(Test_box)

#---Obtenemos la medida de las casas recomendadas--
presicion=Presicion(Training_features=Valid_box,
    peso_us=p_user,
    vec_pesos_us=vp_us,
    vec_pesos_ob=vp_ob)
print(presicion)

