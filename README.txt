-----+++PROGRAMA DE RECOMENDACIÓN DE REVIMEX+++-----

--Programas:

+ MandarURL.py: es el archivo que se encarga de mandar llamar al resto de los programas cuando se realiza una busqueda en la URL indicada a continuación:
-/Recomendacion/<NUM_CASA>
Manda llamar el programa ListaRecUsCa el cual utiliza el modelo ML ya entrenado para dar una lista de recomendaciones de posible compradores para la casa con ID igual a NUM_CASA
-/Recomendacion/ML/UsforOb
Manda llamar el programa MLRecSimObUs en el cual se entrena al modelo con los datos del archivo Inversionistas para recomendar casas a los usuarios
-Recomendacion/ML/ObforUs
Manda llamar el programa MLRecSimUsOb en el cual se entrena al modelo con los datos del archivo Inversionistas para recomendar usuarios a las casas   

--Archivos:

+ Inversionistas.csv: es el dataframe que se utiliza para entrenar el modelo. De este archivo 80% de los datos se utilizan para entrenar 10% para el Test y 10% para la validación. ESTE ES EL ARCHIVO QUE DEBE SUSTITUIRSE PARA ENTRENAR EL MODELO CON UN NUEVO DATAFRAME
+ CasasNuevas.csv: es el dataframe que se utiliza para dar las recomendaciones de posibles compradores a las casas que no se han vendido. ESTE ARCHIVO DEBE REEMPLAZARSE POR UN ARCHIVO DE CASAS NUEVAS
+ CasasValidacion.csv: es el 10% de los datos de Inversionistas extraidos al azar para probar el programa ListaRecUsOb

--Carpetas:

+ Datos: En esta carpeta se guardan los pesos calculados por los ML, al igual que las matrices de similitud y algunas constantes. Estos datos se utilizan para las recomendaciones finales
+ Graficas: Aqui se guardan las graficas de los errores y pruebas de validación y efectividad de los ML
 