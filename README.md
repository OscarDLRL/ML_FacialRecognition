# ML_FacialRecognition
Identificacion de Expresiones faciales de Oscar (y no de Jesse Breaking Bad :( ) 

## Integrantes del Equipo
- Arturo Balboa
- Oscar de la Rosa
- Angel Hernández
- Emiliano Niño
- Rigoberto Soto


## Uso
### Dependencias
Todas las dependencias estan listadas en requirements.txt

    pip install -r requirements.txt

o

    python -m pip install -r requirements.txt


### Ejecucion 
main.py ejecutara un analisis del modelo segun los datos guardados en modelinfo.pkl, entregando la informacion como la precision del modelo y ejemplos de predicciones segun imagenes aleatorias

    python main.py

Si el archivo modelinfo.pkg no es encontrado, debera de ser generado entrenando primero el modelo:

    python main.py -t 


### Definicion de mapeo segun emocion

0 = Asustado
1 = Enojado
2 = Feliz
3 = Gritando enojado
4 = Gritando feliz
5 = MUY feliz
6 = Serio
7 = Sorprendido
8 = Triste
9 = wahhhh

