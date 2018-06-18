# PIIC
Proyecto inscrito como programa de investigación PIIC


Predicción de Ozono troposférico a través de información en cubos de datos.

### Posibilidades:
1. Redes recurrentes con convoluciones 2D en sus operaciones
>> RCNN (RNN with Convolutional inside)
2. Redes convolucionales 3D para entrega la salida final.
>> 3D-CNN (CNN with 3D kernel -> to end)
3. Redes convolucionales (2D o 3D) para extraer caract y luego alimentar una red recurrente con vectores 1D. (*two level network*)
>> CRNN (2D or 3D -Convs -> to RNN)
4. Redes convolucionales 2D a cada instante de tiempo, para entregar una salida final
>> 2D-CNN (CNN with 2D kernel -> to end)
5. MLP

Para todos hay que definir el parámetro del largo del cubo ($T$ o *timesteps*). Sin embargo, hay otros parámetros a definir en cada uno de los tipos de posibilidades.
1. size of Conv2D and embedding
2. size of Conv3D (*size of time on kernel*)
3. same as first
4. size of Conv2D and #filters


Inspired: https://github.com/harvitronix/five-video-classification-methods#five-video-classification-methods

