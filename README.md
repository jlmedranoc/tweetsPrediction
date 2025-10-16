# TweetsPrediction — Clasificación de desastres

**Autor del notebook:** José Luis Medrano Cerdas

El notebook implementa un flujo completo de *clasificación supervisada de texto* orientado a la detección automática de mensajes sobre desastres reales publicados en Twitter, utilizando técnicas de *Procesamiento de Lenguaje Natural (NLP)* y modelos de *Deep Learning* [1]–[4]. El documento completo y el código fuente se encuentran disponibles en el repositorio de GitHub [TweetsPrediction — Disaster Classification](https://github.com/jlmedranoc/tweetsPrediction) [8].

## Dataset
El análisis se fundamenta en un conjunto de datos compuesto por *10000 tweets* etiquetados manualmente para indicar si el texto reporta un desastre real (`target = 1`) o no (`target = 0`), cada registro incluye la siguiente información:

- `id`: identificador único del tweet.
- `text`: contenido textual del tweet.
- `location`: ubicación desde donde se envió el mensaje.
- `keyword`: palabra clave asociada.
- `target`: etiqueta binaria que indica si el tweet trata sobre un desastre real.

## Metodología
El flujo metodológico siguió las fases clásicas del aprendizaje automático de texto, con especial énfasis en la comparación entre un modelo de red neuronal tradicional *Multilayer Perceptron* referenciado como MLP y una red recurrente del tipo *Long Short-Term Memory* referenciada como LSTM:

1. *Carga y exploración del dataset*. Mediante el repositorio clonado de GitHub se cargó el archivo `tweets.csv`, este proceso fue realizado verificando la integridad de las columnas y analizando la distribución de las clases.

2. *Preprocesamiento*. Los textos se normalizaron a minúsculas, se eliminaron URLs, menciones, hashtags y caracteres no alfabéticos, además, se eliminaron *stopwords* en inglés empleando la lista publicada por Dedhia [5], y se realizó tokenización simple para obtener la distribución de palabras más frecuentes en cada clase.

3. *Representación vectorial*. Para el modelo MLP se utilizó una representación *TF-IDF* de unigramas, bigramas y trigramas, limitando el vocabulario a aproximadamente 50000 características. En el caso de la LSTM, los textos fueron convertidos a secuencias de enteros mediante *tokenization* y *padding*, creando tensores compatibles con PyTorch.

4. *Clasificación con MLP*. El modelo MLP se implementó con *Scikit-learn* utilizando una capa oculta de 256 neuronas y activación ReLU. El entrenamiento se realizó con una partición 80/20 entre conjuntos de entrenamiento y prueba.

5. *Clasificación con LSTM*. La LSTM fue implementada en *PyTorch* incluyendo una capa de embeddings, una capa LSTM bidireccional y una capa densa final para la clasificación binaria, como parte del proceso se entrenó durante 6 épocas con lote de 128 registros, registrando las curvas de pérdida y exactitud en entrenamiento y validación.

6. *Evaluación*. Ambos modelos fueron evaluados con las métricas estándar, tales como el *accuracy, precision, recall* y *F1-score*, tanto globales como por clase, los resultados se complementaron con un análisis comparativo y una discusión sobre el posible sobreajuste en la LSTM. Adicionalmente, con la intención de lograr una mejor comprensión se utilizaron 10 frases previamente construidas, esto para valorar el funcionamiento de los dos modelos.

## Resultados
El modelo *MLP (TF-IDF)* mostró un rendimiento importante con un bajo costo computacional, capturando patrones léxicos y expresiones frecuentes, sin embargo, su limitación radica en la incapacidad de modelar dependencias secuenciales.

Por otro lado, la *LSTM*, demostró capacidad para aprender relaciones contextuales y temporales en el texto, aunque con mayor tiempo de entrenamiento y riesgo de sobreajuste.  

En general, los resultados experimentales confirmaron que ambos enfoques pueden ser efectivos en la clasificación binaria de tweets sobre desastres, dependiendo del equilibrio entre complejidad del modelo y recursos computacionales disponibles.  

## Referencias bibliográficas  
[1] I. Goodfellow, Y. Bengio, and A. Courville, *Deep Learning*. Cambridge, MA, USA: MIT Press, 2016. [Online]. Available: https://mitpress.mit.edu/9780262035613/deep-learning/


[2] S. Bird, E. Klein, and E. Loper, *Natural Language Processing with Python*. Sebastopol, CA, USA: O’Reilly Media, 2009.

[3] F. Pedregosa et al., “Scikit-learn: Machine Learning in Python,” *J. Mach. Learn. Res.*, vol. 12, pp. 2825–2830, 2011.

[4] P. Goldberg, “Understanding LSTM Networks,” *Colah’s Blog*, 2015. [Online]. Available: https://colah.github.io/posts/2015-08-Understanding-LSTMs/

[5] H. Dedhia, “Stop words in 28 languages,” Kaggle, 2018. [Online]. Available: https://www.kaggle.com/datasets/heeraldedhia/stop-words-in-28-languages

[6] A. Paszke et al., “PyTorch: An Imperative Style, High-Performance Deep Learning Library,” *Advances in Neural Information Processing Systems*, 2019.

[7] Scikit-learn Documentation. [Online]. Available: https://scikit-learn.org/stable/

[8] J. L. Medrano Cerdas, “TweetsPrediction” GitHub, 2025. [Online]. Available: https://github.com/jlmedranoc/tweetsPrediction
