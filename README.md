# Obligatorio - Taller Agentes Inteligentes 2022

El objetivo es resolver el juego Pong de Atari utilizando Deep Q Learning y Double Deep Q Learning, comparando sus resultados y lograr que al menos uno de los algoritmos supere el puntaje 10 en el ambiente 'PongNoFrameskip-v4'.
El objetivo del juego es lograr pasar la pelota a traves de la linea del oponente, y que esta no pase a traves de la propia.
El juego utilizado fue (https://www.gymlibrary.ml/environments/atari/pong/) que está programado como un ambiente de OpenAI Gym. 

## Comentarios y conclusiones

En ésta instancia fue posible desarrollar con resultados satisfactorios dos agentes a través de técnicas de Reinforcement Learning puro haciendo uso de redes neuronales en diferentes configuraciones. Al finalizar los 5 millones de pasos de entrenamiento, ambos agentes han aprendido a jugar y ganar puntos en el juego al encontrar ciertas estrategias que garantizan la victoria como se aprecia en el último video y de esa manera explotarlas.

En particular para este tipo de problema cabe resaltar que el uso de una doble red neuronal dio mejores resultados como se pueden apreciar en las distintas gráficas, cosa que no había ocurrido en experimentos anteriores realizados en práctico, como es el caso de Diferencias Temporales, donde habían sido siempre subperformantes respecto al uso de una red neuronal.

Destacar que la solución presentada tiene algunas diferencias con respecto a otras que se pueden encontrar abiertamente ya que no hace uso dos redes neuronales: una red neuronal primaria y otra red neuronal objetivo, sino de una única red con lo que simplifica en parte la lógica del agente.

En los experimentos realizados no se utilizó ninguna técnica de optimización de parámetros y simplemente nos limitamos a entrenar ambos agentes utilizando la misma configuración de parámetros la cual fue:

TOTAL_STEPS = 5000000
EPISODES = 10000
STEPS = 100000

EPSILON_INI = 1
EPSILON_MIN = 0.02
EPSILON_TIME = (EPSILON_INI - EPSILON_MIN) * TOTAL_STEPS
EPISODE_BLOCK = 10
USE_PRETRAINED = False

BATCH_SIZE = 32
BUFFER_SIZE = 10000

GAMMA = 0.99
LEARNING_RATE = 0.0001

Se realizaron varias modificaciones al código original en pos de considerar más y mejores prácticas de programación así como resolver algunos bugs que se detectaron durante el desarrollo de la solución.

Notas:

- Como no fue posible guardar todos los pesos intermedios dejamos los mismos disponibles en el github de la solución, el cual luego de la entrega quedará público para compartir y aportar material a la comunidad.
- Recomendamos abrir la notebook a través de google colaboratory para que el contenido multimedia de la misma se muestre de forma apropiada, ya que al hacerlo de manera local los videos no se mostrarían correctamente. De todas formas dejamos una copia ya preparada a través del siguiente enlace
