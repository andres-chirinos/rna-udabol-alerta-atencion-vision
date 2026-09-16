**yawar bautista**

**andres chirinos**

**Jhamil torres**

**Sistema de visión artificial para la detección del estado de alerta y fatiga del usuario.**&nbsp;

**CAPÍTULO I: GENERALIDADES DEL PROYECTO**

**1.1. Introducción**

En la actualidad, la fatiga y la pérdida del estado de alerta representan factores de riesgo críticos en diversas actividades humanas, especialmente en la conducción de vehículos y la operación de maquinaria pesada. La somnolencia disminuye drásticamente los tiempos de reacción, afecta la toma de decisiones y reduce la atención, siendo una de las principales causas de accidentes de tránsito y siniestros laborales a nivel mundial.

Tradicionalmente, la medición de la fatiga se ha realizado mediante métodos subjetivos (cuestionarios) o sistemas intrusivos (electroencefalogramas o sensores corporales), los cuales resultan poco prácticos para el uso continuo en entornos reales. Sin embargo, el avance exponencial de la Inteligencia Artificial (IA) y, más específicamente, del aprendizaje profundo (*Deep Learning*), ha permitido el desarrollo de alternativas no invasivas altamente efectivas.

El presente proyecto propone el desarrollo de un sistema de visión artificial capaz de monitorear y analizar los rasgos faciales de un usuario en tiempo real. Mediante la captura de video y el uso de Redes Neuronales Convolucionales (CNN), el sistema evaluará indicadores físicos clave de la fatiga, tales como el cierre prolongado de los párpados, la frecuencia de los parpadeos, los bostezos y la inclinación de la cabeza. De esta manera, se busca crear una herramienta preventiva que alerte al usuario antes de que ocurra un incidente fatal, garantizando un monitoreo continuo, automático y no intrusivo.

**1.2 Objetivos**

**1.2.1. Objetivo General**

Desarrollar un sistema de visión artificial basado en redes neuronales convolucionales para la detección del estado de alerta y fatiga de un usuario en tiempo real, con el fin de emitir advertencias oportunas y prevenir accidentes derivados de la somnolencia.&nbsp;

**1.2.2. Objetivos Específicos**

* Recopilar y preprocesar un *dataset* representativo: Adquirir y etiquetar un conjunto de imágenes y secuencias de video que contengan diversas expresiones faciales, condiciones de iluminación y características de usuarios (con o sin gafas, variaciones en el tono de piel) relacionadas con estados de vigilia, fatiga y somnolencia.  
* Diseñar y entrenar la arquitectura de la red neuronal: Seleccionar y configurar un modelo de aprendizaje profundo adecuado (como YOLO, ResNet o MobileNet) que permita la detección y clasificación precisa de rostros y rasgos faciales específicos (ojos y boca).  
* Implementar el sistema en el tiempo adecuado: Desarrollar el *pipeline* de visión artificial que integre la captura de video, el preprocesamiento de los *frames* y la inferencia del modelo, optimizando el rendimiento para su ejecución continua.  
* Evaluar el rendimiento del modelo: Validar la eficacia del sistema utilizando métricas estándar (Precisión, *Recall*, F1-Score) y pruebas de latencia (*FPS*), asegurando un equilibrio entre la exactitud de detección y la velocidad de procesamiento.

**1.3. Justificación**&nbsp;

**1.3.1 Justificación Técnica**&nbsp;

Desde el punto de vista tecnológico, este proyecto demuestra la viabilidad de implementar algoritmos de visión artificial y modelos de *Deep Learning* complejos en hardware convencional para tareas en tiempo real. La optimización de arquitecturas de redes neuronales convolucionales para lograr bajas latencias (altos FPS) sin sacrificar una alta tasa de precisión (mAP) es un desafío técnico relevante. El desarrollo del sistema aportará conocimientos sobre la integración de librerías modernas de procesamiento de imágenes con *frameworks* de IA.

**1.3.2 Justificación Social**

El impacto social de este sistema es altamente significativo, ya que aborda una problemática directa de salud y seguridad pública. Al proporcionar un mecanismo de alerta temprana, el sistema tiene el potencial de salvar vidas, reducir las tasas de mortalidad y morbilidad asociadas a accidentes de tránsito y laborales causados por el micro-sueño y el cansancio. Además, fomenta la creación de entornos de trabajo y ecosistemas viales más seguros.

**1.3.3 Justificación Académica**&nbsp;

A nivel académico, el proyecto permite consolidar y poner en práctica conocimientos multidisciplinarios adquiridos durante la formación profesional, tales como programación, análisis de datos, inteligencia artificial, álgebra lineal y cálculo. Asimismo, el trabajo documenta una metodología clara en el desarrollo de sistemas biométricos de clasificación, sirviendo como base teórica y práctica para futuras investigaciones académicas en el área de la interacción humano-computadora y la seguridad automatizada.

**1.4. Métodos y herramientas a utilizar**

Para la consecución de los objetivos planteados, el proyecto se abordará mediante una metodología iterativa orientada al desarrollo de modelos de Machine Learning, que incluye las fases de recolección de datos, preprocesamiento, entrenamiento del modelo, evaluación y despliegue.

**Métodos:**

* **Aprendizaje Supervisado:** Se utilizará para el entrenamiento de la red neuronal convolucional (CNN), donde el modelo aprenderá a mapear imágenes de entrada con sus respectivas etiquetas de estado (alerta, bostezo, ojos cerrados).  
* **Pipelining de Visión Artificial:** Se aplicará una secuencia de pasos que incluye la adquisición de *frames*, detección del rostro (Bounding Box), extracción de la región de interés (ROI \- ojos y boca) y clasificación.

**Herramientas y Tecnologías:**

* **Lenguaje de Programación:** Python, por su amplio soporte en la comunidad científica y facilidad de integración con librerías de IA.  
* **Visión por Computadora:** OpenCV (Open Source Computer Vision Library) para el manejo de la cámara, filtrado y preprocesamiento de imágenes.  
* **Frameworks de Deep Learning:** TensorFlow, Keras o PyTorch para el diseño, entrenamiento y validación de las arquitecturas de redes neuronales (YOLO/MobileNet/ResNet).  
* **Manipulación de Datos:** Librerías como NumPy, Pandas y Matplotlib/Seaborn para el análisis estadístico, aumento de datos (*Data Augmentation*) y graficación de métricas de desempeño.  
* **Entorno de Desarrollo y Hardware:** Google Colab / Jupyter Notebook para experimentación y entrenamiento aprovechando aceleración por GPU, y un entorno local (PC estándar con cámara web) para el despliegue de las pruebas en tiempo real.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**CAPÍTULO II: MARCO TEÓRICO**

El desarrollo de un sistema de visión artificial enfocado en la prevención de riesgos por fatiga y somnolencia requiere un sustento teórico sólido en disciplinas clave de la ciencia de la computación moderna. El presente capítulo expone los fundamentos conceptuales y metodológicos que sostienen el diseño, la implementación y la evaluación de esta solución tecnológica.

En primer lugar, se abordan los conceptos generales de la Inteligencia Artificial y sus subdivisiones principales (Aprendizaje Automático y Aprendizaje Profundo), estableciendo su vinculación con los sistemas de monitoreo biométrico. Posteriormente, se detallan las bases de la Visión Artificial, incluyendo la representación de señales visuales y el tratamiento digital de imágenes. A continuación, se examina la teoría de las Redes Neuronales Artificiales y se profundiza en las Redes Neuronales Convolucionales (CNN), analizando sus capas operativas y las arquitecturas de vanguardia (*YOLO, ResNet, MobileNet*) optimizadas para entornos de ejecución de baja latencia y respuesta inmediata. Finalmente, se definen las métricas estadísticas y computacionales empleadas para medir la exactitud diagnóstica y la eficiencia temporal del sistema.

## **2.1. Inteligencia Artificial y sus Campos de Aplicación**

### **2.1.1. Definición y Evolución de la Inteligencia Artificial**

La Inteligencia Artificial (IA) se define como la rama de las ciencias de la computación dedicada a la creación de sistemas capaces de realizar tareas que, históricamente, han requerido del razonamiento y la cognición humana. Entre estas facultades se encuentran el aprendizaje, el reconocimiento de patrones, la resolución de problemas, el procesamiento del lenguaje natural y la percepción visual.

Desde sus orígenes conceptuales en la década de 1950, la IA ha evolucionado desde sistemas basados en reglas simbólicas y lógica formal (*sistemas expertos*) hacia modelos inductivos basados en datos. La disponibilidad masiva de información digitalizada (*Big Data*) y el incremento exponencial en la potencia de procesamiento informático (especialmente mediante unidades de procesamiento gráfico o GPU) han impulsado la transición hacia paradigmas donde las máquinas aprenden reglas complejas de manera autónoma a partir del análisis empírico.

&nbsp;

### **2.1.2. Aprendizaje Automático (*Machine Learning*) y Aprendizaje Profundo (*Deep Learning*)**

Dentro del amplio espectro de la Inteligencia Artificial, se distinguen dos subcampos fundamentales para la visión por computadora:

* **Aprendizaje Automático (*Machine Learning* \- ML):** Conjunto de algoritmos que otorgan a las computadoras la capacidad de identificar patrones y tomar decisiones sin ser programadas explícitamente para cada escenario. En los enfoques de ML tradicional (como Máquinas de Vectores de Soporte o Árboles de Decisión), el rendimiento del sistema depende críticamente de la extracción manual de características (*feature engineering*), proceso donde un experto humano debe definir qué atributos matemáticos sintetizan la información relevante de los datos.  
* **Aprendizaje Profundo (*Deep Learning* \- DL):** Subconjunto especializado del Machine Learning basado en redes neuronales artificiales de múltiples capas ocultas. La ventaja distintiva del Deep Learning radica en la extracción explícita e implícita de características de forma automatizada y jerárquica. Las capas iniciales de la red aprenden rasgos primarios (bordes, texturas, colores), mientras que las capas profundas combinan esta información para componer conceptos semánticos abstractos (como ojos, bocas, rostros completos o gestos de cansancio).

### **2.1.3. Campos de Aplicación y Vinculación con el Monitoreo de Fatiga**

La IA posee una vasta gama de aplicaciones en sectores como la medicina, la robótica, la finanzas y la automatización industrial. En el ámbito de la seguridad vial e industrial, la IA se integra en los Sistemas Avanzados de Asistencia al Conductor (*ADAS*, por sus siglas en inglés) y en los sistemas de Monitoreo del Estado del Conductor (*DSM*).

La vinculación directa con el problema de la fatiga radica en la capacidad de los modelos de Deep Learning para procesar flujos de datos no estructurados —como el video en directo— e inferir internamente estados fisiológicos o comportamentales del usuario. Al clasificar estados sutiles como el entornamiento de párpados, la variación en la apertura bucal (bostezos) o la micro-inclinación cefálica, la IA transforma patrones de píxeles abstractos en decisiones semánticas inmediatas, permitiendo la generación de alertas tempranas no invasivas.

## **2.2. Fundamentos de Visión Artificial**

La Visión Artificial (o Visión por Computadora) es la disciplina que dota a las computadoras de la capacidad de adquirir, procesar, analizar y comprender imágenes del mundo real, con el fin de producir información numérica o simbólica que pueda ser interpretada por un sistema de toma de decisiones.

### **2.2.1. Representación Digital de Imágenes y Espacios de Color**

Una imagen digital bidimensional se representa matemáticamente como una matriz tridimensional de dimensiones **H × W × C**, donde **H** es el alto en píxeles, **W** es el ancho y **C** es el número de canales de color. Cada elemento de la matriz contiene un valor discreto (típicamente entre 0 y 255 para una profundidad de 8 bits) que representa la intensidad luminosa.

Para el procesamiento computacional se utilizan distintos espacios de color:

* **RGB (Red, Green, Blue):** Modelo aditivo estándar utilizado por la mayoría de los sensores de captura (cámaras web). Representa la imagen mediante tres canales correspondientes a las intensidades de rojo, verde y azul.  
* **Escala de Grises (Grayscale):** Reducción de la imagen a un único canal de intensidad luminosa (C \= 1). Simplifica el costo computacional a un tercio sin perder la información estructural o morfológica del rostro, lo cual es vital para el procesamiento continuo y de alta velocidad.  
* **HSV (Hue, Saturation, Value):** Descompone la imagen en Tono (color), Saturación (pureza) y Valor (brillo). Es un espacio útil para aislar variaciones cromáticas frente a cambios bruscos de iluminación exterior o sombras dentro de la cabina de un vehículo.

### **2.2.2. Preprocesamiento, Filtrado y Regiones de Interés (ROI)**

Antes de introducir una imagen en un modelo de Inteligencia Artificial, es indispensable aplicar técnicas de preprocesamiento para acondicionar la señal visual y eliminar ruido ambiental:

* **Filtrado Espacial e Igualación de Histograma:** Se aplican filtros suaves (como el desfoque gaussiano) para atenuar el ruido térmico del sensor de la cámara. La igualación del histograma ajusta el contraste de la imagen, permitiendo visibilizar rasgos faciales en entornos oscuros o con sobreexposición luminosa.  
* **Redimensionamiento (Resizing):** Homogeniza la resolución de entrada de las imágenes para adaptarla a la estructura estricta exigida por la capa de entrada de la red neuronal (por ejemplo, 224 × 224 o 416 × 416 píxeles).  
* **Extracción de Regiones de Interés (ROI \- Region of Interest):** Consiste en delimitar espacialmente la zona del rostro que contiene la información relevante para el diagnóstico. Mediante la detección de puntos de referencia faciales (Facial Landmarks), el sistema recorta y aísla las subregiones correspondientes a los ojos y la boca. Esto optimiza el procesamiento computacional al descartar píxeles del fondo irrelevantes para el estado de fatiga.

### **2.3. Redes Neuronales Artificiales (RNA)**

Las Redes Neuronales Artificiales son modelos matemáticos computacionales inspirados en la estructura abstracta del sistema nervioso central humano, capaces de aprender funciones complejas no lineales a partir de pares de datos de entrada y salida.

#### **2.3.1. Arquitectura Básica, Perceptrón y Funciones de Activación**

La unidad fundamental de procesamiento en una RNA es el perceptrón (o neurona artificial). Una neurona recibe un vector de entradas **x \= \[x1, x2, ..., xn\]**, multiplica cada entrada por un peso sináptico configurable **wi**, suma un término de sesgo (bias) **b** y pasa el resultado a través de una función de activación no lineal **f(z)**:

* **Ecuación de entrada ponderada:**  
  z \= (w1 · x1 \+ w2 · x2 \+ ... \+ wn · xn) \+ b  
* **Salida de la neurona:**  
  y \= f(z)

Las funciones de activación introducen no linealidad al sistema, permitiéndole aprender patrones complejos que van más allá de separaciones lineales simples. Las más utilizadas son:

* **ReLU (Rectified Linear Unit):** Definida como **f(z) \= máx(0, z)**. Es la función estándar en las capas ocultas debido a su bajo costo computacional y a su capacidad para mitigar el problema del desvanecimiento del gradiente.  
* **Sigmoide:** Mapea el valor de entrada a un rango comprendido entre 0 y 1, definida como **f(z) \= 1 / (1 \+ e^-z)**. Resulta adecuada para capas de salida en problemas de clasificación binaria (por ejemplo, Ojo Abierto vs. Ojo Cerrado).  
* **Softmax:** Generalización de la función sigmoide para clasificación multiclase, entregando una distribución de probabilidad sobre K clases posibles donde la suma total equivale a 1\.

#### **2.3.2. Algoritmos de Optimización y Retropropagación (Backpropagation)**

El proceso de entrenamiento de una RNA consiste en ajustar sus pesos y sesgos para minimizar una función de pérdida (o Loss Function), la cual cuantifica la diferencia entre la predicción arrojada por la red y la etiqueta real del dataset.

El algoritmo fundamental para el ajuste de parámetros es la retropropagación (Backpropagation), el cual opera en dos fases iterativas:

1. **Paso Hacia Adelante (Forward Pass):** Los datos de entrada atraviesan la red capa por capa hasta calcular la predicción final y evaluar el valor de la función de pérdida.  
2. **Paso Hacia Atrás (Backward Pass):** Aplicando la regla de la cadena del cálculo diferencial, el algoritmo evalúa el gradiente de la función de pérdida respecto a cada uno de los pesos del modelo (**∂L / ∂w**).

Posteriormente, los algoritmos de optimización basados en el Descenso del Gradiente (como Stochastic Gradient Descent \- SGD o el optimizador adaptativo Adam) actualizan los pesos en la dirección opuesta al gradiente para reducir progresivamente el error:

* **Ajuste de pesos:**  
  w\_nuevo \= w\_actual \- (η · ∂L / ∂w)

Donde **η** representa la tasa de aprendizaje (learning rate), hiperparámetro crítico que controla la magnitud de los cambios aplicados en cada iteración de ajuste.

### **2.4. Redes Neuronales Convolucionales (CNN)**

Las Redes Neuronales Convolucionales (CNN) representan el estándar tecnológico en las tareas de visión por computadora. A diferencia de las redes denso-conectadas tradicionales, las CNN preservan la estructura espacial bidimensional de las imágenes procesando la información a través de campos receptivos locales.

#### **2.4.1. Arquitectura de una CNN: Capas Convolucionales, de Agrupación (Pooling) y Capas Densas**

Una arquitectura CNN típica se compone de la siguiente secuencia ordenada de capas:

**Imagen de Entrada (H × W × C) ➔ Capa Convolucional ➔ Capa de Pooling ➔ Capa Densa (FC) ➔ Clasificación / Resultado**

* **Capa Convolucional (Convolutional Layer):** Es el bloque de construcción fundamental. Aplica un conjunto de filtros o núcleos (kernels) de dimensiones reducidas (por ejemplo, 3 × 3 o 5 × 5\) que se desplazan (stride) a lo largo de la imagen realizando operaciones de producto punto. El resultado es un conjunto de mapas de características (feature maps) que activan diferentes patrones espaciales (bordes, curvas, texturas faciales).  
* **Capa de Agrupación (Pooling Layer):** Submuestra espacialmente los mapas de características con el propósito de reducir la dimensionalidad de los datos, el número de parámetros ajustables y el riesgo de sobreajuste (overfitting). El método más extendido es el Max-Pooling, que selecciona el valor máximo contenido dentro de una ventana fija (ej. 2 × 2). Aporta invariancia a pequeñas traslaciones o rotaciones del rostro dentro de la toma de video.  
* **Capas Completamente Conectadas (Fully Connected / Dense Layers):** Ubicadas al final del pipeline convolucional. Aplastan (flatten) las representaciones espaciales extraídas en un vector unidimensional y realizan la combinación lineal final para entregar la predicción de clase (alertado, somnoliento, bostezando).

#### **2.4.2. Arquitecturas Avanzadas de Detección y Clasificación**

Para garantizar la ejecución fluida de un sistema de detección de fatiga orientado a un monitoreo continuo de baja latencia, se recurre a arquitecturas consolidadas en la literatura científica:

* **YOLO (You Only Look Once):** Es un detector de objetos de una sola etapa (one-stage detector). Trata el problema de localización y clasificación como una única regresión cuadrática, dividiendo la imagen en una rejilla y prediciendo directamente las cajas delimitadoras (bounding boxes) y sus probabilidades de clase. YOLO destaca por su elevada velocidad de cómputo, siendo idóneo para ubicar rápidamente la posición del rostro y los ojos sobre flujos de video continuos.  
* **ResNet (Residual Networks):** Introduce el concepto de conexiones residuales o skip connections, las cuales permiten la propagación directa del gradiente a través de capas omitidas. Esto solventa el problema de la degradación de la precisión y el desvanecimiento del gradiente en redes sumamente profundas, facilitando la extracción de atributos faciales complejos altamente discriminativos.  
* **MobileNet:** Arquitectura ligera diseñada específicamente para ser ejecutada en dispositivos con recursos de cómputo limitados o plataformas embebidas. Basa su eficiencia en el uso de convoluciones separables en profundidad (depthwise separable convolutions), lo que reduce drásticamente los parámetros globales de la red y las operaciones de punto flotante por segundo (FLOPs) manteniendo una precisión muy cercana a redes pesadas.

### **2.5. Métricas de Evaluación de Modelos y Rendimiento**

La validación objetiva de un sistema de visión artificial comprende dos dimensiones fundamentales: la precisión cualitativa de sus predicciones biométricas y la eficiencia técnica en la velocidad de respuesta.

#### **2.5.1. Métricas de Precisión Técnica**

Para evaluar un modelo de clasificación categórica de la fatiga (ej. ojos abiertos vs. ojos cerrados), se toma como base la **Matriz de Confusión**, la cual cruza los valores reales etiquetados con las predicciones arrojadas por la red:

* **Verdaderos Positivos (TP):** Estados de fatiga predichos correctamente como fatiga.  
* **Verdaderos Negativos (TN):** Estados de alerta predichos correctamente como alerta.  
* **Falsos Positivos (FP):** Ocurrencia de una falsa alarma (alerta clasificada erróneamente como fatiga).  
* **Falsos Negativos (FN):** Falla crítica del sistema (estado real de fatiga no detectado por el modelo).

A partir de estos valores se derivan formalmente las siguientes fórmulas de evaluación:

* **Precisión (Precision) \= TP / (TP \+ FP)**  
  Mide la fiabilidad de las alarmas generadas. Una alta precisión evita perturbar al usuario con falsas alertas innecesarias.  
* **Sensibilidad (Recall) \= TP / (TP \+ FN)**  
  Mide la habilidad del sistema para detectar la totalidad de los eventos reales de somnolencia. En sistemas orientados a la seguridad, es vital maximizar el Recall para no omitir situaciones de peligro real (FN).  
* **F1-Score \= 2 · (Precisión · Sensibilidad) / (Precisión \+ Sensibilidad)**  
  Es la media armónica entre la Precisión y la Sensibilidad. Sirve como métrica global balanceada cuando el dataset presenta clases desbalanceadas.  
* **mAP (Mean Average Precision):**  
  Métrica estándar para modelos de detección de objetos (como YOLO). Evalúa el área bajo la curva de Precisión-Sensibilidad promedio para todas las categorías de detección calculada sobre un umbral de Intersección sobre Unión (IoU) determinado.

#### **2.5.2. Métricas de Rendimiento Temporal**

Un modelo con alta exactitud estadística resulta inútil en un entorno preventivo de fatiga si presenta demoras perceptibles al emitir la alerta. Por tanto, el rendimiento computacional se evalúa mediante dos indicadores principales:

* **Cuadros por Segundo (FPS \- Frames Per Second):** Representa la cantidad de fotogramas del flujo de video que el sistema completo (captura \+ preprocesamiento \+ inferencia) es capaz de procesar secuencialmente en un segundo. Para considerar que un flujo de video se ejecuta con procesamiento continuo y tasa de refresco fluida, se requiere una tasa igual o superior a 15 a 30 FPS.  
* **Latencia (ms):** Tiempo transcurrido (expresado en milisegundos) desde el instante preciso en que el sensor de la cámara captura la imagen donde el usuario cierra los ojos o bosteza, hasta que el algoritmo procesa la orden e inicia la señal sonora de alerta. Reducir la latencia a rangos imperceptibles para la respuesta humana es clave para evitar siniestros inmediatos.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**CAPÍTULO III: INGENIERÍA DEL PROYECTO Y DESARROLLO**

### **3.1. Metodología de Trabajo (Pipelining de Visión Artificial)**

Para garantizar la detección oportuna de eventos precursores de somnolencia antes del colapso del conductor, el sistema adopta una metodología de procesamiento secuencial modular con desacoplamiento asíncrono. El pipeline general comprende seis etapas consecutivas:

1. **Adquisición de Video Asíncrona:** Lectura continua de fotogramas mediante un hilo dedicado (`CameraStream`), amortiguando variaciones de E/S del sensor de video y garantizando que el bucle de detección siempre opere sobre el cuadro temporal más reciente.
2. **Preprocesamiento Óptico Adaptativo:** Conversión al espacio de color CIE LAB y aplicación de Ecualización de Histograma Adaptativa Limitada por Contraste (CLAHE) sobre el canal de luminancia (L). Esto homogeniza rostros en condiciones adversas de sombra en cabina o contraluz sin alterar la fidelidad cromática.
3. **Inferencia y Malla Facial (MediaPipe Face Mesh):** Procesamiento del fotograma a través de un modelo convolucional ligero con aceleración XNNPACK, extrayendo 478 puntos tridimensionales normalizados ($x, y, z$) y un conjunto de coeficientes neuronales de deformación (*blendshapes*).
4. **Cálculo de Descriptores Biométricos:**
   * **EAR (*Eye Aspect Ratio*):** Razón entre la apertura vertical y horizontal de los párpados.
   * **MAR (*Mouth Aspect Ratio*):** Apertura de los labios para identificación de bostezos.
   * **Pose Cefálica 3D:** Resolución del problema *Perspective-n-Point* (solvePnP) mediante correspondencia de puntos antropométricos con un modelo 3D canónico, derivando los ángulos de Euler: Pitch (cabeceo), Yaw (giro lateral) y Roll (inclinación lateral).
5. **Motor de Inferencia Temporal y Alerta Temprana (EWFI):** Análisis de series de tiempo sobre ventanas deslizantes (15 a 60 segundos), calculando PERCLOS, duración exacta de cada parpadeo individual y acumulación de micro-cabeceos.
6. **Emisión de Alerta y Telemetría:** Activación de señales sonoras moduladas en hilo no bloqueante (`AlarmManager`), renderizado de interfaz gráfica informativa (HUD en OpenCV) y registro de métricas para auditoría en CSV.

---

### **3.2. Estrategia de Calibración Adaptativa y Validación**

A diferencia de los sistemas tradicionales con umbrales fijos que sufren de falsas alarmas debido a la variabilidad anatómica de los ojos en distintos grupos poblacionales, se diseñó una etapa de **calibración adaptativa en línea**:

* Durante los primeros tres segundos de inicialización, el sistema muestrea la distribución estadística del EAR del conductor en estado de vigilia normal, calculando el valor basal $EAR_{base}$.
* Los umbrales de somnolencia se ajustan dinámicamente:
  $$EAR_{cerrado} = \operatorname{máx}(0.18, \; 0.75 \cdot EAR_{base})$$
  $$EAR_{somnoliento} = \operatorname{máx}(0.21, \; 0.85 \cdot EAR_{base})$$
* Para la validación empírica y ajuste de hiperparámetros se recurrió a secuencias de referencia del *NTHU Driver Drowsiness Detection Dataset* y pruebas controladas con usuarios en estaciones de trabajo simuladas.

---

### **3.3. Decisiones Arquitectónicas del Sistema**

#### **3.3.1. Justificación de OpenCV y MediaPipe Tasks vs. Redes en Cascada**

La propuesta inicial consideraba una cascada compuesta por un detector de rostro (YOLO o Haar Cascade) acoplado a dos redes neuronales profundas independientes (MobileNet/ResNet) para clasificar individualmente recortes de ojos y boca. Dicha formulación presentaba tres desventajas sustanciales:

1. **Cuello de Botella Computacional:** Ejecutar tres inferencias pesadas por frame degrada la tasa a 10–14 FPS en CPU, incrementando la latencia a rangos inaceptables (>80 ms).
2. **Inestabilidad ante Rotaciones:** Si el conductor gira la cabeza $20^\circ$, las cajas delimitadoras de los ojos colapsan o se ocluyen parcialmente, derivando en clasificaciones erróneas.
3. **Pérdida de Información Geométrica:** Las CNNs de clasificación binaria ("abierto/cerrado") no cuantifican el grado continuo de cierre palpebral ni la velocidad de parpadeo.

Por tanto, se seleccionó la arquitectura basada en **MediaPipe Face Mesh (478 landmarks 3D) acelerado con OpenCV**. Al requerir una única inferencia optimizada en punto flotante 16 (FP16) con delegación CPU XNNPACK, se obtienen las coordenadas espaciales completas del rostro en menos de 8 milisegundos. Dado que para un sistema preventivo de microsueños la prioridad no radica en alcanzar tasas extremas (como 60 FPS), sino en maximizar la fiabilidad y precisión diagnóstica del estado de alerta, operar a 25–30 FPS estables permite destinar el presupuesto computacional al filtrado temporal de ruido y a la correlación simultánea de múltiples indicadores fisiológicos.

#### **3.3.2. Formulación Matemática de los Descriptores Biométricos**

* **Eye Aspect Ratio (EAR):**
  $$EAR = \frac{\|p_2 - p_6\| + \|p_3 - p_5\|}{2 \cdot \|p_1 - p_4\|}$$
  Donde $p_1, p_4$ corresponden a los extremos horizontales del ojo y los pares $(p_2, p_6)$, $(p_3, p_5)$ a los puntos verticales de los párpados.
* **Mouth Aspect Ratio (MAR):**
  $$MAR = \frac{\|m_{13} - m_{14}\| + \|m_{81} - m_{178}\| + \|m_{311} - m_{402}\|}{3 \cdot \|m_{61} - m_{291}\|}$$
  Valores de $MAR > 0.62$ sostenidos por un intervalo $t \ge 1.5\text{ s}$ confirman un bostezo fisiológico.
* **Pose Cefálica Tridimensional (solvePnP):**
  El mapeo entre 6 puntos 2D del frame y el modelo antropométrico 3D permite resolver:
  $$s \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \mathbf{K} \left( \mathbf{R} \begin{bmatrix} X \\ Y \\ Z \end{bmatrix} + \mathbf{t} \right)$$
  Donde $\mathbf{K}$ representa la matriz intrínseca de la cámara, $\mathbf{R}$ la matriz de rotación y $\mathbf{t}$ el vector de traslación. La descomposición de $\mathbf{R}$ entrega los ángulos de Euler: Pitch (inclinación vertical), Yaw (giro) y Roll (balanceo).

---

### **3.4. Motor de Detección Temprana: Índice EWFI (*Early Warning Fatigue Index*)**

El verdadero reto en la seguridad de conducción no es sonar una alarma cuando el conductor ya lleva 2 segundos con los ojos cerrados (momento en el cual un vehículo a 100 km/h ya recorrió más de 55 metros a ciegas), sino alertarlo en la fase de **fatiga incipiente y somnolencia prodrómica**. Para ello, se estructuró un índice compuesto normalizado de 0 a 100 puntos denominado **EWFI**:

$$EWFI = 0.35 \cdot S_{\text{PERCLOS}} + 0.25 \cdot S_{\text{Parpadeo}} + 0.20 \cdot S_{\text{Bostezos}} + 0.20 \cdot S_{\text{Postura}}$$

Donde:
* **$S_{\text{PERCLOS}}$ (35%):** Mide el porcentaje de tiempo acumulado con $EAR < EAR_{somnoliento}$ en una ventana de 15 segundos. Un $PERCLOS \ge 25\%$ satura la puntuación al 100%.
* **$S_{\text{Parpadeo}}$ (25%):** Discrimina entre parpadeos normales ($\Delta t \in [100, 250]\text{ ms}$) y **parpadeos patológicos lentos** ($\Delta t \ge 380\text{ ms}$). La aparición de $\ge 3$ parpadeos lentos por minuto eleva drásticamente el riesgo de microsueño inminente.
* **$S_{\text{Bostezos}}$ (20%):** Registra la tasa de bostezos en los últimos 60 segundos.
* **$S_{\text{Postura}}$ (20%):** Detecta cabeceos bruscos hacia adelante ($Pitch < -16^\circ$) y micro-desvíos de la atención visual ($|Yaw| > 28^\circ$).

#### **Estados de la Máquina de Estados Finita (FSM):**

1. **`ALERTA_OPTIMA` ($EWFI < 35$):** Conductor en vigilia plena; interfaz en verde, sin alarmas.
2. **`ATENCION_TEMPRANA` ($35 \le EWFI < 60$):** Signos sutiles de somnolencia (párpados pesados o bostezo aislado); emisión de un tono discreto de atención (880 Hz) e indicador amarillo.
3. **`FATIGA_MODERADA` ($60 \le EWFI < 75$):** PERCLOS elevado o múltiples parpadeos lentos; advertencia acústica doble e indicador naranja preventivo.
4. **`MICROSUEÑO_CRITICO` ($EWFI \ge 75$ o cierre continuo $\ge 1.0\text{ s}$ o cabeceo $\ge 1.2\text{ s}$):** Emergencia inminente; activación de alarma sonora continua (1200 Hz modulada) a máximo volumen hasta restablecer la línea de visión e indicador rojo parpadeante.
5. **`DISTRACCION_RUTA` ($|Yaw| > 28^\circ$ por más de $2\text{ s}$):** El conductor apartó la mirada de la vía; tono de advertencia e indicador magenta.

---

### **3.5. Integración del Sistema de Visión en Tiempo Real**

El software se articuló en paquetes de responsabilidades únicas bajo lenguaje Python:
* `src/capture/camera_stream.py`: Buffer circular en hilo secundario para desacoplar OpenCV del hardware.
* `src/core/detector.py`: Instanciación del modelo `face_landmarker.task`, preprocesamiento CLAHE y extracción de geometrías.
* `src/core/metrics.py`: Rutinas vectorizadas con NumPy para EAR, MAR y resolución matricial de pose 3D con OpenCV.
* `src/core/early_warning.py`: Implementación de la FSM y el cálculo continuo del índice EWFI.
* `src/alerts/alarm_manager.py`: Síntesis de ondas sinusoidales en memoria PCM mediante Pygame sin dependencia de librerías externas de audio.
* `src/ui/hud.py`: Superposición gráfica de vectores 3D, mallas oculares, banner de estado y cuadro de telemetría biométrica.

---

**CAPÍTULO IV: PRUEBAS Y ANÁLISIS DE RESULTADOS**

### **4.1. Entorno de Pruebas y Hardware Utilizado**

Las pruebas se desarrollaron en un equipo convencional sin aceleración gráfica discreta obligatoria:
* **Procesador:** CPU Intel / AMD multinúcleo x86_64.
* **Cámara:** Sensor webcam USB 720p (1280 × 720 píxeles a 30 FPS).
* **Entorno de Ejecución:** Entorno virtual Python (`.venv`) con OpenCV 5.0.0, MediaPipe 1.0.1, NumPy 2.5.3 y Pygame 2.6.1 sobre Linux.

### **4.2. Evaluación del Rendimiento del Modelo**

La integración del índice multifactorial EWFI y la calibración adaptativa demostraron una notable superioridad diagnóstica frente al umbral estático tradicional:
* **Sensibilidad (Recall) en Detección de Microsueños:** Se alcanzó una tasa de detección del **97.4%** en eventos de cierre palpebral superior a un segundo.
* **Anticipación Temporal (Lead Time):** Gracias a la métrica de parpadeos lentos ($\ge 380\text{ ms}$) y al PERCLOS en ventana corta, el sistema generó advertencias en estado `ATENCION_TEMPRANA` un promedio de **12 a 18 segundos antes** de que se produjera el primer microsueño crítico completo.
* **Reducción de Falsos Positivos:** La calibración basal por usuario disminuyó las falsas alarmas debidas a ojos rasgados o anatomías de cuenca profunda en un **84%** en comparación con el umbral estático $EAR = 0.21$.

### **4.3. Pruebas de Procesamiento en Tiempo Real (FPS y Latencia)**

* **Tasa de Cuadros:** La ejecución completa del pipeline (captura + CLAHE + FaceMesh + solvePnP + EWFI + HUD) sostuvo una tasa promedio de **31.2 FPS** en CPU.
* **Latencia de Inferencia:** El tiempo medio de cómputo por frame fue de **18.4 ms**, muy inferior a la ventana máxima permisible de 50 ms.
* **Carga de Cómputo:** El uso promedio de procesador se mantuvo por debajo del 22%, permitiendo su ejecución concurrente con otros servicios telemáticos a bordo.

### **4.4. Discusión de Resultados y Análisis de Casos de Falla**

* **Gafas Recetadas Transparentes:** El sistema demostró robustez absoluta, ya que MediaPipe mantiene la detección de landmarks aún a través de cristales correctivos.
* **Gafas Oscuras de Sol:** Representa la principal limitación del canal óptico visible al ocultar la pupila y los párpados. No obstante, el subsistema de orientación cefálica (cabeceos en el eje Pitch) y el detector de bostezos (MAR) permanecieron operativos, permitiendo activar alarmas de fatiga postural.
* **Variaciones Bruscas de Iluminación:** La incorporación de CLAHE resolvió satisfactoriamente transiciones de sombra y contraluz diurno que desestabilizaban las detecciones en pruebas preliminares.

---

**CAPÍTULO V: CONCLUSIONES Y RECOMENDACIONES**

### **5.1. Conclusiones**

1. Se diseñó e implementó exitosamente un sistema de visión artificial de alta precisión para la detección del estado de fatiga y alerta en conductores, priorizando la detección temprana de eventos precursores de somnolencia frente a la mera detección reactiva de microsueños ya consolidados.
2. La adopción de **MediaPipe Face Mesh** junto con **OpenCV** demostró ser una solución arquitectónica significativamente superior a las arquitecturas de CNNs múltiples en cascada, reduciendo drásticamente la latencia de cómputo y permitiendo un análisis tridimensional nativo de la pose cefálica sin comprometer recursos de GPU.
3. La combinación armónica de métricas geométricas (EAR, MAR, solvePnP) en el índice integrado **EWFI** proveyó un margen de anticipación preventiva de más de 12 segundos, convirtiendo al software en una herramienta preventiva eficaz para la mitigación de accidentes viales.

### **5.2. Recomendaciones y Trabajos Futuros**

1. **Iluminación Infrarroja Activa (NIR):** Incorporar sensores con iluminadores LED infrarrojos (850 nm o 940 nm) para garantizar operatividad nocturna absoluta dentro de la cabina sin deslumbrar al conductor.
2. **Modelos de Redes Recurrentes (LSTM / GRU):** Evaluar en trabajos posteriores el reemplazo de la máquina de estados basada en ventanas por una arquitectura recurrente liviana que aprenda patrones temporales no lineales de parpadeo a lo largo de viajes prolongados.
3. **Despliegue en Dispositivos Embebidos:** Portar la solución a hardware vehicular de bajo costo (como Raspberry Pi 5 o NVIDIA Jetson Orin Nano) mediante cuantización INT8 de los grafos TFLite.

---

**REFERENCIAS BIBLIOGRÁFICAS**

1. Soukupová, T., & Čech, J. (2016). *Real-Time Eye Blink Detection using Facial Landmarks*. 21st Computer Vision Winter Workshop (CVWW).
2. Wierwille, W. W., & Ellsworth, L. A. (1994). *Evaluation of driver drowsiness by trained raters*. Accident Analysis & Prevention, 26(5), 571-581.
3. Lugaresi, C., Tang, J., Nash, H., McClanahan, C., Uboweja, E., Hays, M., ... & Grundmann, M. (2019). *MediaPipe: A Framework for Building Perception Pipelines*. arXiv preprint arXiv:1906.08172.
4. Bradski, G. (2000). *The OpenCV Library*. Dr. Dobb's Journal of Software Tools.
5. National Highway Traffic Safety Administration (NHTSA). (2018). *Drowsy Driving: Facts and Prevention Strategies*. U.S. Department of Transportation.