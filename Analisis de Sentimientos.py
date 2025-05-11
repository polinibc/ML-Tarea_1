
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# Dataset de ejemplo: comentarios positivos (1) y negativos (0)
data = {
    'comentario': [
        'Me encantó este producto, es excelente',
        'Muy malo, no lo recomiendo',
        'Súper rápido y funciona perfecto',
        'Terrible experiencia, no volveré a comprar',
        'El servicio fue muy bueno',
        'Producto defectuoso, llegó roto',
        'Buena calidad, satisfecho con la compra',
        'Pésimo servicio, llegó tarde y mal',
        'La atención al cliente fue excelente',
        'Horrible, se descompuso en un día'
    ],
    'sentimiento': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 = positivo, 0 = negativo
}

# Crear DataFrame
df = pd.DataFrame(data)

# Convertir texto en vectores de palabras
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['comentario'])

# Variable objetivo
y = df['sentimiento']

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo
modelo = MultinomialNB()
modelo.fit(X_train, y_train)

# Evaluar el modelo
y_pred = modelo.predict(X_test)
print("=== Reporte de Clasificación ===")
print(classification_report(y_test, y_pred))

# Mostrar matriz de confusión
disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["Negativo", "Positivo"])
plt.title("Matriz de Confusión")
plt.show()

# Probar con un nuevo comentario
nuevo_comentario = ["El producto es una muy bueno, me encanta"]
nuevo_vector = vectorizer.transform(nuevo_comentario)
prediccion = modelo.predict(nuevo_vector)
print("\nComentario:", nuevo_comentario[0])
print("Sentimiento:", "Positivo" if prediccion[0] == 1 else "Negativo")
