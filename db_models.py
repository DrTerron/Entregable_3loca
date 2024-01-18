# =======================================================================================
#                       IEXE Tec - Maestría en Ciencia de Datos 
#                       Productos de Datos. Proyecto Integrador
# =======================================================================================
import math
from datetime import datetime
from model_api import db


# =======================================================================================
# Esta clase mapea una predicción a una tabla en la base de datos mediante la biblioteca
# SQL Alchemy. Consulta la documentación de SQL Alchemy aquí:
# https://flask-sqlalchemy.palletsprojects.com/en/2.x/models/
#
# Las columnas está adaptadas para el modelo de ejemplo de tipos de flores 
# (https://en.wikipedia.org/wiki/Iris_flower_data_set). Modifica el nombre de esta 
# clase para que sea más acorde a lo que hace tu modelo predictivo.
#
# ** IMPORTANTE: ** Cualquier modificación a las bases de datos requiere eliminar el
#       archivo de SQLite3 para que SQL Alchemy pueda reconstruir la base de datos
class Prediction(db.Model):
    """ Una predicción en la base de datos.
    """
    __tablename__ = 'prediction'  # Nombre de la tabla en la base de datos

    # -----------------------------------------------------------------------------------
    # Declaración de columnas de la tabla. Modifica estas propiedades para que sean
    # más acorde a las variables que componen una observación de tu modelo.
    
    # La columna ID será la llave primaria de la predicción
    prediction_id = db.Column('id', db.Integer, primary_key=True)

    sepal_length = db.Column('sepal_length', db.Float, nullable=False)
    sepal_width = db.Column('sepal_width', db.Float, nullable=False)
    petal_length = db.Column('petal_length', db.Float, nullable=False)
    petal_width = db.Column('petal_width', db.Float, nullable=False)
    predicted_class = db.Column('class', db.Text, nullable=False)
    # score = db.Column('score', db.Float, nullable=False)
    # El campo que tiene fecha de creación de este modelo. Por defecto toma la fecha
    # actual del sistema en la zona horarua UTC.
    # https://docs.python.org/3/library/datetime.html
    created_date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    # -----------------------------------------------------------------------------------
    def __init__(self, representation=None):
        """ Construye una Prediccion nueva usando su representación REST
        """
        super(Prediction, self).__init__()
        # Modifica estas líneas para que se guarde en la base de datos la representación
        # de una observación de tu modelo.
        # 
        # ** IMPORTANTE: ** Cualquier modificación a las bases de datos requiere eliminar 
        #     el archivo de SQLite3 para que SQL Alchemy pueda reconstruir la base de datos
        self.sepal_length = representation.get('sepal_length')
        self.sepal_width = representation.get('sepal_width')
        self.petal_length = representation.get('petal_length')
        self.petal_width = representation.get('petal_width')

    # -----------------------------------------------------------------------------------
    def __repr__(self):
        """ Convierte una Predicción a una cadena de texto
        """
        template_str = '<Prediction [{}]: sepal_length={}, sepal_width={}, petal_length={}, petal_width={}, class={}>'
        return template_str.format(
            str(self.prediction_id) if self.prediction_id else 'NOT COMMITED', 
            self.sepal_length, self.sepal_width, self.petal_length, self.petal_width,
            self.predicted_class or 'No calculado'
        )