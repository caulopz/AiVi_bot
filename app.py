# -----------------------------------------------------------------------------
# AiVi - Servidor Backend v0.4 (Con Memoria Permanente)
#
# Se modifica el código para que el registro sea persistente. Las nuevas caras
# se guardan y se leen desde la base de datos SQLite 'memoria_aivi.db'.
# -----------------------------------------------------------------------------

import os
import markdown
import fitz
import face_recognition
import numpy as np
import base64
import io
import sqlite3 # <--- NUEVO import para la base de datos
from PIL import Image
from flask import Flask, request, jsonify, render_template, session
from flask_session import Session
from flask_cors import CORS
import google.generativeai as genai

# --- 1. CONFIGURACIÓN GENERAL ---
app = Flask(__name__)
CORS(app)
app.secret_key = "una-clave-secreta-muy-segura"
app.config["SESSION_TYPE"] = "filesystem"
Session(app)
DB_NAME = "memoria_aivi.db" # <--- NUEVO nombre de la base de datos

# --- 2. CEREBRO CONVERSACIONAL (TU CÓDIGO) ---
def extract_pdf_text(pdf_path):
    # ... (tu función sin cambios) ...
    if not os.path.exists(pdf_path): return "Error: PDF no encontrado."
    try:
        doc = fitz.open(pdf_path)
        text = "".join(page.get_text() for page in doc)
        doc.close()
        return text
    except Exception as e: return f"Error al leer PDF: {e}"

COMPANY_DATA_PDF = "empresa_data.pdf"
COMPANY_KNOWLEDGE = extract_pdf_text(COMPANY_DATA_PDF)

try:
    # ... (tu código de Gemini sin cambios) ...
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key: raise ValueError("Variable GEMINI_API_KEY no encontrada.")
    genai.configure(api_key=api_key)
    model_gemini = genai.GenerativeModel("gemini-1.5-flash")
except Exception as e:
    print(f"Error configurando Gemini: {e}")
    model_gemini = None

@app.route("/chat", methods=["POST"])
def handle_chat():
    # ... (tu función sin cambios) ...
    return "Respuesta del chat"

# --- 3. CEREBRO VISUAL (CON MEMORIA PERMANENTE) ---
known_face_encodings = []
known_face_names = []

# --- NUEVA FUNCIÓN para crear la base de datos ---
def setup_database():
    """Crea la tabla en la base de datos si no existe."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    # Usamos UNIQUE en el nombre para evitar duplicados
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS personas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nombre TEXT NOT NULL UNIQUE,
            encoding BLOB NOT NULL
        )
    """)
    conn.commit()
    conn.close()

# --- FUNCIÓN MODIFICADA para cargar desde archivos Y base de datos ---
def load_known_faces():
    """Carga rostros desde la carpeta 'known_faces' Y desde la base de datos."""
    print("Cargando rostros conocidos para AiVi...")
    face_folder = 'known_faces'
    if not os.path.exists(face_folder): os.makedirs(face_folder)
    
    # Cargar desde la carpeta
    for filename in os.listdir(face_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                name = os.path.splitext(filename)[0]
                if name not in known_face_names: # Evitar duplicados si ya está en DB
                    image_path = os.path.join(face_folder, filename)
                    person_image = face_recognition.load_image_file(image_path)
                    person_face_encoding = face_recognition.face_encodings(person_image)[0]
                    known_face_encodings.append(person_face_encoding)
                    known_face_names.append(name)
                    print(f"- Rostro de '{name}' (archivo) cargado.")
            except Exception as e:
                print(f"Error al cargar {filename} de la carpeta: {e}")

    # Cargar desde la base de datos
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT nombre, encoding FROM personas")
        for name, encoding_blob in cursor.fetchall():
            if name not in known_face_names:
                encoding = np.frombuffer(encoding_blob, dtype=np.float64)
                known_face_encodings.append(encoding)
                known_face_names.append(name)
                print(f"- Rostro de '{name}' (base de datos) cargado.")
    except sqlite3.OperationalError:
        print("Tabla 'personas' aún no creada. Se creará al iniciar.")
    conn.close()

@app.route("/analyze_vision", methods=['POST'])
def handle_vision():
    # (Esta función no tiene cambios)
    # ...
    try:
        data = request.json
        if 'image' not in data: return jsonify({'error': 'No se proporcionó imagen'}), 400
        image_data = base64.b64decode(data['image'].split(',')[1])
        frame = np.array(Image.open(io.BytesIO(image_data)))
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        recognized_people = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
            name = "Desconocido"
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
            recognized_people.append(name)
        return jsonify({'people': recognized_people})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# --- FUNCIÓN MODIFICADA para guardar en la base de datos ---
@app.route('/register', methods=['POST'])
def handle_register():
    try:
        data = request.json
        name = data.get('name')
        images_data = data.get('images')
        if not name or not images_data: return jsonify({'error': 'Faltan datos'}), 400

        # ... (cálculo del encoding promedio sin cambios) ...
        new_face_encodings = []
        for image_data in images_data:
            img_bytes = base64.b64decode(image_data.split(',')[1])
            img = np.array(Image.open(io.BytesIO(img_bytes)))
            encodings = face_recognition.face_encodings(img)
            if encodings: new_face_encodings.append(encodings[0])
        if not new_face_encodings: return jsonify({'error': 'No se pudo encontrar una cara'}), 400
        average_encoding = np.mean(new_face_encodings, axis=0)

        # Añadir a la memoria de la sesión actual
        known_face_encodings.append(average_encoding)
        known_face_names.append(name)

        # Guardado permanente en la base de datos
        encoding_bytes = average_encoding.tobytes()
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        # Usamos INSERT OR IGNORE para evitar errores si el nombre ya existe
        cursor.execute("INSERT OR IGNORE INTO personas (nombre, encoding) VALUES (?, ?)", (name, encoding_bytes))
        conn.commit()
        conn.close()

        print(f"¡NUEVA PERSONA GUARDADA EN DB: {name}!")
        return jsonify({'status': 'success', 'message': f'{name} registrado correctamente'})
    except Exception as e:
        print(f"Error durante el registro: {e}")
        return jsonify({'error': str(e)}), 500

# --- 4. RUTA PRINCIPAL Y ARRANQUE ---
@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    setup_database() # Se asegura de que la tabla exista al arrancar
    load_known_faces()
    print(f"Servidor AiVi (con memoria permanente) listo. {len(known_face_names)} rostros en memoria.")
    app.run(host='0.0.0.0', port=5000, debug=True)