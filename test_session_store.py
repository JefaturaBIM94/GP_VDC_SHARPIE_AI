# test_session_store.py
from session_store import (
    create_session,
    save_session,
    load_session,
    list_sessions,
    IMAGES_DIR,
)

print("=== TEST SESSION STORE ===")

# 1) Simular que tenemos una imagen llamada test_image.jpeg
fake_image_name = "test_image.jpeg"
print(f"Creando sesión para imagen: {fake_image_name!r}")

sess = create_session(image_filename=fake_image_name)
print("Sesión creada:")
print(sess)

# 2) Simular que ya hicimos un análisis con SAM3
print("\nActualizando sesión con resultados de segmentación...")

sess["classes_counts"]["columns"] = 22  # por ejemplo, 22 columnas
sess["segments"].append({
    "class_name": "columns",
    "threshold": 0.5,
    "num_objects": 22,
})

save_session(sess)

print("\nSesión actualizada guardada en disco.")

# 3) Leer de nuevo la sesión desde disco para comprobar
print("\nLeyendo sesión desde disco...")
loaded = load_session(sess["session_id"])
print("Sesión leída:")
print(loaded)

# 4) Listar todas las sesiones disponibles
print("\nListado de todas las sesiones registradas:")
all_sess = list_sessions()
for s in all_sess:
    print(f"- {s['session_id']} | {s['image_filename']} | {s['classes_counts']}")

print("\nRuta base de imágenes asociadas a sesiones:")
print(IMAGES_DIR)

print("\n=== FIN TEST SESSION STORE ===")
