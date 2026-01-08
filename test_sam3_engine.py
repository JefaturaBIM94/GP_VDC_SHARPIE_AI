from PIL import Image
from sam3_engine import Sam3Engine

engine = Sam3Engine()
image_path = "test_image.jpeg"
image = Image.open(image_path).convert("RGB")

prompt = "columns"
threshold = 0.5

print(f"Ejecutando SAM3 sobre '{image_path}' con prompt='{prompt}'...")
masks_np, scores_np = engine.segment_image(image, prompt, threshold)

if masks_np is None:
    print("SAM3 no encontró objetos para ese prompt / threshold.")
else:
    num_instances = masks_np.shape[0]
    print(f"SAM3 encontró {num_instances} instancias.")
    print("Primeros 5 scores:", scores_np[:5])
