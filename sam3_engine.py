# sam3_engine.py
import torch
from PIL import Image
from transformers import Sam3Processor, Sam3Model


class Sam3Engine:
    """
    Motor ligero para SAM3.
    - Carga modelo y processor una sola vez.
    - segment_image() recibe PIL, prompt y threshold.
    """

    def __init__(self, model_name: str = "facebook/sam3"):
        # Detectar si hay GPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Sam3Engine] Inicializando SAM3 en dispositivo: {self.device}")

        # Cargar processor y modelo
        self.processor = Sam3Processor.from_pretrained(model_name)
        self.model = Sam3Model.from_pretrained(model_name)

        # Enviar modelo al dispositivo (CPU o GPU)
        self.model.to(self.device)
        self.model.eval()

    def segment_image(self, image_pil: Image.Image, text_prompt: str,
                      score_threshold: float):
        """
        Ejecuta SAM3 para segmentación guiada por texto.

        Devuelve:
        - masks_np: np.ndarray [N, H, W] (0/1)
        - scores_np: np.ndarray [N]
        """
        # Preparar inputs para el modelo
        inputs = self.processor(
            images=image_pil,
            text=text_prompt,
            return_tensors="pt"
        ).to(self.device)

        # Forward sin gradientes (solo inferencia)
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-proceso: obtener instancias segmentadas
        results_list = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=score_threshold,
            mask_threshold=0.5,
            target_sizes=inputs["original_sizes"].tolist()
        )

        result = results_list[0]
        masks = result.get("masks")
        scores = result.get("scores")

        if masks is None or scores is None or masks.shape[0] == 0:
            # No encontró nada
            return None, None

        masks_np = masks.cpu().numpy()
        scores_np = scores.cpu().numpy()

        return masks_np, scores_np
