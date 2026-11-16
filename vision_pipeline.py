import os
import sys
import time
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Type, Optional, Literal
import pandas as pd
from PIL import Image
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from google import genai
from google.genai import types

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("VisionStruct")


class StructuralAuditSchema(BaseModel):
    dano_fachada: Literal["Sí", "No"] = Field(
        description="Sí si se observan muros caídos, grietas severas o pérdida de fachada."
    )
    apuntalamiento_expuesto: Literal["Sí", "No"] = Field(
        description="Sí si hay puntales/andamios de refuerzo o estructura interna expuesta."
    )
    escombros_via: Literal["Sí", "No"] = Field(
        description="Sí si hay acumulación de escombros o materiales bloqueando el suelo o vereda."
    )
    personas: Literal["Sí", "No"] = Field(
        description="Sí si se distinguen personas, operarios o transeúntes en la escena."
    )
    nivel_severidad: Literal["Bajo", "Medio", "Crítico"] = Field(
        description="Nivel estimado de riesgo estructural."
    )
    diagnostico_tecnico: str = Field(
        description="Resumen técnico de máximo 15 palabras sobre el hallazgo visual."
    )


class EcommerceCatalogSchema(BaseModel):
    categoria_producto: str = Field(description="Categoría principal del artículo (ej. Calzado, Electrónica, Ropa).")
    color_predominante: str = Field(description="Color principal visible del producto.")
    material_estimado: str = Field(description="Material aparente (ej. Cuero, Metal, Plástico, Algodón).")
    fondo_limpio: Literal["Sí", "No"] = Field(description="Sí si el producto tiene fondo blanco o de estudio.")
    estado_visual: Literal["Nuevo", "Usado", "Dañado"] = Field(description="Condición visual del producto.")
    titulo_seo_sugerido: str = Field(description="Título comercial atractivo de máximo 10 palabras para catálogo.")


class DocumentOCRSchema(BaseModel):
    tipo_documento: Literal["Factura", "Boleta", "Recibo", "Otro"]
    emisor_proveedor: str = Field(description="Nombre de la empresa o comercio emisor.")
    fecha_emision: str = Field(description="Fecha en formato YYYY-MM-DD o 'No legible'.")
    monto_total: float = Field(description="Monto total numérico del documento (0.0 si no es legible).")
    moneda: str = Field(description="Código de moneda (CLP, USD, EUR, etc.).")
    legibilidad_alta: Literal["Sí", "No"] = Field(description="Sí si el texto del documento es nítido.")


PROFILES: Dict[str, tuple[Type[BaseModel], str]] = {
    "structural_audit": (
        StructuralAuditSchema,
        "Analiza esta imagen como un perito en ingeniería estructural y auditoría de siniestros."
    ),
    "ecommerce": (
        EcommerceCatalogSchema,
        "Analiza esta fotografía de producto como un especialista en catalogación de E-Commerce."
    ),
    "document_ocr": (
        DocumentOCRSchema,
        "Extrae los datos contables clave de este documento comercial o comprobante."
    ),
}


class VisionBatchProcessor:
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def analyze_image(
        self,
        image_path: Path,
        schema: Type[BaseModel],
        system_instruction: str,
        max_retries: int = 3
    ) -> Optional[Dict]:
        for attempt in range(1, max_retries + 1):
            try:
                with Image.open(image_path) as img:
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=[img, system_instruction],
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=schema,
                            temperature=0.0,
                        ),
                    )
                data = json.loads(response.text)
                return {"archivo": image_path.name, **data, "estado_proceso": "OK"}

            except Exception as e:
                wait_time = 2 ** attempt
                logger.warning(
                    f"Intento {attempt}/{max_retries} fallido en '{image_path.name}': {e}. "
                    f"Reintentando en {wait_time}s..."
                )
                time.sleep(wait_time)

        logger.error(f"Fallo definitivo al procesar '{image_path.name}'.")
        return {"archivo": image_path.name, "estado_proceso": "ERROR"}

    def process_directory(
        self,
        input_dir: Path,
        profile_name: str,
        output_csv: Path
    ) -> pd.DataFrame:
        schema, prompt = PROFILES[profile_name]
        valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}
        image_files = sorted([
            f for f in input_dir.iterdir()
            if f.is_file() and f.suffix.lower() in valid_extensions
        ])

        if not image_files:
            logger.error(f"No se encontraron imágenes válidas en: {input_dir}")
            sys.exit(1)

        logger.info(f"Iniciando lote: {len(image_files)} imágenes | Perfil: '{profile_name}'")
        results: List[Dict] = []

        for idx, img_path in enumerate(image_files, start=1):
            logger.info(f"[{idx}/{len(image_files)}] Procesando: {img_path.name}...")
            row = self.analyze_image(img_path, schema, prompt)
            if row:
                results.append(row)
            time.sleep(0.5)

        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        logger.info(f"Reporte exportado en: {output_csv}")
        return df


def evaluate_accuracy(predictions_df: pd.DataFrame, ground_truth_csv: Path) -> None:
    gt_df = pd.read_csv(ground_truth_csv)
    merged = pd.merge(gt_df, predictions_df, on="archivo", suffixes=("_humano", "_ia"))

    if merged.empty:
        logger.warning("No hay coincidencias de nombres de archivo entre Ground Truth y predicciones.")
        return

    common_cols = [
        col.replace("_humano", "")
        for col in merged.columns
        if col.endswith("_humano")
    ]

    total_checks = 0
    total_matches = 0

    print("\n--- REPORTE DE AUDITORÍA DE PRECISIÓN (AI vs. GROUND TRUTH) ---")
    for col in common_cols:
        h_col = f"{col}_humano"
        ia_col = f"{col}_ia"
        if ia_col in merged.columns:
            matches = (merged[h_col] == merged[ia_col]).sum()
            count = len(merged)
            acc = (matches / count) * 100
            total_checks += count
            total_matches += matches
            print(f" * {col}: {matches}/{count} aciertos ({acc:.1f}%)")

    global_acc = (total_matches / total_checks * 100) if total_checks > 0 else 0
    print(f"EXACTITUD GLOBAL: {total_matches}/{total_checks} ({global_acc:.1f}%)\n")


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="VisionStruct-AI: Extracción masiva de datos desde imágenes hacia CSV."
    )
    parser.add_argument("-i", "--input", type=Path, required=True, help="Carpeta con imágenes.")
    parser.add_argument("-o", "--output", type=Path, default=Path("reporte_vision.csv"), help="CSV de salida.")
    parser.add_argument(
        "-p", "--profile",
        choices=list(PROFILES.keys()),
        default="structural_audit",
        help="Perfil de extracción (structural_audit, ecommerce, document_ocr)."
    )
    parser.add_argument(
        "-b", "--benchmark",
        type=Path,
        default=None,
        help="CSV de Ground Truth opcional para medir exactitud."
    )

    args = parser.parse_args()
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        logger.error("Falta la variable GEMINI_API_KEY en el entorno o archivo .env")
        sys.exit(1)

    processor = VisionBatchProcessor(api_key=api_key)
    df_results = processor.process_directory(args.input, args.profile, args.output)

    if args.benchmark and args.benchmark.exists():
        evaluate_accuracy(df_results, args.benchmark)


if __name__ == "__main__":
    main()
