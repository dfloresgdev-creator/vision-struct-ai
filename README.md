# VisionStruct-AI: Batch Image-to-CSV Extraction Pipeline

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Gemini API](https://img.shields.io/badge/Vision_AI-Google_Gemini_2.5-orange?logo=google)
![Pydantic](https://img.shields.io/badge/Validation-Pydantic_v2-green)
![Output](https://img.shields.io/badge/Export-CSV_%2F_Excel-success)

Herramienta CLI en Python para transformar lotes de imágenes no estructuradas en **tablas de datos limpias (CSV/Excel)** mediante Visión Artificial Multimodal y esquemas tipados con **Pydantic**.

## Perfiles de Extracción Incluidos
1. **`ecommerce`**: Catalogación automática de fotos de productos (categoría, color, material, estado, fondo limpio y generación de títulos SEO).
2. **`document_ocr`**: Extracción de datos contables desde fotos de boletas, recibos y facturas (proveedor, fecha, monto total, moneda).
3. **`structural_audit`**: Inspección técnica de infraestructura, siniestros y archivos patrimoniales (detección de daños en fachadas, apuntalamientos, obstrucción de vías y nivel de severidad).

## Características Técnicas
- **Salidas Estructuradas (`Pydantic`)**: Garantiza respuestas tipadas sin errores de parseo JSON.
- **Tolerancia a Fallos**: Reintentos automáticos con *exponential backoff* ante límites de cuota (Rate Limits) o errores de red.
- **Módulo de Auditoría (`--benchmark`)**: Cruza las respuestas de la IA contra una muestra de control humano (*Ground Truth*) para calcular la exactitud (*Accuracy*) por variable.

## Instalación y Uso

```bash
git clone [https://github.com/dfloresgdev-creator/vision-struct-ai.git](https://github.com/dfloresgdev-creator/vision-struct-ai.git)
cd vision-struct-ai
pip install -r requirements.txt
