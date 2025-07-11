# 🤖 Asesor Redondos IA - Predicción y Conocimiento

**Tu asistente inteligente para consulta de documentos, preguntas empresariales y predicción de demanda.**

---

## 🚀 ¿Qué hace esta aplicación?

- **Responde consultas sobre tus documentos internos** (PDF, Word, PPTX) usando inteligencia artificial generativa y búsqueda semántica.
- **Contesta preguntas generales de negocio** como ChatGPT.
- **Predice demanda de materiales usando tu modelo en Azure ML:**
  - Por consulta puntual en el chat: "¿Cuál será la demanda de POLLO para 2025-12-31?"
  - Por carga masiva de Excel con columnas `material` y `fecha`.

---

## 🛠️ ¿Cómo usarla?

1. **Sube tus archivos** (PDF, Word, PowerPoint) para preguntas sobre documentos internos.
2. **Ingresa tu API Key de OpenAI** en la barra lateral.
3. **Haz preguntas** en lenguaje natural:
   - Sobre documentos (“¿Quién es responsable de la política X?”)
   - Sobre temas generales (“¿Qué es gobierno de datos?”)
   - Para predicción de demanda (“¿Cuál será la demanda de PAVO para 2025-12-31?”)
4. **Carga un Excel** para predecir demanda masiva:
   - Debe tener columnas: `material` y `fecha` (ejemplo en `/ejemplos/prediccion_demanda.xlsx`).
5. **Recibe respuestas inteligentes**:
   - El chat indica si la respuesta viene de tus documentos, del conocimiento general IA o es una predicción de demanda ML.

---

## ⚙️ ¿Cómo desplegar?

1. Clona este repositorio:
   ```sh
   git clone https://github.com/tu-usuario/redondos-ia-predictiva.git
   cd redondos-ia-predictiva
