# ============================================================
# TRAVELGEN AI
# Aplicación web de planificación personalizada de viajes
# ============================================================

import os
import json
import re
import unicodedata
import subprocess
import sys
from pathlib import Path

import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from google import genai


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="TravelGen AI",
    page_icon="✈️",
    layout="wide"
)

MODELO_GEMINI = "gemini-3.6-flash"

NOMBRE_MODELO_EMBEDDING = (
    "paraphrase-multilingual-MiniLM-L12-v2"
)

RUTA_PROYECTO = Path(
    "/tmp/TravelGen_AI"
)

RUTA_KNOWLEDGE_BASE = (
    RUTA_PROYECTO / "knowledge_base"
)

RUTA_CHROMA = (
    RUTA_PROYECTO / "chroma_db"
)


# ============================================================
# DESCARGA DE RECURSOS
# ============================================================

URL_RECURSOS = (
    "https://drive.google.com/drive/folders/"
    "1Al34AwFInFq_Cp_1mCJJHJ-ZlxEvttxf"
    "?usp=drive_link"
)


@st.cache_resource
def preparar_recursos():

    RUTA_PROYECTO.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Verificar si los recursos ya existen
    # --------------------------------------------------------

    if (
        RUTA_KNOWLEDGE_BASE.exists()
        and RUTA_CHROMA.exists()
    ):
        return True

    # --------------------------------------------------------
    # Instalar gdown si es necesario
    # --------------------------------------------------------

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "gdown"
        ],
        check=True
    )

    # --------------------------------------------------------
    # Descargar recursos
    # --------------------------------------------------------

    resultado = subprocess.run(
        [
            "gdown",
            "--folder",
            URL_RECURSOS,
            "-O",
            str(RUTA_PROYECTO),
            "--quiet"
        ],
        capture_output=True,
        text=True
    )

    if resultado.returncode != 0:

        raise RuntimeError(
            "No fue posible descargar los recursos "
            "de TravelGen AI."
        )

    if not RUTA_KNOWLEDGE_BASE.exists():
        raise FileNotFoundError(
            "No se encontró la Knowledge Base."
        )

    if not RUTA_CHROMA.exists():
        raise FileNotFoundError(
            "No se encontró ChromaDB."
        )

    return True


# ============================================================
# CONFIGURACIÓN DE GEMINI
# ============================================================

def obtener_api_key():

    # Primero intentamos obtenerla desde Streamlit Secrets.
    try:

        api_key = st.secrets.get(
            "GEMINI_API_KEY"
        )

        if api_key:
            return api_key

    except Exception:
        pass

    # Como alternativa, variable de entorno.
    api_key = os.environ.get(
        "GEMINI_API_KEY"
    )

    return api_key


api_key = obtener_api_key()


# ============================================================
# VALIDACIÓN DE API KEY
# ============================================================

if not api_key:

    st.error(
        "No se encontró GEMINI_API_KEY."
    )

    st.info(
        "Configura GEMINI_API_KEY en los Secrets "
        "de Streamlit Community Cloud."
    )

    st.stop()


client = genai.Client(
    api_key=api_key
)


# ============================================================
# CARGAR RECURSOS
# ============================================================

try:

    preparar_recursos()

except Exception as error:

    st.error(
        "No fue posible preparar los recursos "
        "de TravelGen AI."
    )

    st.exception(error)

    st.stop()


# ============================================================
# CARGAR CHROMADB
# ============================================================

@st.cache_resource
def cargar_chromadb():

    chroma_client = chromadb.PersistentClient(
        path=str(RUTA_CHROMA)
    )

    collection = chroma_client.get_collection(
        name="travel_knowledge"
    )

    return chroma_client, collection


chroma_client, collection = cargar_chromadb()


# ============================================================
# CARGAR MODELO DE EMBEDDINGS
# ============================================================

@st.cache_resource
def cargar_embedding_model():

    return SentenceTransformer(
        NOMBRE_MODELO_EMBEDDING
    )


embedding_model = cargar_embedding_model()


# ============================================================
# NORMALIZACIÓN DE DESTINOS
# ============================================================

def normalizar_destino(destino):

    if not destino:
        return ""

    destino = (
        str(destino)
        .strip()
        .lower()
    )

    destino = unicodedata.normalize(
        "NFD",
        destino
    )

    destino = "".join(
        caracter
        for caracter in destino
        if unicodedata.category(caracter) != "Mn"
    )

    destino = re.sub(
        r"[\s\-]+",
        "_",
        destino
    )

    destino = re.sub(
        r"[^a-z0-9_]",
        "",
        destino
    )

    destino = re.sub(
        r"_+",
        "_",
        destino
    )

    destino = destino.strip("_")

    return destino


# ============================================================
# EVALUACIÓN DE SUFICIENCIA DEL CONTEXTO
# ============================================================

def evaluar_suficiencia_contexto(
    consulta,
    resultados
):

    if not resultados:
        return False

    contexto = "\n\n".join(
        [
            (
                f"DOCUMENTO: {resultado.get('documento', '')}\n"
                f"FRAGMENTO: {resultado.get('fragmento', '')}\n"
                f"INFORMACIÓN:\n{resultado.get('texto', '')}"
            )
            for resultado in resultados
        ]
    )

    prompt = f"""
Eres un evaluador de contexto para un sistema RAG
especializado en planificación de viajes.

Determina si la información recuperada desde la base de
conocimiento es suficiente para atender la consulta.

CONSULTA DEL USUARIO:

{consulta}

INFORMACIÓN RECUPERADA:

{contexto}

CRITERIOS:

- La información debe ser relevante para el destino.
- Debe contener información turística útil.
- Debe tener relación con las necesidades expresadas
  por el viajero.
- No es necesario que todos los datos solicitados estén
  presentes.
- Si existe información relevante y utilizable, responde
  SUFICIENTE.
- Si la información es insuficiente o irrelevante, responde
  INSUFICIENTE.

RESPONDE ÚNICAMENTE CON UNA DE ESTAS DOS PALABRAS:

SUFICIENTE

INSUFICIENTE
"""

    interaction = client.interactions.create(
        model=MODELO_GEMINI,
        input=prompt
    )

    respuesta = (
        interaction.output_text
        .strip()
        .upper()
        .replace(".", "")
        .strip()
    )

    if respuesta == "SUFICIENTE":
        return True

    if respuesta == "INSUFICIENTE":
        return False

    return False


# ============================================================
# RETRIEVER RAG
# ============================================================

def recuperar_informacion_v3(
    consulta,
    destino,
    numero_resultados=3
):

    destino_normalizado = normalizar_destino(
        destino
    )

    embedding_consulta = embedding_model.encode(
        consulta,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).tolist()

    resultados = collection.query(
        query_embeddings=[embedding_consulta],
        n_results=numero_resultados,
        where={
            "destino": destino_normalizado
        }
    )

    documentos = resultados["documents"][0]
    metadatos = resultados["metadatas"][0]
    distancias = resultados["distances"][0]

    resultados_formateados = []

    for documento, metadata, distancia in zip(
        documentos,
        metadatos,
        distancias
    ):

        resultados_formateados.append(
            {
                "fuente": "base_conocimiento",
                "texto": documento,
                "documento": metadata["documento"],
                "destino": metadata["destino"],
                "fragmento": metadata["fragmento"],
                "distancia": distancia
            }
        )

    suficiente = evaluar_suficiencia_contexto(
        consulta=consulta,
        resultados=resultados_formateados
    )

    return (
        resultados_formateados,
        suficiente
    )


# ============================================================
# RAG HÍBRIDO
# ============================================================

def recuperar_contexto_hibrido(
    consulta,
    destino,
    numero_resultados=3
):

    resultados_rag, suficiente = (
        recuperar_informacion_v3(
            consulta=consulta,
            destino=destino,
            numero_resultados=numero_resultados
        )
    )

    if suficiente:

        return (
            resultados_rag,
            "base_conocimiento",
            True
        )

    interaction = client.interactions.create(
        model=MODELO_GEMINI,
        input=f"""
Busca información turística relevante y actualizada para
complementar la información disponible sobre el destino.

DESTINO:
{destino}

CONSULTA:
{consulta}

La información será utilizada posteriormente por otro módulo
para generar un itinerario.

Utiliza fuentes confiables y no inventes información.
""",
        tools=[
            {
                "type": "google_search"
            }
        ]
    )

    informacion_web = (
        interaction.output_text.strip()
    )

    contexto_web = [
        {
            "fuente": "web",
            "documento": "Google Search",
            "texto": informacion_web
        }
    ]

    return (
        contexto_web,
        "web",
        bool(informacion_web)
    )


# ============================================================
# CONSTRUCCIÓN DEL PROMPT
# ============================================================

def construir_prompt_hibrido(
    destino,
    duracion,
    presupuesto,
    intereses,
    restricciones,
    tipo_viajero,
    contexto
):

    bloques_contexto = []

    for resultado in contexto:

        if resultado["fuente"] == "base_conocimiento":

            bloques_contexto.append(
                f"""
FUENTE: BASE DE CONOCIMIENTO
Documento: {resultado["documento"]}
Información:
{resultado["texto"]}
"""
            )

        elif resultado["fuente"] == "web":

            bloques_contexto.append(
                f"""
FUENTE: INTERNET
Origen: Google Search
Información:
{resultado["texto"]}
"""
            )

    contexto_texto = "\n".join(
        bloques_contexto
    )

    prompt = f"""
Eres TravelGen AI, un planificador de viajes personalizado.

Tu tarea es generar un itinerario práctico, coherente y
adaptado a las preferencias del viajero.

UTILIZA LA SIGUIENTE INFORMACIÓN DEL USUARIO:

- Destino: {destino}
- Duración: {duracion} días
- Presupuesto: {presupuesto}
- Tipo de viajero: {tipo_viajero}
- Intereses: {", ".join(intereses)}
- Restricciones: {", ".join(restricciones)}

INFORMACIÓN RECUPERADA:

{contexto_texto}

INSTRUCCIONES:

1. Genera un itinerario organizado exactamente en {duracion} días.
2. Adapta las actividades a los intereses del usuario.
3. Respeta todas las restricciones indicadas.
4. Utiliza la información de la BASE DE CONOCIMIENTO como
   fuente principal cuando esté disponible.
5. Utiliza la información proveniente de INTERNET únicamente
   como complemento cuando la base de conocimiento no sea
   suficiente.
6. No atribuyas información de Internet a los documentos de
   la base de conocimiento.
7. No inventes información que no aparezca en el contexto.
8. Si la información disponible no permite responder algún
   aspecto, indícalo claramente.
9. Ten en cuenta el presupuesto indicado por el usuario.
10. Prioriza recomendaciones relevantes para el perfil del
    viajero.
11. Mantén coherencia entre las actividades y la duración
    del viaje.
12. Devuelve exclusivamente un objeto JSON válido.
13. No incluyas Markdown, explicaciones ni bloques de código
    fuera del JSON.

La estructura debe contener:

- destino
- duracion_dias
- presupuesto
- tipo_viajero
- resumen
- dias
- presupuesto_estimado
- recomendaciones

Cada día debe contener:
- dia
- titulo
- actividades

Cada actividad debe contener:
- hora
- actividad
- descripcion
- motivo

El campo presupuesto_estimado debe contener:
- transporte
- alimentacion
- actividades
- total
"""

    return prompt


# ============================================================
# CONVERSIÓN DE PRESUPUESTO
# ============================================================

def convertir_presupuesto_a_numero(
    presupuesto
):

    if presupuesto is None:
        return 0

    texto = (
        str(presupuesto)
        .lower()
        .strip()
    )

    if (
        "millón" in texto
        or "millon" in texto
    ):

        texto_limpio = (
            texto
            .replace("millones", "")
            .replace("millón", "")
            .replace("millon", "")
            .replace("de pesos", "")
            .replace("pesos", "")
            .replace("cop", "")
            .replace("$", "")
            .strip()
        )

        texto_limpio = (
            texto_limpio
            .replace(",", ".")
        )

        try:

            valor = float(
                texto_limpio
            )

            return int(
                valor * 1_000_000
            )

        except ValueError:

            return 0

    texto_limpio = (
        texto
        .replace("$", "")
        .replace("cop", "")
        .replace("pesos", "")
        .replace(".", "")
        .replace(",", "")
        .strip()
    )

    try:

        return int(
            float(texto_limpio)
        )

    except ValueError:

        return 0


# ============================================================
# VALIDACIÓN ESTRUCTURAL
# ============================================================

def validar_itinerario(
    itinerario,
    destino_esperado,
    duracion_esperada
):

    errores = []

    campos_obligatorios = [
        "destino",
        "duracion_dias",
        "dias",
        "presupuesto_estimado",
        "recomendaciones"
    ]

    for campo in campos_obligatorios:

        if campo not in itinerario:

            errores.append(
                f"Falta el campo obligatorio: '{campo}'."
            )

    if errores:
        return False, errores

    destino_generado = str(
        itinerario["destino"]
    ).strip().lower()

    destino_comparacion = str(
        destino_esperado
    ).strip().lower()

    if destino_generado != destino_comparacion:

        errores.append(
            f"El destino generado es "
            f"'{itinerario['destino']}', "
            f"pero se esperaba "
            f"'{destino_esperado}'."
        )

    duracion_generada = (
        itinerario["duracion_dias"]
    )

    if duracion_generada != duracion_esperada:

        errores.append(
            f"Se solicitaron {duracion_esperada} días, "
            f"pero el modelo generó "
            f"{duracion_generada}."
        )

    dias = itinerario["dias"]

    if not isinstance(dias, list):

        errores.append(
            "El campo 'dias' no tiene formato de lista."
        )

        return False, errores

    if len(dias) != duracion_esperada:

        errores.append(
            f"Se esperaban {duracion_esperada} días "
            f"en la lista, pero se encontraron "
            f"{len(dias)}."
        )

    numeros_dias = [
        dia.get("dia")
        for dia in dias
        if isinstance(dia, dict)
    ]

    numeros_esperados = list(
        range(1, duracion_esperada + 1)
    )

    if numeros_dias != numeros_esperados:

        errores.append(
            f"La numeración de los días es incorrecta."
        )

    campos_actividad = [
        "hora",
        "actividad",
        "descripcion",
        "motivo"
    ]

    for dia in dias:

        if not isinstance(dia, dict):

            errores.append(
                "Se encontró un elemento de 'dias' "
                "con formato incorrecto."
            )

            continue

        actividades = dia.get(
            "actividades",
            []
        )

        if not isinstance(
            actividades,
            list
        ):

            errores.append(
                f"El día {dia.get('dia')} no tiene "
                f"una lista válida de actividades."
            )

            continue

        if not actividades:

            errores.append(
                f"El día {dia.get('dia')} "
                f"no contiene actividades."
            )

            continue

        for actividad in actividades:

            if not isinstance(
                actividad,
                dict
            ):

                errores.append(
                    f"El día {dia.get('dia')} contiene "
                    f"una actividad con formato incorrecto."
                )

                continue

            for campo in campos_actividad:

                if not actividad.get(campo):

                    errores.append(
                        f"El día {dia.get('dia')} tiene "
                        f"una actividad sin '{campo}'."
                    )

    return (
        len(errores) == 0,
        errores
    )


# ============================================================
# VALIDACIÓN DE PRESUPUESTO
# ============================================================

def validar_presupuesto(
    itinerario,
    presupuesto_maximo
):

    errores = []

    presupuesto_generado = (
        itinerario
        .get("presupuesto_estimado", {})
        .get("total")
    )

    if not presupuesto_generado:

        errores.append(
            "No se encontró el presupuesto total generado."
        )

        return False, errores

    valor_generado = (
        convertir_presupuesto_a_numero(
            presupuesto_generado
        )
    )

    if valor_generado is None:

        errores.append(
            "No fue posible interpretar el presupuesto."
        )

        return False, errores

    if valor_generado > presupuesto_maximo:

        errores.append(
            f"El presupuesto generado de "
            f"${valor_generado:,} COP supera el límite de "
            f"${presupuesto_maximo:,} COP."
        )

    return (
        len(errores) == 0,
        errores
    )


# ============================================================
# VALIDACIÓN DE RESTRICCIONES
# ============================================================

def validar_restricciones(
    itinerario,
    restricciones
):

    errores = []

    texto_itinerario = (
        str(itinerario)
        .lower()
    )

    for restriccion in restricciones:

        restriccion = (
            restriccion
            .lower()
        )

        if "niños" in restriccion:

            indicadores = [
                "niños",
                "familia",
                "infantil",
                "niño"
            ]

            if not any(
                indicador in texto_itinerario
                for indicador in indicadores
            ):

                errores.append(
                    "No se encontró evidencia de adaptación "
                    "del itinerario para niños."
                )

        elif "caminatas largas" in restriccion:

            indicadores = [
                "caminatas largas",
                "caminatas",
                "evita caminatas",
                "evitar caminatas",
                "transporte",
                "taxi",
                "vehículo",
                "traslado"
            ]

            if not any(
                indicador in texto_itinerario
                for indicador in indicadores
            ):

                errores.append(
                    "No se encontró evidencia de que se haya "
                    "considerado la restricción de evitar "
                    "caminatas largas."
                )

    return (
        len(errores) == 0,
        errores
    )


# ============================================================
# CONTROL DE CALIDAD COMPLETO
# ============================================================

def validar_itinerario_v4(
    itinerario,
    destino_esperado,
    duracion_esperada,
    presupuesto_maximo,
    restricciones
):

    errores = []

    resultado_estructura, errores_estructura = (
        validar_itinerario(
            itinerario=itinerario,
            destino_esperado=destino_esperado,
            duracion_esperada=duracion_esperada
        )
    )

    errores.extend(
        errores_estructura
    )

    resultado_presupuesto, errores_presupuesto = (
        validar_presupuesto(
            itinerario=itinerario,
            presupuesto_maximo=presupuesto_maximo
        )
    )

    errores.extend(
        errores_presupuesto
    )

    resultado_restricciones, errores_restricciones = (
        validar_restricciones(
            itinerario=itinerario,
            restricciones=restricciones
        )
    )

    errores.extend(
        errores_restricciones
    )

    return (
        len(errores) == 0,
        errores
    )


# ============================================================
# MOTOR PRINCIPAL
# ============================================================

def generar_itinerario_v4(
    destino,
    duracion,
    presupuesto,
    intereses,
    restricciones,
    tipo_viajero
):

    consulta = f"""
Información turística necesaria para planificar un viaje a
{destino}.

Intereses principales:
{", ".join(intereses)}

Busca información sobre lugares, actividades, experiencias
y gastronomía relacionados con estos intereses.
"""

    contexto, fuente, informacion_suficiente = (
        recuperar_contexto_hibrido(
            consulta=consulta,
            destino=destino,
            numero_resultados=3
        )
    )

    prompt = construir_prompt_hibrido(
        destino=destino,
        duracion=duracion,
        presupuesto=presupuesto,
        intereses=intereses,
        restricciones=restricciones,
        tipo_viajero=tipo_viajero,
        contexto=contexto
    )

    interaction = client.interactions.create(
        model=MODELO_GEMINI,
        input=prompt
    )

    respuesta = (
        interaction.output_text
    )

    try:

        itinerario = json.loads(
            respuesta
        )

    except json.JSONDecodeError:

        return {
            "exito": False,
            "error": (
                "Gemini no devolvió un JSON válido."
            ),
            "respuesta_original": respuesta,
            "fuente": fuente
        }

    resultado_validacion, errores = (
        validar_itinerario_v4(
            itinerario=itinerario,
            destino_esperado=destino,
            duracion_esperada=duracion,
            presupuesto_maximo=(
                convertir_presupuesto_a_numero(
                    presupuesto
                )
            ),
            restricciones=restricciones
        )
    )

    return {
        "exito": resultado_validacion,
        "itinerario": itinerario,
        "fuente": fuente,
        "informacion_suficiente": (
            informacion_suficiente
        ),
        "errores": errores
    }


# ============================================================
# INTERPRETACIÓN DE LA SOLICITUD
# ============================================================

def interpretar_solicitud_viaje(
    consulta_usuario
):

    prompt = f"""
Eres el módulo de interpretación de TravelGen AI.

Analiza la solicitud del usuario y extrae únicamente la
información necesaria para planificar un viaje.

SOLICITUD DEL USUARIO:
{consulta_usuario}

Extrae:

- destino
- duración_dias
- presupuesto
- tipo_viajero
- intereses
- restricciones

REGLAS:

1. No inventes información que el usuario no haya proporcionado.
2. Si un dato no aparece, utiliza null.
3. Los intereses deben ser una lista.
4. Las restricciones deben ser una lista.
5. duración_dias debe ser un número entero cuando esté disponible.
6. Devuelve exclusivamente JSON válido.
7. No incluyas explicaciones fuera del JSON.

FORMATO:

{{
    "destino": null,
    "duracion_dias": null,
    "presupuesto": null,
    "tipo_viajero": null,
    "intereses": [],
    "restricciones": []
}}
"""

    interaction = client.interactions.create(
        model=MODELO_GEMINI,
        input=prompt
    )

    respuesta = (
        interaction.output_text
    )

    try:

        return json.loads(
            respuesta
        )

    except json.JSONDecodeError:

        return {
            "error": (
                "No fue posible interpretar "
                "la solicitud."
            ),
            "respuesta_original": respuesta
        }


# ============================================================
# VALIDACIÓN DE PREFERENCIAS
# ============================================================

def validar_preferencias_viaje(
    preferencias
):

    campos_obligatorios = {
        "destino": "el destino",
        "duracion_dias": "la duración del viaje",
        "presupuesto": "el presupuesto aproximado"
    }

    faltantes = []

    for campo, descripcion in (
        campos_obligatorios.items()
    ):

        valor = preferencias.get(
            campo
        )

        if (
            valor is None
            or str(valor).strip() == ""
        ):

            faltantes.append(
                descripcion
            )

    if faltantes:

        return False, faltantes

    return True, []


# ============================================================
# GENERAR PREGUNTA POR DATOS FALTANTES
# ============================================================

def generar_pregunta_datos_faltantes(
    preferencias,
    datos_faltantes
):

    contexto = []

    if preferencias.get(
        "destino"
    ):

        contexto.append(
            f"destino: {preferencias['destino']}"
        )

    if preferencias.get(
        "intereses"
    ):

        contexto.append(
            f"intereses: "
            f"{', '.join(preferencias['intereses'])}"
        )

    contexto_texto = "; ".join(
        contexto
    )

    prompt = f"""
Eres TravelGen AI, un planificador de viajes.

El usuario proporcionó la siguiente información:

{contexto_texto}

Todavía falta la siguiente información:
{", ".join(datos_faltantes)}

Formula una pregunta breve, natural y amable para solicitar
únicamente los datos que faltan.

No vuelvas a preguntar información que el usuario ya proporcionó.
No inventes datos.
No utilices lenguaje técnico.
No devuelvas JSON.
Devuelve únicamente la pregunta que verá el usuario.
"""

    interaction = client.interactions.create(
        model=MODELO_GEMINI,
        input=prompt
    )

    return (
        interaction.output_text.strip()
    )


# ============================================================
# PROCESAR SOLICITUD
# ============================================================

def procesar_solicitud_usuario(
    consulta_usuario
):

    preferencias = (
        interpretar_solicitud_viaje(
            consulta_usuario
        )
    )

    if "error" in preferencias:

        return {
            "tipo": "error",
            "mensaje": preferencias["error"]
        }

    datos_validos, datos_faltantes = (
        validar_preferencias_viaje(
            preferencias
        )
    )

    if not datos_validos:

        pregunta = (
            generar_pregunta_datos_faltantes(
                preferencias=preferencias,
                datos_faltantes=datos_faltantes
            )
        )

        return {
            "tipo": "pregunta",
            "mensaje": pregunta,
            "preferencias": preferencias
        }

    resultado = (
        generar_itinerario_v4(
            destino=preferencias["destino"],
            duracion=preferencias["duracion_dias"],
            presupuesto=preferencias["presupuesto"],
            intereses=preferencias["intereses"],
            restricciones=preferencias["restricciones"],
            tipo_viajero=preferencias["tipo_viajero"]
        )
    )

    if not resultado["exito"]:

        return {
            "tipo": "error",
            "mensaje": (
                "No fue posible generar un "
                "itinerario válido."
            ),
            "errores": resultado["errores"]
        }

    return {
        "tipo": "itinerario",
        "itinerario": resultado["itinerario"],
        "fuente": resultado["fuente"],
        "informacion_suficiente": (
            resultado["informacion_suficiente"]
        )
    }


# ============================================================
# FORMATEAR ITINERARIO
# ============================================================

def formatear_itinerario(
    itinerario
):

    partes = []

    partes.append(
        f"# Itinerario para "
        f"{itinerario['destino']}"
    )

    partes.append(
        f"**Duración:** "
        f"{itinerario['duracion_dias']} días  \n"
        f"**Presupuesto:** "
        f"{itinerario['presupuesto']}  \n"
        f"**Tipo de viajero:** "
        f"{itinerario['tipo_viajero']}"
    )

    if itinerario.get(
        "resumen"
    ):

        partes.append(
            f"### Resumen\n\n"
            f"{itinerario['resumen']}"
        )

    for dia in itinerario.get(
        "dias",
        []
    ):

        partes.append(
            f"## Día {dia['dia']} — "
            f"{dia['titulo']}"
        )

        for actividad in dia.get(
            "actividades",
            []
        ):

            partes.append(
                f"### {actividad['hora']} — "
                f"{actividad['actividad']}\n\n"
                f"{actividad['descripcion']}\n\n"
                f"**Motivo:** "
                f"{actividad['motivo']}"
            )

    presupuesto = itinerario.get(
        "presupuesto_estimado",
        {}
    )

    if presupuesto:

        partes.append(
            "## Presupuesto estimado"
        )

        for concepto, valor in (
            presupuesto.items()
        ):

            partes.append(
                f"- **{concepto.capitalize()}:** "
                f"{valor}"
            )

    recomendaciones = itinerario.get(
        "recomendaciones",
        []
    )

    if recomendaciones:

        partes.append(
            "## Recomendaciones"
        )

        for recomendacion in (
            recomendaciones
        ):

            partes.append(
                f"- {recomendacion}"
            )

    return "\n\n".join(
        partes
    )


# ============================================================
# INTERFAZ STREAMLIT
# ============================================================

st.title(
    "✈️ TravelGen AI"
)

st.subheader(
    "Planificador de viajes personalizado con IA"
)

st.write(
    "Cuéntame qué viaje quieres realizar y TravelGen AI "
    "generará un itinerario adaptado a tus preferencias."
)


# ============================================================
# SOLICITUD DEL USUARIO
# ============================================================

consulta = st.text_area(
    "Solicitud de viaje",
    placeholder=(
        "Ejemplo: Quiero viajar a Cartagena durante "
        "4 días con mi familia. Tengo un presupuesto "
        "de 2 millones de pesos y me interesa la "
        "cultura y la gastronomía."
    ),
    height=150
)


if st.button(
    "✈️ Generar itinerario",
    type="primary"
):

    if not consulta.strip():

        st.warning(
            "Por favor, cuéntame qué viaje quieres realizar."
        )

    else:

        with st.spinner(
            "Analizando tus preferencias y preparando "
            "el itinerario..."
        ):

            resultado = (
                procesar_solicitud_usuario(
                    consulta.strip()
                )
            )

        if resultado["tipo"] == "pregunta":

            st.info(
                resultado["mensaje"]
            )

        elif resultado["tipo"] == "error":

            st.error(
                resultado.get(
                    "mensaje",
                    "No fue posible generar el itinerario."
                )
            )

            errores = resultado.get(
                "errores",
                []
            )

            for error in errores:

                st.write(
                    f"- {error}"
                )

        elif resultado["tipo"] == "itinerario":

            itinerario = (
                resultado["itinerario"]
            )

            st.markdown(
                formatear_itinerario(
                    itinerario
                )
            )

            st.caption(
                f"Fuente principal del contexto: "
                f"{resultado['fuente']}"
            )


# ============================================================
# MODIFICACIÓN DEL ITINERARIO
# ============================================================

st.divider()

st.subheader(
    "🔄 Modificar itinerario"
)

st.write(
    "Si deseas ajustar el itinerario generado, "
    "puedes solicitar los cambios aquí."
)

modificacion = st.text_area(
    "¿Qué deseas cambiar?",
    placeholder=(
        "Ejemplo: Quiero más actividades de senderismo, "
        "menos gastronomía y que el segundo día incluya "
        "más ciclismo."
    ),
    height=120
)


if st.button(
    "🔄 Actualizar itinerario"
):

    if not consulta.strip():

        st.warning(
            "Primero debes generar un itinerario."
        )

    elif not modificacion.strip():

        st.warning(
            "Escribe qué deseas modificar."
        )

    else:

        nueva_consulta = f"""
Solicitud original del usuario:

{consulta.strip()}

El usuario desea modificar el itinerario anterior de la
siguiente manera:

{modificacion.strip()}

Genera nuevamente el itinerario teniendo en cuenta la
solicitud original y aplicando la modificación indicada.

Mantén las demás preferencias del usuario que no hayan sido
modificadas.
"""

        with st.spinner(
            "Actualizando el itinerario..."
        ):

            resultado_actualizado = (
                procesar_solicitud_usuario(
                    nueva_consulta
                )
            )

        if resultado_actualizado["tipo"] == "itinerario":

            st.markdown(
                formatear_itinerario(
                    resultado_actualizado["itinerario"]
                )
            )

        elif resultado_actualizado["tipo"] == "pregunta":

            st.info(
                resultado_actualizado["mensaje"]
            )

        else:

            st.error(
                resultado_actualizado.get(
                    "mensaje",
                    "No fue posible actualizar el itinerario."
                )
            )
