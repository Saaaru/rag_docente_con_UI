import streamlit as st
import uuid
import os
from datetime import datetime
import sys

# Configuración de la página - DEBE SER LA PRIMERA LLAMADA A STREAMLIT
st.set_page_config(
    page_title="Asistente Docente Chileno",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Importar las funciones necesarias del archivo principal
from app import (
    load_dotenv,
    VertexAIEmbeddings, 
    ChatVertexAI,
    Chroma,
    create_planning_agent,
    create_evaluation_agent,
    create_study_guide_agent,
    create_router_agent,
    format_and_save_conversation
)

# Cargar variables de entorno
dotenv_path = os.path.join(os.path.dirname(__file__), 'db', '.env')
load_dotenv(dotenv_path)

# Credenciales para usar VERTEX_AI
credentials_path = r"C:/Users/mfuen/OneDrive/Desktop/rag_docente_con_UI/db/gen-lang-client-0115469242-239dc466873d.json"
if not os.path.exists(credentials_path):
    st.error(f"No se encontró el archivo de credenciales en: {credentials_path}")
    st.stop()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

# Para tracear con langsmith
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
if not LANGSMITH_API_KEY:
    st.error("La variable LANGSMITH_API_KEY no está definida en el archivo .env en la carpeta db/")
    st.stop()
os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY
os.environ["LANGSMITH_TRACING"] = "true"

# Modificar la función initialize_resources para que no muestre spinner
@st.cache_resource(show_spinner=False)  # Removemos el spinner de la cache
def initialize_resources():
    """
    Inicializa los recursos necesarios para el sistema.
    No debe contener elementos de UI de Streamlit.
    """
    llm = ChatVertexAI(
        model_name="gemini-1.5-flash",
        temperature=0.5,
        max_output_tokens=8192,
        top_p=0.95,
        top_k=40
    )

    embeddings = VertexAIEmbeddings(model_name="text-multilingual-embedding-002")
    
    collection_name = "pdf-rag-chroma"
    persist_directory = f"./{collection_name}"
    
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name=collection_name
    )
    
    planning_agent = create_planning_agent(llm, vectorstore)
    evaluation_agent = create_evaluation_agent(llm, vectorstore)
    study_guide_agent = create_study_guide_agent(llm, vectorstore)
    router = create_router_agent(llm, planning_agent, evaluation_agent, study_guide_agent)
    
    return router, llm, vectorstore

# Función para mostrar mensajes con formato adecuado
def display_message(role, content, avatar=None):
    with st.chat_message(role, avatar=avatar):
        st.markdown(content)

# Inicializar historial si no existe
def initialize_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())[:8]
    if "pending_request" not in st.session_state:
        st.session_state.pending_request = False
    if "last_query" not in st.session_state:
        st.session_state.last_query = ""
    if "asignatura" not in st.session_state:
        st.session_state.asignatura = None
    if "nivel" not in st.session_state:
        st.session_state.nivel = None
    if "tipo" not in st.session_state:
        st.session_state.tipo = None

def main():
    # Ya no incluimos st.set_page_config() aquí
    initialize_session()
    
    # Verificaciones de archivos y credenciales
    if not os.path.exists(credentials_path):
        st.error(f"No se encontró el archivo de credenciales en: {credentials_path}")
        st.stop()

    collection_name = "pdf-rag-chroma"
    persist_directory = f"./{collection_name}"
    
    if not os.path.exists(persist_directory):
        st.error("No se encontró la base de datos. Por favor, ejecuta app.py primero para crearla.")
        st.stop()

    # Sidebar con información
    with st.sidebar:
        st.title("Asistente Docente Chileno")
        st.markdown("---")
        st.markdown("### 📝 Tipo de contenidos")
        st.markdown("- **Planificaciones educativas** (planes de clase, anuales, etc.)")
        st.markdown("- **Evaluaciones** (pruebas, exámenes, etc.)")
        st.markdown("- **Guías de estudio** (material para estudiantes)")
        st.markdown("---")
        st.markdown(f"🔑 ID de sesión: `{st.session_state.thread_id}`")
        
        # Botón para descargar la conversación actual
        if st.button("💾 Descargar conversación"):
            # Generar contenido markdown de la conversación
            markdown_content = "# Conversación con Asistente Docente\n\n"
            markdown_content += f"Fecha: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n\n"
            markdown_content += f"ID de sesión: {st.session_state.thread_id}\n\n"
            markdown_content += "---\n\n"
            
            for msg in st.session_state.messages:
                role = "Usuario" if msg["role"] == "user" else "Asistente"
                markdown_content += f"## {role}\n\n{msg['content']}\n\n---\n\n"
            
            # Crear archivo temporal para descarga
            filename = f"conversacion_{st.session_state.thread_id}.md"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(markdown_content)
            
            # Ofrecer descarga
            with open(filename, "r", encoding="utf-8") as f:
                st.download_button(
                    label="📥 Descargar archivo Markdown",
                    data=f,
                    file_name=filename,
                    mime="text/markdown"
                )
        
        st.markdown("---")
        st.markdown("### 👋 Iniciar nueva conversación")
        if st.button("Nueva conversación"):
            st.session_state.messages = []
            st.session_state.thread_id = str(uuid.uuid4())[:8]
            st.session_state.pending_request = False
            st.session_state.last_query = ""
            st.session_state.asignatura = None
            st.session_state.nivel = None
            st.session_state.tipo = None
            st.rerun()
    
    # Título principal
    st.title("💬 Chat con Asistente Docente")
    st.markdown("Consulta sobre planificaciones, evaluaciones o guías de estudio para el sistema educativo chileno.")
    
    # Inicializar recursos silenciosamente
    router, llm, vectorstore = initialize_resources()
    
    # Contenedor para los mensajes
    chat_container = st.container()
    
    with chat_container:
        # Mostrar mensajes anteriores
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Mensaje de bienvenida si no hay mensajes
        if not st.session_state.messages:
            with st.chat_message("assistant"):
                st.markdown("""
                👋 ¡Hola! Soy tu asistente para crear contenido educativo para el sistema chileno.
                
                Puedo ayudarte con:
                - **Planificaciones** para cualquier nivel y asignatura
                - **Evaluaciones** alineadas con el currículum nacional
                - **Guías de estudio** para tus estudiantes
                
                ¿En qué puedo ayudarte hoy?
                """)
    
    # Campo de entrada de texto
    user_input = st.chat_input("Escribe tu consulta aquí...")
    
    # Procesar el input del usuario
    if user_input:
        # Mostrar mensaje del usuario inmediatamente
        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_input)
        
        # Guardar mensaje del usuario en el historial
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Crear un contenedor específico para la respuesta del asistente
        with chat_container:
            with st.chat_message("assistant"):
                try:
                    # Pequeño indicador de "Procesando..."
                    processing_placeholder = st.empty()
                    processing_placeholder.markdown("*Procesando tu solicitud...*")
                    
                    # Procesar la solicitud
                    if st.session_state.pending_request:
                        if not st.session_state.asignatura:
                            st.session_state.asignatura = user_input
                            response, needs_info, info, tipo = router(
                                st.session_state.last_query, 
                                st.session_state.asignatura, 
                                st.session_state.nivel
                            )
                        elif not st.session_state.nivel:
                            st.session_state.nivel = user_input
                            response, needs_info, info, tipo = router(
                                st.session_state.last_query, 
                                st.session_state.asignatura, 
                                st.session_state.nivel
                            )
                    else:
                        # Nueva solicitud
                        response, needs_info, info, tipo = router(user_input)
                    
                    # Eliminar el indicador de "Procesando..."
                    processing_placeholder.empty()
                    
                    # Mostrar la respuesta
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    # Actualizar el estado según la respuesta
                    if needs_info:
                        st.session_state.pending_request = True
                        st.session_state.last_query = user_input
                        st.session_state.asignatura = info.get("asignatura")
                        st.session_state.nivel = info.get("nivel")
                        st.session_state.tipo = tipo
                    else:
                        format_and_save_conversation(user_input, response, st.session_state.thread_id)
                        # Reiniciar el estado
                        st.session_state.pending_request = False
                        st.session_state.last_query = ""
                        st.session_state.asignatura = None
                        st.session_state.nivel = None
                        st.session_state.tipo = None
                
                except Exception as e:
                    error_message = f"❌ Lo siento, ocurrió un error: {str(e)}\n\nPor favor, intenta reformular tu solicitud."
                    st.markdown(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
                    # Reiniciar el estado en caso de error
                    st.session_state.pending_request = False
                    st.session_state.last_query = ""
                    st.session_state.asignatura = None
                    st.session_state.nivel = None
                    st.session_state.tipo = None

if __name__ == "__main__":
    main() 